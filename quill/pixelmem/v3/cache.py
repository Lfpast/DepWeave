"""Query-plan result cache for PixelMem V3.

Caches EvidenceBundle results keyed by normalized query strings.
Supports TTL-based expiry, LRU eviction, and targeted invalidation
when specific entities are updated.

Typical usage::

    from pixelmem.v3.cache import PlanCache

    cache = PlanCache(max_size=256, ttl_seconds=300)
    cache.put("Where does Alice work?", bundle)
    hit = cache.get("where does alice work?")  # normalised match
    cache.invalidate_entity("alice")            # evict stale entries
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from pixelmem.v3.evidence_bundle import EvidenceBundle


# ---------------------------------------------------------------------------
# Stopwords for query normalization
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must",
    "i", "me", "my", "we", "our", "you", "your",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "so", "if", "then",
    "what", "where", "when", "who", "how", "which", "that", "this",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached query result.

    Attributes:
        key: The SHA-256 hash of the normalised query.
        bundle: The cached EvidenceBundle.
        timestamp: Unix timestamp when the entry was created.
        hit_count: Number of times this entry has been served.
        entities_involved: Frozen set of entity names touched by the bundle.
        plan_hash: Optional hash of the retrieval plan that produced
            the bundle (for debugging / correlation).
    """
    key: str = ""
    bundle: EvidenceBundle = field(default_factory=EvidenceBundle)
    timestamp: float = 0.0
    hit_count: int = 0
    entities_involved: frozenset[str] = field(default_factory=frozenset)
    plan_hash: str = ""


# ---------------------------------------------------------------------------
# PlanCache
# ---------------------------------------------------------------------------

class PlanCache:
    """LRU + TTL cache for EvidenceBundle results.

    Parameters
    ----------
    max_size : int
        Maximum number of entries.  When exceeded the least-recently-used
        entry is evicted.
    ttl_seconds : float
        Time-to-live in seconds.  Entries older than this are considered
        stale and evicted on access.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ------------------------------------------------------------------
    # Query normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize a query for cache lookup.

        Lowercases, extracts alphanumeric tokens, removes stopwords,
        and sorts the remaining tokens alphabetically.
        """
        tokens = _TOKEN_RE.findall(query.lower())
        filtered = [t for t in tokens if t not in _STOPWORDS]
        filtered.sort()
        return " ".join(filtered)

    @staticmethod
    def _hash_query(normalized: str) -> str:
        """SHA-256 hash of a normalised query string."""
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Get / Put
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[EvidenceBundle]:
        """Look up a cached bundle for *query*.

        Returns ``None`` on miss or TTL expiry.  On hit, the entry is
        bumped to the most-recently-used position.
        """
        key = self._hash_query(self._normalize_query(query))

        if key not in self._store:
            self._misses += 1
            return None

        entry = self._store[key]

        # TTL check
        if (time.time() - entry.timestamp) > self._ttl:
            del self._store[key]
            self._misses += 1
            self._evictions += 1
            return None

        # Bump LRU
        self._store.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1
        return entry.bundle

    def put(
        self,
        query: str,
        bundle: EvidenceBundle,
        plan_hash: str = "",
    ) -> None:
        """Cache *bundle* under *query*.

        If the cache is at capacity, the least-recently-used entry is
        evicted first.
        """
        normalized = self._normalize_query(query)
        key = self._hash_query(normalized)

        # Collect entities from the bundle's evidence
        entities: set[str] = set()
        for ev in bundle.evidences:
            entities.add(ev.fact.subject.lower())
            entities.add(ev.fact.object.lower())

        entry = CacheEntry(
            key=key,
            bundle=bundle,
            timestamp=time.time(),
            hit_count=0,
            entities_involved=frozenset(entities),
            plan_hash=plan_hash,
        )

        # Evict LRU if over capacity
        while len(self._store) >= self._max_size:
            self._store.popitem(last=False)
            self._evictions += 1

        self._store[key] = entry
        self._store.move_to_end(key)

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate_entity(self, entity: str) -> int:
        """Evict all entries whose bundle touches *entity*.

        Parameters
        ----------
        entity : str
            Entity name (case-insensitive).

        Returns
        -------
        int
            Number of entries evicted.
        """
        target = entity.strip().lower()
        keys_to_remove: list[str] = []

        for key, entry in self._store.items():
            if target in entry.entities_involved:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._store[key]

        self._evictions += len(keys_to_remove)
        return len(keys_to_remove)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        count = len(self._store)
        self._store.clear()
        self._evictions += count

    # ------------------------------------------------------------------
    # Properties and stats
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._store)

    def stats(self) -> dict:
        """Return cache performance statistics.

        Returns
        -------
        dict
            Keys: ``size``, ``max_size``, ``ttl_seconds``, ``hits``,
            ``misses``, ``evictions``, ``hit_rate``.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(hit_rate, 4),
        }
