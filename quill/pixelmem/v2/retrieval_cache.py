"""Retrieval Plan Cache — cache retrieval plans, not just answers.

Normalizes query signatures so "Where does Alice work?" and
"where does alice work" hit the same cache entry.

Invalidates when relevant entities are updated.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from pixelmem.v2.compact_payload import RetrievalPayload


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "would", "should", "about", "and", "but", "or",
    "for", "in", "on", "at", "to", "of", "with", "that", "this",
})


@dataclass
class CacheEntry:
    key: str
    payload: RetrievalPayload
    timestamp: float
    hit_count: int = 0
    entities_involved: frozenset[str] = field(default_factory=frozenset)
    query_mode: str = ""


class RetrievalPlanCache:
    """LRU cache with entity-based invalidation."""

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _normalize_query(self, query: str) -> str:
        """Normalize query for cache key: lowercase, sort tokens, strip stops."""
        tokens = re.findall(r'\w+', query.lower())
        meaningful = sorted(t for t in tokens if t not in _STOP_WORDS and len(t) > 1)
        return " ".join(meaningful)

    def _hash_query(self, query: str) -> str:
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[RetrievalPayload]:
        """Look up a cached retrieval plan."""
        key = self._hash_query(query)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        if time.time() - entry.timestamp > self.ttl:
            del self._cache[key]
            self._misses += 1
            return None

        # Cache hit — move to end (LRU)
        self._cache.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1
        return entry.payload

    def put(
        self,
        query: str,
        payload: RetrievalPayload,
        query_mode: str = "",
    ) -> None:
        """Store a retrieval plan."""
        key = self._hash_query(query)

        # Extract entities for invalidation tracking
        entities = frozenset(payload.entity_dict)

        entry = CacheEntry(
            key=key,
            payload=payload,
            timestamp=time.time(),
            entities_involved=entities,
            query_mode=query_mode,
        )

        # Evict if over max size
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
            self._evictions += 1

        self._cache[key] = entry

    def invalidate_entity(self, entity: str) -> int:
        """Evict all entries whose payload contains the entity."""
        canonical = entity.strip().lower()
        to_remove = [
            key for key, entry in self._cache.items()
            if canonical in entry.entities_involved
        ]
        for key in to_remove:
            del self._cache[key]
        return len(to_remove)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, total) * 100, 1),
            "evictions": self._evictions,
        }
