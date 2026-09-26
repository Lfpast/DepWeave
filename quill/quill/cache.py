"""Pixel-encoded persistent cache for V5 primitive extraction.

On first query against a document set, the extractor runs, primitives are
encoded into PNG matrices via ``ShardManager``, and a provenance sidecar
is written. On subsequent queries against the same document set, the
primitives are reconstructed from the PNG store + sidecar.

Why this sits below the extractor:
    extractor.extract(docs) ── expensive on big repos (parsing, regex)
    pixel store write        ── cheap (KB-scale PNGs per shard)
    pixel store read         ── fast (mmap + index lookup)

Expected wins:
    RepoQA — same repo queried once per needle (10 needles/repo).
    SWE-Bench — same repo queried for many issues.

The cache does NOT change the extractor's output — it's a pure
pass-through layer. Primitives out on a hit equal primitives out on a
miss + save.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from pixelmem.shard_manager import ShardManager
from pixelmem.triple_extractor import Triple

from quill.plugins import Extractor
from quill.types import Primitive


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    total_wallclock_extract_s: float = 0.0
    total_wallclock_load_s: float = 0.0
    total_wallclock_save_s: float = 0.0
    bytes_written: int = 0
    primitives_cached: int = 0
    cache_entries: int = 0
    entry_sizes: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(self.hits + self.misses, 1),
            "total_wallclock_extract_s": self.total_wallclock_extract_s,
            "total_wallclock_load_s": self.total_wallclock_load_s,
            "total_wallclock_save_s": self.total_wallclock_save_s,
            "bytes_written": self.bytes_written,
            "primitives_cached": self.primitives_cached,
            "cache_entries": self.cache_entries,
            "avg_entry_size_bytes": (
                sum(self.entry_sizes) / len(self.entry_sizes)
                if self.entry_sizes else 0
            ),
        }


def _hash_documents(documents: dict[str, str]) -> str:
    """Stable 16-hex-char hash of a document set's contents."""
    h = hashlib.sha256()
    for key in sorted(documents):
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        value = documents[key]
        if not isinstance(value, str):
            value = str(value)
        h.update(value.encode("utf-8", errors="replace"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


def _dir_size_bytes(p: Path) -> int:
    total = 0
    for child in p.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


class CachedExtractor(Extractor):
    """Wraps an inner Extractor with a PixelMem-backed cache.

    Delegates ``extract(documents)`` to the inner extractor on cache miss,
    writes the result to PNG, and reads it back on hit.
    """

    def __init__(self, inner: Extractor, cache: "PixelMemCache") -> None:
        self._inner = inner
        self._cache = cache
        self.last_hit = False

    def extract(self, documents: dict[str, str], **kwargs) -> list[Primitive]:
        primitives, self.last_hit = self._cache.get_or_extract_with_status(
            documents, self._inner, **kwargs
        )
        return primitives


class PixelMemCache:
    """Persistent PNG-backed cache for V5 primitives.

    Usage::

        cache = PixelMemCache("/tmp/v5_cache", shard_size=64)
        cached_extractor = cache.wrap(my_extractor)
        # or directly:
        primitives = cache.get_or_extract(documents, my_extractor)
    """

    def __init__(self, cache_root: str | Path, shard_size: int = 64) -> None:
        self._root = Path(cache_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._shard_size = shard_size
        self._stats = CacheStats()
        # Per-key locks so concurrent queries against the same doc set
        # serialize at the cache layer; different keys still parallelize.
        self._lock_registry_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._stats_lock = threading.Lock()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._lock_registry_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wrap(self, extractor: Extractor) -> CachedExtractor:
        """Return an Extractor that reads from this cache."""
        return CachedExtractor(extractor, self)

    def get_or_extract(
        self,
        documents: dict[str, str],
        extractor: Extractor,
        **kwargs,
    ) -> list[Primitive]:
        primitives, _ = self.get_or_extract_with_status(documents, extractor, **kwargs)
        return primitives

    def get_or_extract_with_status(
        self,
        documents: dict[str, str],
        extractor: Extractor,
        **kwargs,
    ) -> tuple[list[Primitive], bool]:
        key = _hash_documents(documents)
        entry_dir = self._root / key

        # Serialize all operations on this key. Different keys run in
        # parallel. Inside the critical section: re-check for hit (some
        # other thread may have finished saving while we waited).
        with self._lock_for(key):
            if (entry_dir / "index.json").exists() and (entry_dir / "provenance.json.gz").exists():
                t0 = time.perf_counter()
                primitives = self._load_from_pixels(entry_dir)
                with self._stats_lock:
                    self._stats.total_wallclock_load_s += time.perf_counter() - t0
                    self._stats.hits += 1
                return primitives, True

            # Miss — run extractor, save, return.
            with self._stats_lock:
                self._stats.misses += 1

            t0 = time.perf_counter()
            primitives = list(extractor.extract(documents, **kwargs))
            extract_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            self._save_to_pixels(entry_dir, primitives)
            save_s = time.perf_counter() - t1

            size = _dir_size_bytes(entry_dir)
            with self._stats_lock:
                self._stats.total_wallclock_extract_s += extract_s
                self._stats.total_wallclock_save_s += save_s
                self._stats.bytes_written += size
                self._stats.primitives_cached += len(primitives)
                self._stats.cache_entries += 1
                self._stats.entry_sizes.append(size)
            return primitives, False

    def clear(self) -> None:
        if self._root.exists():
            shutil.rmtree(self._root)
            self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internals: save / load
    # ------------------------------------------------------------------

    def _save_to_pixels(self, entry_dir: Path, primitives: list[Primitive]) -> None:
        """Write primitives to PNG store + provenance sidecar.

        The pixel store encodes (subject, relation, object, condition). The
        sidecar keeps the provenance dicts (which aren't part of the triple
        schema) and preserves primitive ORDER (so the load path can
        realign them with the decoded triples).
        """
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        entry_dir.mkdir(parents=True, exist_ok=True)

        # 1) Pixel-encode the quadruples via ShardManager.
        # `sparse=True` uses a global color/entity registry instead of
        # per-shard JSON maps — across 8 shards this cuts ~100 KB of
        # duplicated metadata.
        triples = [
            Triple(p.subject, p.relation, p.object, p.condition or "")
            for p in primitives
        ]
        if triples:
            mgr = ShardManager(entry_dir, shard_size=self._shard_size)
            # Encode in chunks so any single shard stays under its density cap.
            chunk = 10
            for i in range(0, len(triples), chunk):
                mgr.encode("", triples=triples[i : i + chunk])
            mgr.save(sparse=True)
        else:
            # Still create an index.json so the hit-check finds the entry.
            with open(entry_dir / "index.json", "w") as f:
                json.dump({"shard_size": self._shard_size,
                           "density_target": 0.0,
                           "shards": [], "format": "sparse"}, f)

        # 2) Sidecar: a gzipped JSON list with one row per primitive, in
        # the same order as the pixel encoding. This is what load uses to
        # reconstruct V5 Primitives with their provenance intact. We gzip
        # because provenance dicts dominate entry size otherwise (seen in
        # exp40: raw JSON sidecar was 87% of a 1MB entry; gzipped is ~10%).
        #
        # We do NOT try to decode the PNG back into triples (lossy round-
        # trip because entity IDs are per-shard). The pixel store is the
        # fast-load structural index; the sidecar keeps the exact V5
        # semantics.
        sidecar = [
            {
                "subject": p.subject,
                "relation": p.relation,
                "object": p.object,
                "condition": p.condition or "",
                "provenance": p.provenance,
            }
            for p in primitives
        ]
        raw = json.dumps(sidecar, default=str).encode("utf-8")
        with gzip.open(entry_dir / "provenance.json.gz", "wb", compresslevel=6) as f:
            f.write(raw)

    def _load_from_pixels(self, entry_dir: Path) -> list[Primitive]:
        """Reconstruct primitives from the PNG store + sidecar.

        We rely on the sidecar for primitive ordering + provenance. The
        pixel store is verified to exist (index.json) so callers can trust
        the cache ran the encode path even though we don't re-decode the
        pixels here.

        Accepts either the gzipped sidecar (preferred) or legacy raw JSON.
        """
        # Prove the pixel store is intact by loading it (cheap).
        ShardManager.load(entry_dir)
        gz_path = entry_dir / "provenance.json.gz"
        if gz_path.exists():
            with gzip.open(gz_path, "rb") as f:
                rows = json.loads(f.read())
        else:
            with open(entry_dir / "provenance.json") as f:
                rows = json.load(f)
        return [
            Primitive(
                subject=r["subject"],
                relation=r["relation"],
                object=r["object"],
                condition=r.get("condition", ""),
                provenance=r.get("provenance"),
            )
            for r in rows
        ]


__all__ = ["PixelMemCache", "CachedExtractor", "CacheStats"]
