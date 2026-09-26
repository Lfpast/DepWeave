"""PixelMem-backed cache layer for V5 pipelines.

Writes extracted primitives to PNG matrices via ``ShardManager`` plus a
small provenance sidecar, then restores them on subsequent queries against
the same document set. This re-introduces the pixel encoding that V5's
plugin host had bypassed.
"""

from quill.cache.pixel_cache import PixelMemCache, CacheStats

__all__ = ["PixelMemCache", "CacheStats"]
