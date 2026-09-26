"""PixelMem: Pixel-Encoded Knowledge Graphs as Vision-Native Memory for LLMs."""

__all__ = ["PixelMemUnit", "ShardManager", "encode_text", "decode_query"]


def __getattr__(name):
    # Lazy re-exports (PEP 562): importing a submodule (e.g. pixelmem.v6.*) must
    # not pull in the pixel encoder/decoder and their numpy/Pillow deps. These
    # names still resolve on first access for v1–v5 callers.
    if name == "PixelMemUnit":
        from pixelmem.memory import PixelMemUnit
        return PixelMemUnit
    if name == "ShardManager":
        from pixelmem.shard_manager import ShardManager
        return ShardManager
    if name == "encode_text":
        from pixelmem.encoder import encode_text
        return encode_text
    if name == "decode_query":
        from pixelmem.decoder import decode_query
        return decode_query
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
