"""PixelMem v2 — tool-routed retrieval with compact payloads.

Reduces token usage by:
  1. Compact index replaces verbose prose summaries
  2. BM25 lexical router replaces LLM entity selection
  3. Query mode classifier routes to specialized mini-tools
  4. Compact payloads use interned string tables
  5. Extraction cascade avoids LLM for ~70% of writes
  6. Content lanes skip chit-chat entirely
  7. Retrieval plan cache avoids re-routing similar queries

Usage:
    from pixelmem.v2 import V2ReadPipeline, V2WritePipeline
    from pixelmem.shard_manager import ShardManager

    mgr = ShardManager.load("./store")
    reader = V2ReadPipeline(mgr)
    writer = V2WritePipeline(mgr, reader)

    # Write
    writer.store("I work at Acme Corp since 2023.")

    # Read
    payload = reader.query("Where do I work?")
    print(payload.to_text())

    # Stats
    print(reader.stats())
"""

from pixelmem.v2.pipeline import V2ReadPipeline, V2WritePipeline
from pixelmem.v2.compact_index import CompactIndex, build_compact_index
from pixelmem.v2.lexical_router import LexicalRouter
from pixelmem.v2.query_classifier import classify, QueryMode
from pixelmem.v2.compact_payload import RetrievalPayload
from pixelmem.v2.extraction_cascade import ExtractionCascade
from pixelmem.v2.content_lanes import classify_content, ContentLane
from pixelmem.v2.retrieval_cache import RetrievalPlanCache
from pixelmem.v2.instrumentation import Instrumentor

__all__ = [
    "V2ReadPipeline", "V2WritePipeline",
    "CompactIndex", "build_compact_index",
    "LexicalRouter",
    "classify", "QueryMode",
    "RetrievalPayload",
    "ExtractionCascade",
    "classify_content", "ContentLane",
    "RetrievalPlanCache",
    "Instrumentor",
]
