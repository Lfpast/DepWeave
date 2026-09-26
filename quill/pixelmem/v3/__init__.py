"""PixelMem V3 -- summary-aware retrieval with evidence bundles.

Provides the full V3 pipeline: lane-classified writes, algebra-driven
reads, cross-lane routing, plan caching, and structured metrics.

Quick start::

    from pixelmem.shard_manager import ShardManager
    from pixelmem.v3 import (
        WritePipeline, WriteResult,
        SummaryStore, SummaryBuilder, Canonicalizer,
        CrossLaneRouter, PlanCache, Metrics,
    )

    mgr = ShardManager("./store")
    store = SummaryStore()
    canon = Canonicalizer(mgr)
    writer = WritePipeline(mgr, store, canonicalizer=canon)
    result = writer.store("Alice works at Acme Corp.")
"""

# -- Read pipeline ---------------------------------------------------------
from pixelmem.v3.read_pipeline import ReadPipeline

# -- Query planner ---------------------------------------------------------
from pixelmem.v3.query_planner import QueryPlanner, RetrievalPlan, QueryMode

# -- Write pipeline --------------------------------------------------------
from pixelmem.v3.write_pipeline import WritePipeline, WriteResult

# -- Retrieval algebra and evidence ----------------------------------------
from pixelmem.v3.evidence_bundle import Fact
from pixelmem.v3.evidence_bundle import EvidenceBundle, Evidence

# -- Summary layer ---------------------------------------------------------
from pixelmem.v3.summary_objects import EntitySummary, SourceSummary, TaskSummary
from pixelmem.v3.summary_store import SummaryStore
from pixelmem.v3.summary_builder import SummaryBuilder

# -- Canonicalization ------------------------------------------------------
from pixelmem.v3.canonicalizer import Canonicalizer

# -- Lane classification ---------------------------------------------------
from pixelmem.v3.lane_classifier import ContentLane, classify_lane

# -- Cross-lane routing ----------------------------------------------------
from pixelmem.v3.cross_lane import CrossLaneRouter

# -- Caching ---------------------------------------------------------------
from pixelmem.v3.cache import PlanCache

# -- Metrics ---------------------------------------------------------------
from pixelmem.v3.metrics import Metrics

__all__ = [
    # Read pipeline
    "ReadPipeline",
    # Query planner
    "QueryPlanner",
    "RetrievalPlan",
    "QueryMode",
    # Write pipeline
    "WritePipeline",
    "WriteResult",
    # Retrieval algebra / evidence
    "Fact",
    "EvidenceBundle",
    "Evidence",
    # Summaries
    "EntitySummary",
    "SourceSummary",
    "TaskSummary",
    "SummaryStore",
    "SummaryBuilder",
    # Canonicalization
    "Canonicalizer",
    # Lane classification
    "ContentLane",
    "classify_lane",
    # Cross-lane
    "CrossLaneRouter",
    # Cache
    "PlanCache",
    # Metrics
    "Metrics",
]
