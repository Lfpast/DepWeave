"""V2 Pipeline — integrates all components into cohesive read/write paths.

READ PATH:
  query → query_classifier → lexical_router → relation_tools → compact_payload
  → optional cache → LLM final answer

WRITE PATH:
  raw text → content_lanes → extraction_cascade → confidence_gate
  → encode into shard → invalidate cache
"""

from __future__ import annotations

import time
from typing import Optional

from pixelmem.shard_manager import ShardManager
from pixelmem.encoder import encode_text

from pixelmem.v2.compact_index import CompactIndex, build_compact_index
from pixelmem.v2.lexical_router import LexicalRouter, RouteResult
from pixelmem.v2.query_classifier import (
    classify, ClassificationResult, QueryMode,
)
from pixelmem.v2.relation_tools import (
    find_rel, find_between, find_recent, count_relations,
    resolve_alias, date_diff, sum_values, FactEntry,
)
from pixelmem.v2.compact_payload import RetrievalPayload, merge_payloads
from pixelmem.v2.extraction_cascade import ExtractionCascade, ExtractionResult
from pixelmem.v2.content_lanes import (
    classify_content, lane_config, ContentLane, LaneClassification,
)
from pixelmem.v2.retrieval_cache import RetrievalPlanCache
from pixelmem.v2.instrumentation import Instrumentor, MetricEvent


def _rerank_payload(
    payload: RetrievalPayload,
    query: str,
    mgr: "ShardManager" = None,
    max_chunks: int = 5,
) -> RetrievalPayload:
    """Summary-based chunk selection — read summaries, not all facts.

    Instead of reading 40+ facts then filtering:
    1. Build one-line summary per chunk (~10 tokens each)
    2. BM25 score summaries against query (free, instant)
    3. Return FULL facts from only the top-N chunks

    For 10 chunks with 50 total facts:
      Old: LLM reads all 50 facts (~500 tokens) to rerank
      New: BM25 reads 10 summaries (~100 tokens), returns ~15 focused facts
    """
    if mgr is None:
        return payload

    from rank_bm25 import BM25Okapi
    from pixelmem.decoder import reconstruct_chunks
    from pixelmem.v2.relation_tools import FactEntry
    import re, numpy as np

    # Build per-chunk summaries from tapes
    all_chunks = []
    for si, shard in enumerate(mgr.shards):
        chunks = reconstruct_chunks(shard)
        for ci, chunk in enumerate(chunks):
            if not chunk:
                continue
            entities = sorted(set(t.subject for t in chunk) | set(t.object for t in chunk))
            relations = sorted(set(t.relation for t in chunk))
            conditions = [t.condition for t in chunk if t.condition]
            summary = f"{', '.join(entities[:6])} | {', '.join(relations[:4])}"
            if conditions:
                summary += f" | {', '.join(conditions[:2])}"
            all_chunks.append({
                "summary": summary,
                "facts": chunk,
                "shard_idx": si,
            })

    if not all_chunks:
        return payload

    # BM25 over compact summaries (free, instant, ~10 tokens per summary)
    summaries = [c["summary"] for c in all_chunks]
    docs = [re.findall(r'\w+', s.lower()) for s in summaries]
    if not docs or not any(docs):
        return payload
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(re.findall(r'\w+', query.lower()))
    top_idx = np.argsort(scores)[-max_chunks:][::-1]

    # Return full facts from selected chunks only
    new_payload = RetrievalPayload()
    for idx in top_idx:
        if scores[idx] <= 0:
            continue
        for t in all_chunks[idx]["facts"]:
            new_payload.add_facts([FactEntry(
                t.subject, t.relation, t.object, t.condition,
                all_chunks[idx]["shard_idx"]
            )])

    new_payload.deduplicate()
    new_payload.query_mode = payload.query_mode
    new_payload.tool_calls = payload.tool_calls
    new_payload.sources = payload.sources
    return new_payload


class V2ReadPipeline:
    """Unified read path: classify → route → tool scan → compact payload.

    Replaces the pattern of:
      1. LLM reads full entity list (expensive)
      2. LLM picks entities (unreliable at scale)
      3. Tool scans (cheap)
      4. LLM answers (expensive context)

    With:
      1. Heuristic classifies query mode (free)
      2. BM25 routes to records (free)
      3. Direct tool scan of top entities (cheap)
      4. Compact payload to LLM (minimal tokens)
    """

    def __init__(
        self,
        mgr: ShardManager,
        cache_size: int = 256,
        cache_ttl: float = 300,
    ):
        self.mgr = mgr
        self.index = build_compact_index(mgr)
        self.router = LexicalRouter(self.index)
        self.cache = RetrievalPlanCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self.instrumentor = Instrumentor()

    def query(
        self,
        query_text: str,
        budget_tokens: int = 2000,
        use_cache: bool = True,
    ) -> RetrievalPayload:
        """Full read pipeline: classify → route → scan → payload."""

        with self.instrumentor.timed("read") as event:
            # Step 0: Cache check
            if use_cache:
                cached = self.cache.get(query_text)
                if cached is not None:
                    event.cache_hit = True
                    event.n_facts = cached.n_facts
                    event.query_mode = cached.query_mode
                    return cached

            # Step 1: Classify query mode
            classification = classify(query_text)
            event.query_mode = classification.mode.value

            # Step 2: Route via lexical scoring
            route = self.router.route(query_text)

            # Step 3: Execute mode-specific retrieval
            payload = self._execute_by_mode(
                classification, route, budget_tokens, query_text
            )
            payload.query_mode = classification.mode.value

            # Step 4: Populate event metrics
            event.n_facts = payload.n_facts
            event.n_entities_scanned = payload.n_entities
            event.n_shards_touched = len(payload.sources)
            event.n_tool_calls = payload.tool_calls
            event.tokens_in = payload.estimate_tokens()

            # Step 5: Cache the result
            if use_cache:
                self.cache.put(
                    query_text, payload,
                    query_mode=classification.mode.value,
                )

        return payload

    def _execute_by_mode(
        self,
        classification: ClassificationResult,
        route: RouteResult,
        budget_tokens: int,
        query_text: str = "",
    ) -> RetrievalPayload:
        """Dispatch to mode-specific retrieval strategy."""
        mode = classification.mode
        payload = RetrievalPayload()

        # Get entities to scan — prioritize exact matches from classifier
        entities = list(classification.extracted_entities)
        # Add top entities from router
        for ent in route.top_entities(8):
            if ent not in entities:
                entities.append(ent)

        if mode == QueryMode.RELATION_QUERY:
            p = self._mode_relation(entities, payload)
        elif mode == QueryMode.TEMPORAL_DIFF:
            p = self._mode_temporal_diff(entities, payload)
        elif mode == QueryMode.COUNT_QUERY:
            p = self._mode_count(entities, payload)
        elif mode == QueryMode.TEMPORAL_LOOKUP:
            p = self._mode_temporal(entities, payload)
        else:
            p = self._mode_general(entities, payload, budget_tokens, query_text)

        # Re-rank only if way too many facts and reranker is available
        # Disabled by default — reranker can throw away entity-matched facts
        # TODO: improve reranker to preserve entity-matched results
        return p

    def _mode_general(
        self,
        entities: list[str],
        payload: RetrievalPayload,
        budget_tokens: int,
        query_text: str = "",
    ) -> RetrievalPayload:
        """General fact lookup — scan top entities with decreasing limits."""
        max_entities = min(len(entities), budget_tokens // 30)
        for i, ent in enumerate(entities[:max_entities]):
            # Primary entities get more facts, secondary get fewer
            limit = 50 if i < 2 else 10
            facts = find_rel(self.mgr, ent, limit=limit)
            payload.add_facts(facts)
            payload.tool_calls += 1
        payload.deduplicate()

        return payload

    def _mode_relation(
        self,
        entities: list[str],
        payload: RetrievalPayload,
    ) -> RetrievalPayload:
        """Find relations between specific entities."""
        if len(entities) >= 2:
            # Check pairwise first — most targeted
            for i in range(min(len(entities), 3)):
                for j in range(i + 1, min(len(entities), 3)):
                    facts = find_between(self.mgr, entities[i], entities[j])
                    payload.add_facts(facts)
                    payload.tool_calls += 1
            # Deduplicate before adding broader scans
            payload.deduplicate()
        # Scan individual entities only if pairwise found nothing
        if payload.n_facts == 0:
            for ent in entities[:4]:
                facts = find_rel(self.mgr, ent, limit=10)
                payload.add_facts(facts)
                payload.tool_calls += 1
            payload.deduplicate()
        return payload

    def _mode_temporal_diff(
        self,
        entities: list[str],
        payload: RetrievalPayload,
    ) -> RetrievalPayload:
        """Temporal difference — find dates and compute diff."""
        # Scan entities for temporal conditions
        for ent in entities[:5]:
            facts = find_recent(self.mgr, ent)
            payload.add_facts(facts)
            payload.tool_calls += 1
        payload.deduplicate()
        return payload

    def _mode_temporal(
        self,
        entities: list[str],
        payload: RetrievalPayload,
    ) -> RetrievalPayload:
        """Temporal lookup — find time-annotated facts."""
        for ent in entities[:5]:
            facts = find_recent(self.mgr, ent)
            payload.add_facts(facts)
            payload.tool_calls += 1
        payload.deduplicate()
        return payload

    def _mode_count(
        self,
        entities: list[str],
        payload: RetrievalPayload,
    ) -> RetrievalPayload:
        """Count query — scan and count relations."""
        for ent in entities[:5]:
            facts = find_rel(self.mgr, ent, limit=50)
            payload.add_facts(facts)
            payload.tool_calls += 1
        payload.deduplicate()
        return payload

    def rebuild_index(self) -> None:
        """Rebuild compact index and router after writes."""
        self.index = build_compact_index(self.mgr)
        self.router = LexicalRouter(self.index)

    def stats(self) -> dict:
        """Pipeline statistics."""
        return {
            "index": {
                "n_records": len(self.index.records),
                "n_entities": len(self.index.all_entities()),
                "n_relations": len(self.index.all_relations()),
                "token_cost": self.index.token_cost(),
                "summary": self.index.compact_summary(),
            },
            "cache": self.cache.stats(),
            "instrumentation": self.instrumentor.summary(),
        }


class V2WritePipeline:
    """Unified write path: classify → extract → encode → update index.

    Replaces: always-LLM extraction → encode
    With: content-lane routing → cascade extraction → conditional LLM
    """

    def __init__(
        self,
        mgr: ShardManager,
        read_pipeline: V2ReadPipeline,
        skip_llm: bool = False,
    ):
        self.mgr = mgr
        self.read_pipeline = read_pipeline
        self.cascade = ExtractionCascade(skip_llm=skip_llm)
        self.instrumentor = read_pipeline.instrumentor  # shared

    def store(self, text: str) -> dict:
        """Full write pipeline: classify → extract → encode → update."""

        with self.instrumentor.timed("write") as event:
            # Step 1: Content lane classification
            lane_result = classify_content(text)
            config = lane_config(lane_result.lane)
            event.content_lane = lane_result.lane.value

            # Step 2: Skip if chit-chat
            if config["extraction"] == "skip":
                event.extraction_method = "skip"
                event.confidence = lane_result.confidence
                return {
                    "stored": False,
                    "reason": "chit_chat",
                    "lane": lane_result.lane.value,
                }

            # Step 3: Extraction cascade
            content_type = lane_result.lane.value
            extraction = self.cascade.extract(text, content_type=content_type)
            event.extraction_method = extraction.method
            event.confidence = extraction.confidence

            if not extraction.triples:
                return {
                    "stored": False,
                    "reason": "no_triples_extracted",
                    "method": extraction.method,
                    "confidence": extraction.confidence,
                }

            # Step 4: Encode into shard
            shard, encoded = self.mgr.encode("", triples=extraction.triples)
            event.n_facts = len(encoded)
            event.n_shards_touched = 1

            # Step 5: Update index and invalidate cache
            self.read_pipeline.rebuild_index()
            for triple in extraction.triples:
                self.read_pipeline.cache.invalidate_entity(triple.subject)
                self.read_pipeline.cache.invalidate_entity(triple.object)

            return {
                "stored": True,
                "n_triples": len(encoded),
                "method": extraction.method,
                "confidence": extraction.confidence,
                "lane": lane_result.lane.value,
                "shard": shard.name,
            }

    @property
    def extraction_stats(self) -> dict:
        return self.cascade.stats
