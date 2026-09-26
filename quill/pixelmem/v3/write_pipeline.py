"""PixelMem V3 Write Pipeline -- staged ingestion with lane routing.

Orchestrates the full write path:
  1. Classify incoming text into a content lane
  2. Route to the appropriate extractor (fast regex, AST, or LLM)
  3. Canonicalize extracted triples
  4. Encode into the best shard via ShardManager
  5. Incrementally update the summary store

Typical usage::

    from pixelmem.shard_manager import ShardManager
    from pixelmem.v3.summary_store import SummaryStore
    from pixelmem.v3.write_pipeline import WritePipeline

    mgr = ShardManager("./store")
    store = SummaryStore()
    pipe = WritePipeline(mgr, store)
    result = pipe.store("Alice works at Acme Corp since 2023.")
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from pixelmem.encoder import encode_text
from pixelmem.shard_manager import ShardManager
from pixelmem.triple_extractor import Triple, extract_triples

from pixelmem.v3.lane_classifier import (
    ContentLane,
    LaneResult,
    classify_lane,
    lane_config,
)
from pixelmem.v3.canonicalizer import Canonicalizer
from pixelmem.v3.summary_builder import SummaryBuilder
from pixelmem.v3.summary_store import SummaryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WriteResult:
    """Outcome of a single :meth:`WritePipeline.store` call.

    Attributes:
        stored: Whether anything was actually written to a shard.
        n_triples: Number of triples encoded.
        method: Extraction method used (``"fast"``, ``"cli"``, ``"ast"``,
            ``"regex_workflow"``, ``"skip"``).
        confidence: Aggregate extraction confidence in [0, 1].
        lane: Content lane the text was routed to.
        shard: Name of the shard that received the triples.
        canonicalized: Number of triples that were modified by canonicalization.
        summaries_updated: Number of entity summaries refreshed.
    """

    stored: bool = False
    n_triples: int = 0
    method: str = ""
    confidence: float = 0.0
    lane: str = ""
    shard: str = ""
    canonicalized: int = 0
    summaries_updated: int = 0


# ---------------------------------------------------------------------------
# Confidence scorer
# ---------------------------------------------------------------------------

_MEANINGFUL_RELATIONS = {
    "works_at", "lives_in", "born_in", "studied_at", "likes", "prefers",
    "has_skill", "knows", "partner_of", "collaborates_with", "manages",
    "contains_function", "contains_class", "imports", "calls", "extends",
    "depends_on", "configures", "tests", "treats", "inhibits",
}

_GENERIC_RELATIONS = {"is", "is_a", "has", "related_to"}


def _confidence_score(triples: list[Triple], text: str) -> float:
    """Heuristic confidence for a set of extracted triples.

    Higher when:
      - More triples per sentence
      - Relations are specific (not just "is" / "has")
      - Entity names appear verbatim in the source text
    """
    if not triples:
        return 0.0

    text_lower = text.lower()
    n_sentences = max(1, len(re.split(r"[.!?]+", text.strip())))

    # Coverage: triples per sentence
    coverage = min(1.0, len(triples) / max(1, n_sentences))

    # Specificity: fraction of relations that are meaningful
    meaningful = 0
    for t in triples:
        rel = t.relation.lower().replace(" ", "_")
        if rel in _MEANINGFUL_RELATIONS:
            meaningful += 1
        elif rel not in _GENERIC_RELATIONS:
            meaningful += 0.5
    specificity = min(1.0, meaningful / len(triples)) if triples else 0.0

    # Grounding: fraction of entities found in source text
    grounded = 0
    total_entities = 0
    for t in triples:
        for name in (t.subject, t.object):
            total_entities += 1
            if name.lower() in text_lower:
                grounded += 1
    grounding = grounded / max(1, total_entities)

    # Weighted combination
    score = 0.40 * coverage + 0.35 * specificity + 0.25 * grounding
    return round(min(1.0, score), 4)


# ---------------------------------------------------------------------------
# Workflow regex fallback
# ---------------------------------------------------------------------------

_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+)?import\s+([\w., ]+)", re.MULTILINE
)
_FUNC_RE = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)
_CLASS_RE = re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)
_ASSIGN_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*=", re.MULTILINE)


def _extract_workflow_regex(text: str) -> list[Triple]:
    """Fast regex-based extraction for code-like text.

    Catches imports, function defs, class defs, and constant assignments
    without needing a full AST parse.
    """
    triples: list[Triple] = []

    # Imports
    for m in _IMPORT_RE.finditer(text):
        module = m.group(1) or ""
        names = [n.strip() for n in m.group(2).split(",")]
        for name in names:
            if name and name != "*":
                source = module if module else name
                triples.append(Triple(
                    subject="__module__",
                    relation="imports",
                    object=source if module else name,
                    condition=name if module else "",
                ))

    # Function definitions
    for m in _FUNC_RE.finditer(text):
        triples.append(Triple(
            subject="__module__",
            relation="contains_function",
            object=m.group(1),
        ))

    # Class definitions
    for m in _CLASS_RE.finditer(text):
        triples.append(Triple(
            subject="__module__",
            relation="contains_class",
            object=m.group(1),
        ))

    # Constant assignments
    for m in _ASSIGN_RE.finditer(text):
        triples.append(Triple(
            subject="__module__",
            relation="defines_constant",
            object=m.group(1),
        ))

    return triples


# ---------------------------------------------------------------------------
# WritePipeline
# ---------------------------------------------------------------------------

class WritePipeline:
    """Staged write pipeline for PixelMem V3.

    Classifies incoming text, extracts triples via the appropriate
    strategy, canonicalizes them, encodes into a shard, and updates
    the summary store.

    Parameters
    ----------
    mgr : ShardManager
        The shard manager to write into.
    summary_store : SummaryStore
        Store to receive incremental summary updates.
    canonicalizer : Canonicalizer, optional
        Entity/relation normalizer. If ``None``, canonicalization is skipped.
    skip_llm : bool
        If ``True``, never escalate to CLI-based LLM extraction even when
        the fast extractor produces low-confidence results.
    """

    # Confidence threshold below which we escalate to LLM
    _CONFIDENCE_THRESHOLD = 0.45

    def __init__(
        self,
        mgr: ShardManager,
        summary_store: SummaryStore,
        canonicalizer: Optional[Canonicalizer] = None,
        skip_llm: bool = False,
    ) -> None:
        self._mgr = mgr
        self._summary_store = summary_store
        self._canonicalizer = canonicalizer
        self._skip_llm = skip_llm
        self._summary_builder = SummaryBuilder(mgr)

        # Counters
        self._total_writes = 0
        self._total_triples = 0
        self._total_skips = 0
        self._lane_counts: dict[str, int] = {}
        self._method_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def store(self, text: str, context: Optional[dict] = None) -> WriteResult:
        """Ingest *text* through the full write pipeline.

        Steps:
          1. Classify content lane.
          2. Skip if chit-chat.
          3. Extract triples according to lane strategy.
          4. Canonicalize triples.
          5. Encode into shard via ShardManager.
          6. Incrementally update summaries for touched entities.
          7. Return WriteResult.

        Parameters
        ----------
        text : str
            Raw input text.
        context : dict, optional
            Caller-provided hints (``file_path``, ``is_assistant``, etc.).

        Returns
        -------
        WriteResult
        """
        ctx = context or {}
        t0 = time.monotonic()

        # 1. Classify lane
        lane_result = classify_lane(text, ctx)
        lane_name = lane_result.lane.value

        # Track lane
        self._lane_counts[lane_name] = self._lane_counts.get(lane_name, 0) + 1

        # 2. Skip chit-chat
        if lane_result.lane == ContentLane.CHIT_CHAT:
            self._total_skips += 1
            logger.debug("Skipping chit-chat: %.40s...", text)
            return WriteResult(
                stored=False,
                method="skip",
                lane=lane_name,
                confidence=lane_result.confidence,
            )

        # 3. Extract triples
        triples, method, confidence = self._extract(text, lane_result, ctx)

        if not triples:
            self._total_skips += 1
            return WriteResult(
                stored=False,
                n_triples=0,
                method=method,
                confidence=confidence,
                lane=lane_name,
            )

        # 4. Canonicalize
        canonicalized_count = 0
        if self._canonicalizer is not None:
            original_tuples = [t.as_tuple() for t in triples]
            triples = self._canonicalizer.canonicalize_batch(triples)
            for orig, canon in zip(original_tuples, triples):
                if orig != canon.as_tuple():
                    canonicalized_count += 1

        # 5. Encode into shard
        shard, encoded = self._mgr.encode(text, triples=triples)
        shard_name = getattr(shard, "name", "unknown")

        # 6. Incrementally update summaries
        touched_entities: set[str] = set()
        for t in triples:
            touched_entities.add(t.subject.strip().lower())
            touched_entities.add(t.object.strip().lower())

        summaries_updated = 0
        if touched_entities:
            try:
                self._summary_builder.incremental_update(
                    self._summary_store, list(touched_entities)
                )
                summaries_updated = len(touched_entities)
            except Exception:
                logger.warning(
                    "Summary update failed for entities: %s",
                    touched_entities,
                    exc_info=True,
                )

        # Update counters
        self._total_writes += 1
        self._total_triples += len(encoded)
        self._method_counts[method] = self._method_counts.get(method, 0) + 1

        elapsed = time.monotonic() - t0
        logger.debug(
            "Stored %d triples via %s in %.1fms (lane=%s, shard=%s)",
            len(encoded), method, elapsed * 1000, lane_name, shard_name,
        )

        return WriteResult(
            stored=True,
            n_triples=len(encoded),
            method=method,
            confidence=confidence,
            lane=lane_name,
            shard=shard_name,
            canonicalized=canonicalized_count,
            summaries_updated=summaries_updated,
        )

    # ------------------------------------------------------------------
    # Extraction dispatcher
    # ------------------------------------------------------------------

    def _extract(
        self,
        text: str,
        lane_result: LaneResult,
        ctx: dict,
    ) -> tuple[list[Triple], str, float]:
        """Route to the appropriate extractor based on lane.

        Returns
        -------
        (triples, method_name, confidence)
        """
        lane = lane_result.lane

        # --- WORKFLOW lane ---
        if lane == ContentLane.WORKFLOW:
            file_path = ctx.get("file_path")
            if file_path:
                try:
                    from pixelmem.workflow.extractor import extract_file
                    triples = extract_file(file_path)
                    conf = _confidence_score(triples, text)
                    return triples, "ast", conf
                except Exception:
                    logger.debug("AST extraction failed, falling back to regex")
            # Regex fallback for workflow text without a file path
            triples = _extract_workflow_regex(text)
            conf = _confidence_score(triples, text)
            return triples, "regex_workflow", conf

        # --- MIXED lane: try both fast + workflow regex ---
        if lane == ContentLane.MIXED:
            fast_triples = extract_triples(text, fast=True)
            wf_triples = _extract_workflow_regex(text)
            # Merge, dedup by (s, r, o)
            seen: set[tuple[str, str, str]] = set()
            merged: list[Triple] = []
            for t in fast_triples + wf_triples:
                key = (t.subject.lower(), t.relation.lower(), t.object.lower())
                if key not in seen:
                    seen.add(key)
                    merged.append(t)
            conf = _confidence_score(merged, text)
            return merged, "fast+regex", conf

        # --- Default: cheap extraction cascade ---
        triples = extract_triples(text, fast=True)
        conf = _confidence_score(triples, text)

        # Escalate to CLI LLM if confidence is low
        if conf < self._CONFIDENCE_THRESHOLD and not self._skip_llm:
            logger.debug(
                "Low confidence (%.2f), escalating to CLI extraction", conf
            )
            try:
                llm_triples = extract_triples(text, use_cli=True)
                llm_conf = _confidence_score(llm_triples, text)
                if llm_conf > conf:
                    return llm_triples, "cli", llm_conf
            except Exception:
                logger.warning("CLI extraction failed, using fast results", exc_info=True)

        return triples, "fast", conf

    # ------------------------------------------------------------------
    # Batch store
    # ------------------------------------------------------------------

    def batch_store(self, texts: list[str], contexts: Optional[list[dict]] = None) -> list[WriteResult]:
        """Store multiple texts through the pipeline.

        Parameters
        ----------
        texts : list[str]
            Raw input texts.
        contexts : list[dict], optional
            Per-text context dicts. If shorter than *texts*, missing
            entries default to ``{}``.

        Returns
        -------
        list[WriteResult]
        """
        ctxs = contexts or []
        results: list[WriteResult] = []
        for i, text in enumerate(texts):
            ctx = ctxs[i] if i < len(ctxs) else {}
            results.append(self.store(text, context=ctx))
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Aggregate pipeline statistics.

        Returns
        -------
        dict
            Keys: ``total_writes``, ``total_triples``, ``total_skips``,
            ``lane_counts``, ``method_counts``.
        """
        return {
            "total_writes": self._total_writes,
            "total_triples": self._total_triples,
            "total_skips": self._total_skips,
            "lane_counts": dict(self._lane_counts),
            "method_counts": dict(self._method_counts),
        }
