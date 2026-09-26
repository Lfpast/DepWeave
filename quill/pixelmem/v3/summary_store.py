"""BM25-searchable store for PixelMem V3 summary objects.

Provides indexed storage of EntitySummary, SourceSummary, and TaskSummary
objects with BM25 retrieval. Falls back to simple token-overlap scoring
if rank_bm25 is not installed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pixelmem.v3.summary_objects import (
    EntitySummary,
    SourceSummary,
    TaskSummary,
    summary_to_tokens,
    summary_to_dict,
    summary_from_dict,
)

# ---------------------------------------------------------------------------
# BM25 backend — graceful fallback
# ---------------------------------------------------------------------------

try:
    from rank_bm25 import BM25Okapi

    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenise(text: str) -> List[str]:
    """Cheap whitespace + punctuation tokeniser (lowercase)."""
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


# ---------------------------------------------------------------------------
# Fallback scorer when rank_bm25 is unavailable
# ---------------------------------------------------------------------------

class _SimpleOverlapScorer:
    """Minimal token-overlap scorer that mimics the BM25Okapi interface."""

    def __init__(self, corpus: List[List[str]]) -> None:
        self.corpus = corpus

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        query_set = set(query_tokens)
        scores: List[float] = []
        for doc_tokens in self.corpus:
            if not doc_tokens:
                scores.append(0.0)
                continue
            doc_set = set(doc_tokens)
            overlap = len(query_set & doc_set)
            # Normalise by document length to penalise very long docs
            scores.append(overlap / (1 + len(doc_set) * 0.1))
        return scores


# ---------------------------------------------------------------------------
# SummaryStore
# ---------------------------------------------------------------------------

SummaryUnion = Union[EntitySummary, SourceSummary, TaskSummary]


class SummaryStore:
    """Indexed store for EntitySummary, SourceSummary, and TaskSummary objects.

    Supports BM25 search across all summaries, with optional type filtering.
    The BM25 index is built lazily on first search and can be rebuilt
    explicitly via :meth:`rebuild_bm25`.
    """

    def __init__(self) -> None:
        self._entities: Dict[str, EntitySummary] = {}
        self._sources: Dict[str, SourceSummary] = {}
        self._tasks: Dict[str, TaskSummary] = {}

        # Lazy BM25 index
        self._bm25: Any = None  # BM25Okapi | _SimpleOverlapScorer | None
        self._index_keys: List[Tuple[str, str]] = []  # (type_tag, key)
        self._index_tokens: List[List[str]] = []
        self._dirty: bool = True  # index needs rebuild

    # ------------------------------------------------------------------
    # Add methods
    # ------------------------------------------------------------------

    def add_entity_summary(self, summary: EntitySummary) -> None:
        """Add or replace an entity summary (keyed by entity_name)."""
        self._entities[summary.entity_name] = summary
        self._dirty = True

    def add_source_summary(self, summary: SourceSummary) -> None:
        """Add or replace a source summary (keyed by source_id)."""
        self._sources[summary.source_id] = summary
        self._dirty = True

    def add_task_summary(self, summary: TaskSummary) -> None:
        """Add or replace a task summary (keyed by task_id)."""
        self._tasks[summary.task_id] = summary
        self._dirty = True

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_entity(self, name: str) -> Optional[EntitySummary]:
        """Return the EntitySummary for *name*, or None."""
        return self._entities.get(name)

    def get_source(self, source_id: str) -> Optional[SourceSummary]:
        """Return the SourceSummary for *source_id*, or None."""
        return self._sources.get(source_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Build or rebuild the BM25 index if dirty."""
        if not self._dirty and self._bm25 is not None:
            return

        self._index_keys = []
        self._index_tokens = []

        for name, s in self._entities.items():
            tokens = summary_to_tokens(s)
            self._index_keys.append(("entity", name))
            self._index_tokens.append(tokens)

        for sid, s in self._sources.items():
            tokens = summary_to_tokens(s)
            self._index_keys.append(("source", sid))
            self._index_tokens.append(tokens)

        for tid, s in self._tasks.items():
            tokens = summary_to_tokens(s)
            self._index_keys.append(("task", tid))
            self._index_tokens.append(tokens)

        if not self._index_tokens:
            self._bm25 = None
            self._dirty = False
            return

        if _HAS_BM25:
            self._bm25 = BM25Okapi(self._index_tokens)
        else:
            self._bm25 = _SimpleOverlapScorer(self._index_tokens)

        self._dirty = False

    def search(
        self,
        query: str,
        top_k: int = 5,
        summary_type: Optional[str] = None,
    ) -> List[Tuple[float, SummaryUnion]]:
        """BM25 search across all summaries.

        Args:
            query: Free-text search query.
            top_k: Maximum results to return.
            summary_type: Optional filter — ``"entity"``, ``"source"``,
                or ``"task"``.

        Returns:
            List of (score, summary) tuples sorted by descending score.
        """
        self._ensure_index()

        if self._bm25 is None:
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)

        # BM25Okapi can produce negative scores with very few documents.
        # Use a threshold relative to the max score, not absolute > 0.
        max_score = max(raw_scores) if len(raw_scores) > 0 else 0.0
        threshold = max_score * 0.1 if max_score > 0 else -float("inf")

        scored: List[Tuple[float, str, str]] = []
        for idx, score in enumerate(raw_scores):
            type_tag, key = self._index_keys[idx]
            if summary_type and type_tag != summary_type:
                continue
            if score >= threshold:
                scored.append((float(score), type_tag, key))

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:top_k]

        results: List[Tuple[float, SummaryUnion]] = []
        for score, type_tag, key in scored:
            summary = self._resolve(type_tag, key)
            if summary is not None:
                results.append((score, summary))

        return results

    def search_entities(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[float, EntitySummary]]:
        """Shortcut: BM25 search restricted to entity summaries."""
        return self.search(query, top_k=top_k, summary_type="entity")  # type: ignore[return-value]

    def _resolve(self, type_tag: str, key: str) -> Optional[SummaryUnion]:
        if type_tag == "entity":
            return self._entities.get(key)
        elif type_tag == "source":
            return self._sources.get(key)
        elif type_tag == "task":
            return self._tasks.get(key)
        return None

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update_entity(self, name: str, **kwargs: Any) -> None:
        """Partial update of an existing EntitySummary.

        Only fields present in *kwargs* are overwritten; others are kept.
        Raises ``KeyError`` if the entity does not exist.
        """
        if name not in self._entities:
            raise KeyError(f"Entity '{name}' not found in store")
        summary = self._entities[name]
        for field_name, value in kwargs.items():
            if hasattr(summary, field_name):
                setattr(summary, field_name, value)
        self._dirty = True

    def invalidate_entity(self, name: str) -> None:
        """Remove an entity summary from the store."""
        self._entities.pop(name, None)
        self._dirty = True

    def rebuild_bm25(self) -> None:
        """Force a BM25 index rebuild (e.g. after batch changes)."""
        self._dirty = True
        self._ensure_index()

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def all_entities(self) -> List[str]:
        """Return all entity names in the store."""
        return list(self._entities.keys())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the store contents."""
        return {
            "entity_count": len(self._entities),
            "source_count": len(self._sources),
            "task_count": len(self._tasks),
            "total": len(self._entities) + len(self._sources) + len(self._tasks),
            "bm25_backend": "rank_bm25" if _HAS_BM25 else "simple_overlap",
            "index_dirty": self._dirty,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the store to a JSON file at *path*."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {
            "entities": {
                name: summary_to_dict(s) for name, s in self._entities.items()
            },
            "sources": {
                sid: summary_to_dict(s) for sid, s in self._sources.items()
            },
            "tasks": {
                tid: summary_to_dict(s) for tid, s in self._tasks.items()
            },
        }

        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SummaryStore":
        """Deserialise a store from a JSON file produced by :meth:`save`."""
        p = Path(path)
        with open(p) as f:
            data = json.load(f)

        store = cls()

        for _name, d in data.get("entities", {}).items():
            store.add_entity_summary(summary_from_dict(d))

        for _sid, d in data.get("sources", {}).items():
            store.add_source_summary(summary_from_dict(d))

        for _tid, d in data.get("tasks", {}).items():
            store.add_task_summary(summary_from_dict(d))

        return store

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"SummaryStore(entities={s['entity_count']}, "
            f"sources={s['source_count']}, tasks={s['task_count']}, "
            f"backend={s['bm25_backend']})"
        )
