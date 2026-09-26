"""First-class summary dataclasses for PixelMem V3.

Provides structured, serialisable summaries for entities, sources, and tasks.
No dependencies outside the Python standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# EntitySummary
# ---------------------------------------------------------------------------

@dataclass
class EntitySummary:
    """Compact summary of a single entity across shards.

    Attributes:
        entity_name: Canonical name (lower-cased, deduplicated).
        aliases: Alternative surface forms seen for this entity.
        lane: Memory lane the entity belongs to (e.g. "biomedical").
        relation_counts: Mapping from relation type to occurrence count.
        top_relations: Top-5 (relation, target, condition) triples by salience.
        first_seen: ISO-8601 timestamp of earliest mention.
        last_seen: ISO-8601 timestamp of most recent mention.
        salience: Importance score in [0, 1].
        text_digest: Human-readable digest of ~30 tokens.
        bm25_tokens: Pre-tokenised bag of words for BM25 retrieval.
    """

    entity_name: str
    aliases: List[str] = field(default_factory=list)
    lane: str = ""
    relation_counts: Dict[str, int] = field(default_factory=dict)
    top_relations: List[Tuple[str, str, str]] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    salience: float = 0.0
    text_digest: str = ""
    bm25_tokens: List[str] = field(default_factory=list)

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation (JSON-safe)."""
        d = asdict(self)
        # tuples become lists via asdict; keep them as lists for JSON compat
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntitySummary":
        """Reconstruct from a plain dict."""
        d = dict(d)  # shallow copy
        # Restore tuples for top_relations
        if "top_relations" in d:
            d["top_relations"] = [tuple(t) for t in d["top_relations"]]
        return cls(**d)

    def to_compact_str(self) -> str:
        """One-line compact representation suitable for prompt injection."""
        rels = "; ".join(
            f"{r}->{t}" + (f" [{c}]" if c else "")
            for r, t, c in self.top_relations[:5]
        )
        return (
            f"[Entity] {self.entity_name} "
            f"(lane={self.lane}, salience={self.salience:.2f}, "
            f"rels={len(self.relation_counts)}): {rels}"
        )


# ---------------------------------------------------------------------------
# SourceSummary
# ---------------------------------------------------------------------------

@dataclass
class SourceSummary:
    """Summary of one ingested source chunk.

    Attributes:
        source_id: Identifier like ``shard_0:chunk_3``.
        lane: Memory lane.
        timestamp: ISO-8601 ingest timestamp.
        entities: Entity names mentioned in this source.
        relations: Relation types present.
        conditions: Conditions / qualifiers attached to facts.
        n_facts: Number of extracted facts.
        text_digest: Short prose digest.
        bm25_tokens: Pre-tokenised bag of words for BM25 retrieval.
    """

    source_id: str = ""
    lane: str = ""
    timestamp: str = ""
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    n_facts: int = 0
    text_digest: str = ""
    bm25_tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceSummary":
        return cls(**d)

    def to_compact_str(self) -> str:
        ents = ", ".join(self.entities[:6])
        return (
            f"[Source] {self.source_id} "
            f"(lane={self.lane}, facts={self.n_facts}): entities=[{ents}]"
        )


# ---------------------------------------------------------------------------
# TaskSummary
# ---------------------------------------------------------------------------

@dataclass
class TaskSummary:
    """Summary of a completed or in-progress retrieval task.

    Attributes:
        task_id: Unique task identifier.
        query_text: The original user query.
        mode: Retrieval mode used (e.g. "entity", "relational", "hybrid").
        entities_touched: Entity names accessed during the task.
        confidence: Aggregate confidence in [0, 1].
        plan_hash: Hash of the query-plan for deduplication / caching.
    """

    task_id: str = ""
    query_text: str = ""
    mode: str = ""
    entities_touched: List[str] = field(default_factory=list)
    confidence: float = 0.0
    plan_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskSummary":
        return cls(**d)

    def to_compact_str(self) -> str:
        ents = ", ".join(self.entities_touched[:6])
        return (
            f"[Task] {self.task_id} mode={self.mode} "
            f"conf={self.confidence:.2f} entities=[{ents}]"
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenise(text: str) -> List[str]:
    """Cheap whitespace + punctuation tokeniser (lowercase)."""
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


def summary_to_tokens(
    summary: "EntitySummary | SourceSummary | TaskSummary",
) -> List[str]:
    """Extract BM25-ready tokens from any summary object.

    If the summary already has ``bm25_tokens`` populated, those are returned
    directly.  Otherwise tokens are derived from the compact string form.
    """
    if hasattr(summary, "bm25_tokens") and summary.bm25_tokens:
        return list(summary.bm25_tokens)
    return _tokenise(summary.to_compact_str())


def summary_to_dict(
    summary: "EntitySummary | SourceSummary | TaskSummary",
) -> Dict[str, Any]:
    """Serialise any summary to a tagged dict (includes ``_type`` key)."""
    d = summary.to_dict()
    d["_type"] = type(summary).__name__
    return d


_SUMMARY_CLASSES = {
    "EntitySummary": EntitySummary,
    "SourceSummary": SourceSummary,
    "TaskSummary": TaskSummary,
}


def summary_from_dict(d: Dict[str, Any]) -> "EntitySummary | SourceSummary | TaskSummary":
    """Deserialise a tagged dict produced by :func:`summary_to_dict`.

    Raises ``ValueError`` if the ``_type`` key is missing or unrecognised.
    """
    d = dict(d)
    type_name = d.pop("_type", None)
    if type_name is None:
        raise ValueError("Dict is missing '_type' key; cannot reconstruct summary.")
    cls = _SUMMARY_CLASSES.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown summary type '{type_name}'.")
    return cls.from_dict(d)
