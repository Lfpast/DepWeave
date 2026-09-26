"""Compact structured evidence output for PixelMem V3.

Provides :class:`Fact`, :class:`Evidence`, and :class:`EvidenceBundle` for
collecting, deduplicating, budgeting, and serialising retrieval results.
No dependencies outside the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Fact
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """A single knowledge-graph quad with provenance metadata.

    Attributes:
        subject: Entity on the left-hand side of the relation.
        relation: Relation type (e.g. "treats", "inhibits").
        object: Entity on the right-hand side.
        condition: Optional qualifier (dosage, context, etc.).
        shard_idx: Index of the shard this fact was extracted from.
        chunk_idx: Chunk index within the shard.
        confidence: Extraction confidence in [0, 1].
    """

    subject: str = ""
    relation: str = ""
    object: str = ""
    condition: str = ""
    shard_idx: int = -1
    chunk_idx: int = -1
    confidence: float = 1.0

    def key(self) -> Tuple[str, str, str]:
        """Return the (s, r, o) identity key (compat with retrieval_algebra.Fact)."""
        return (self.subject, self.relation, self.object)

    def quad_key(self) -> Tuple[str, str, str, str]:
        """Return a hashable (s, r, o, c) key for deduplication."""
        return (
            self.subject.lower(),
            self.relation.lower(),
            self.object.lower(),
            self.condition.lower(),
        )

    def to_compact_str(self) -> str:
        """One-line compact representation."""
        c = f" [{self.condition}]" if self.condition else ""
        return f"({self.subject} -{self.relation}-> {self.object}{c})"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fact":
        return cls(**d)

    def to_compact_str(self) -> str:
        cond = f" [{self.condition}]" if self.condition else ""
        return f"({self.subject} -{self.relation}-> {self.object}{cond})"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """A single piece of evidence wrapping a :class:`Fact`.

    Attributes:
        fact: The underlying structured fact.
        relevance: Relevance to the current query in [0, 1].
        provenance: Human-readable provenance string (e.g. "shard_2:chunk_7").
    """

    fact: Fact = field(default_factory=Fact)
    relevance: float = 0.0
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact.to_dict(),
            "relevance": self.relevance,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Evidence":
        return cls(
            fact=Fact.from_dict(d["fact"]),
            relevance=d.get("relevance", 0.0),
            provenance=d.get("provenance", ""),
        )

    def to_compact_str(self) -> str:
        return f"{self.fact.to_compact_str()} rel={self.relevance:.2f} @{self.provenance}"


# ---------------------------------------------------------------------------
# EvidenceBundle
# ---------------------------------------------------------------------------

@dataclass
class EvidenceBundle:
    """Collection of evidence items with interned string tables and metadata.

    The bundle supports deduplication, top-k selection, token-budget trimming,
    and multiple serialisation formats.

    Attributes:
        query: The original query text.
        mode: Retrieval mode (e.g. "entity", "relational", "hybrid").
        evidences: Ordered list of :class:`Evidence` objects.
        entity_dict: Interned entity string table  (id -> name).
        relation_dict: Interned relation string table (id -> name).
        condition_dict: Interned condition string table (id -> name).
        metadata: Arbitrary metadata (plan_id, operators_used, etc.).
    """

    query: str = ""
    mode: str = ""
    evidences: List[Evidence] = field(default_factory=list)

    # interned string tables
    entity_dict: Dict[int, str] = field(default_factory=dict)
    relation_dict: Dict[int, str] = field(default_factory=dict)
    condition_dict: Dict[int, str] = field(default_factory=dict)

    # metadata
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "plan_id": "",
        "operators_used": [],
        "tool_calls": 0,
        "latency_ms": 0.0,
    })

    # -- intern helpers (private) --------------------------------------------

    def _intern(self, table: Dict[int, str], value: str) -> int:
        """Add *value* to *table* if absent; return its integer id."""
        for k, v in table.items():
            if v == value:
                return k
        new_id = len(table)
        table[new_id] = value
        return new_id

    def _intern_fact(self, fact: Fact) -> None:
        """Ensure all strings in *fact* appear in the interned tables."""
        self._intern(self.entity_dict, fact.subject)
        self._intern(self.entity_dict, fact.object)
        self._intern(self.relation_dict, fact.relation)
        if fact.condition:
            self._intern(self.condition_dict, fact.condition)

    # -- mutation methods ----------------------------------------------------

    def add_evidence(self, evidence: Evidence) -> None:
        """Append an evidence item and update interned string tables."""
        self._intern_fact(evidence.fact)
        self.evidences.append(evidence)

    def deduplicate(self) -> None:
        """Remove duplicate facts, keeping the highest-relevance instance."""
        seen: Dict[Tuple[str, str, str, str], int] = {}
        deduped: List[Evidence] = []
        for ev in self.evidences:
            key = ev.fact.quad_key()
            if key in seen:
                idx = seen[key]
                if ev.relevance > deduped[idx].relevance:
                    deduped[idx] = ev
            else:
                seen[key] = len(deduped)
                deduped.append(ev)
        self.evidences = deduped

    def top_k(self, k: int) -> List[Evidence]:
        """Return the *k* most relevant evidence items (does not mutate)."""
        return sorted(self.evidences, key=lambda e: e.relevance, reverse=True)[:k]

    # -- quads view ----------------------------------------------------------

    @property
    def quads(self) -> List[Tuple[str, str, str, str, float]]:
        """Return evidence as ``(subject, relation, object, condition, relevance)`` tuples."""
        return [
            (
                ev.fact.subject,
                ev.fact.relation,
                ev.fact.object,
                ev.fact.condition,
                ev.relevance,
            )
            for ev in self.evidences
        ]

    # -- serialisation -------------------------------------------------------

    def to_compact_json(self) -> str:
        """Serialise to a compact JSON string (no unnecessary whitespace)."""
        return json.dumps(self.as_dict(), separators=(",", ":"), ensure_ascii=False)

    def to_text(self, fmt: str = "plain") -> str:
        """Render evidence as human-readable text.

        Args:
            fmt: ``"plain"`` for simple text, ``"markdown"`` for Markdown.
        """
        lines: List[str] = []
        header = f"Query: {self.query}  |  Mode: {self.mode}  |  Results: {len(self.evidences)}"
        if fmt == "markdown":
            lines.append(f"## {header}")
        else:
            lines.append(header)
            lines.append("-" * len(header))

        for i, ev in enumerate(self.evidences, 1):
            f = ev.fact
            cond = f" [{f.condition}]" if f.condition else ""
            line = (
                f"{i}. {f.subject} -{f.relation}-> {f.object}{cond}  "
                f"(rel={ev.relevance:.2f}, conf={f.confidence:.2f}, "
                f"@{ev.provenance})"
            )
            lines.append(line)
        return "\n".join(lines)

    def to_fact_list(self) -> List[Dict[str, Any]]:
        """Return facts as a flat list of dicts (no evidence wrapper)."""
        return [ev.fact.to_dict() for ev in self.evidences]

    def estimate_tokens(self) -> int:
        """Rough token estimate (4 chars per token heuristic)."""
        text = self.to_text()
        return max(1, len(text) // 4)

    def answer_context(self, budget_tokens: int) -> str:
        """Build a context string that fits within *budget_tokens*.

        Evidence is added in relevance order until the budget is exhausted.
        """
        sorted_evs = sorted(self.evidences, key=lambda e: e.relevance, reverse=True)
        parts: List[str] = []
        used = 0
        for ev in sorted_evs:
            line = ev.to_compact_str()
            est = max(1, len(line) // 4)
            if used + est > budget_tokens:
                break
            parts.append(line)
            used += est
        return "\n".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        """Full dict serialisation including interned tables and metadata."""
        return {
            "query": self.query,
            "mode": self.mode,
            "evidences": [ev.to_dict() for ev in self.evidences],
            "entity_dict": {str(k): v for k, v in self.entity_dict.items()},
            "relation_dict": {str(k): v for k, v in self.relation_dict.items()},
            "condition_dict": {str(k): v for k, v in self.condition_dict.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceBundle":
        """Reconstruct an EvidenceBundle from a dict (e.g. parsed JSON)."""
        bundle = cls(
            query=d.get("query", ""),
            mode=d.get("mode", ""),
            entity_dict={int(k): v for k, v in d.get("entity_dict", {}).items()},
            relation_dict={int(k): v for k, v in d.get("relation_dict", {}).items()},
            condition_dict={int(k): v for k, v in d.get("condition_dict", {}).items()},
            metadata=d.get("metadata", {}),
        )
        for ev_d in d.get("evidences", []):
            bundle.evidences.append(Evidence.from_dict(ev_d))
        return bundle


# ---------------------------------------------------------------------------
# merge_bundles
# ---------------------------------------------------------------------------

def merge_bundles(
    bundles: List[EvidenceBundle],
    *,
    deduplicate: bool = True,
    query: Optional[str] = None,
    mode: Optional[str] = None,
) -> EvidenceBundle:
    """Merge multiple :class:`EvidenceBundle` instances into one.

    Args:
        bundles: Bundles to merge (order is preserved within each bundle).
        deduplicate: If ``True``, remove duplicate facts after merging.
        query: Override query text; defaults to the first bundle's query.
        mode: Override mode; defaults to ``"merged"``.

    Returns:
        A new :class:`EvidenceBundle` containing all evidence.
    """
    if not bundles:
        return EvidenceBundle(query=query or "", mode=mode or "merged")

    merged = EvidenceBundle(
        query=query or bundles[0].query,
        mode=mode or "merged",
    )

    total_latency = 0.0
    total_tool_calls = 0
    all_operators: List[str] = []

    for b in bundles:
        for ev in b.evidences:
            merged.add_evidence(ev)
        # aggregate metadata
        meta = b.metadata
        total_latency += meta.get("latency_ms", 0.0)
        total_tool_calls += meta.get("tool_calls", 0)
        ops = meta.get("operators_used", [])
        if isinstance(ops, list):
            all_operators.extend(ops)

    if deduplicate:
        merged.deduplicate()

    merged.metadata = {
        "plan_id": f"merged_{hashlib.md5(merged.query.encode()).hexdigest()[:8]}",
        "operators_used": sorted(set(all_operators)),
        "tool_calls": total_tool_calls,
        "latency_ms": round(total_latency, 2),
    }
    return merged
