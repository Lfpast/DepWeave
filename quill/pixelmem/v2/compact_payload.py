"""Compact Retrieval Payloads — structured IDs instead of verbose text.

Returns interned dictionaries + integer quad arrays instead of
"alice works_at acme [since 2023]" strings. Resolves to readable
text only at final answer stage or debug mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from pixelmem.v2.relation_tools import FactEntry


@dataclass
class RetrievalPayload:
    """Compact structured retrieval result."""
    # String tables (interned dictionaries)
    entity_dict: list[str] = field(default_factory=list)
    relation_dict: list[str] = field(default_factory=list)
    condition_dict: list[str] = field(default_factory=list)
    # Quads as integer indices into the dicts: (s, r, o, c)
    quads: list[tuple[int, int, int, int]] = field(default_factory=list)
    # Metadata
    query_mode: str = "general"
    sources: list[int] = field(default_factory=list)  # shard indices
    tool_calls: int = 0
    # Internals for building
    _entity_idx: dict[str, int] = field(default_factory=dict, repr=False)
    _relation_idx: dict[str, int] = field(default_factory=dict, repr=False)
    _condition_idx: dict[str, int] = field(default_factory=dict, repr=False)

    def _intern_entity(self, name: str) -> int:
        if name not in self._entity_idx:
            self._entity_idx[name] = len(self.entity_dict)
            self.entity_dict.append(name)
        return self._entity_idx[name]

    def _intern_relation(self, name: str) -> int:
        if name not in self._relation_idx:
            self._relation_idx[name] = len(self.relation_dict)
            self.relation_dict.append(name)
        return self._relation_idx[name]

    def _intern_condition(self, cond: str) -> int:
        if not cond:
            cond = ""
        if cond not in self._condition_idx:
            self._condition_idx[cond] = len(self.condition_dict)
            self.condition_dict.append(cond)
        return self._condition_idx[cond]

    def add_fact(self, fact: FactEntry) -> None:
        """Add a fact, interning all strings."""
        si = self._intern_entity(fact.subject)
        ri = self._intern_relation(fact.relation)
        oi = self._intern_entity(fact.object)
        ci = self._intern_condition(fact.condition)
        self.quads.append((si, ri, oi, ci))
        if fact.shard_idx >= 0 and fact.shard_idx not in self.sources:
            self.sources.append(fact.shard_idx)

    def add_facts(self, facts: list[FactEntry]) -> None:
        for f in facts:
            self.add_fact(f)

    @property
    def n_facts(self) -> int:
        return len(self.quads)

    @property
    def n_entities(self) -> int:
        return len(self.entity_dict)

    # ── Output formats ──────────────────────────────────────────

    def to_compact_json(self) -> str:
        """Ultra-compact JSON — minimal tokens."""
        return json.dumps({
            "e": self.entity_dict,
            "r": self.relation_dict,
            "c": self.condition_dict,
            "q": self.quads,
        }, separators=(",", ":"))

    def to_fact_list(self) -> list[dict]:
        """Expand to list of {s, r, o, c} dicts."""
        return [
            {
                "s": self.entity_dict[s],
                "r": self.relation_dict[r],
                "o": self.entity_dict[o],
                "c": self.condition_dict[c] if c < len(self.condition_dict) else "",
            }
            for s, r, o, c in self.quads
        ]

    def to_text(self, fmt: str = "compact") -> str:
        """Render as text for LLM consumption.

        Formats:
          "compact": "user works_at acme [since 2023]"
          "verbose": "- user --[works_at]--> acme (context: since 2023)"
          "csv": "user,works_at,acme,since 2023"
        """
        lines = []
        for s, r, o, c in self.quads:
            subj = self.entity_dict[s]
            rel = self.relation_dict[r]
            obj = self.entity_dict[o]
            cond = self.condition_dict[c] if c < len(self.condition_dict) else ""

            if fmt == "compact":
                line = f"{subj} {rel} {obj}"
                if cond:
                    line += f" [{cond}]"
            elif fmt == "verbose":
                line = f"- {subj} --[{rel}]--> {obj}"
                if cond:
                    line += f" ({cond})"
            elif fmt == "csv":
                line = f"{subj},{rel},{obj},{cond}"
            else:
                line = f"{subj} {rel} {obj}"
            lines.append(line)
        return "\n".join(lines)

    def estimate_tokens(self) -> int:
        """Estimate token cost of the compact text representation."""
        return len(self.to_text("compact")) // 4

    # ── Merge and deduplicate ───────────────────────────────────

    def deduplicate(self) -> None:
        """Remove duplicate quads."""
        seen = set()
        unique = []
        for q in self.quads:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        self.quads = unique

    # ── Serialization ───────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "entity_dict": self.entity_dict,
            "relation_dict": self.relation_dict,
            "condition_dict": self.condition_dict,
            "quads": self.quads,
            "query_mode": self.query_mode,
            "sources": self.sources,
            "tool_calls": self.tool_calls,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievalPayload":
        p = cls(
            entity_dict=d["entity_dict"],
            relation_dict=d["relation_dict"],
            condition_dict=d.get("condition_dict", []),
            quads=[tuple(q) for q in d["quads"]],
            query_mode=d.get("query_mode", "general"),
            sources=d.get("sources", []),
            tool_calls=d.get("tool_calls", 0),
        )
        # Rebuild intern indices
        for i, e in enumerate(p.entity_dict):
            p._entity_idx[e] = i
        for i, r in enumerate(p.relation_dict):
            p._relation_idx[r] = i
        for i, c in enumerate(p.condition_dict):
            p._condition_idx[c] = i
        return p


def merge_payloads(payloads: list[RetrievalPayload]) -> RetrievalPayload:
    """Merge multiple payloads, deduplicating across them."""
    merged = RetrievalPayload()
    for p in payloads:
        for s, r, o, c in p.quads:
            fact = FactEntry(
                subject=p.entity_dict[s],
                relation=p.relation_dict[r],
                object=p.entity_dict[o],
                condition=p.condition_dict[c] if c < len(p.condition_dict) else "",
            )
            merged.add_fact(fact)
        merged.sources.extend(s for s in p.sources if s not in merged.sources)
    merged.deduplicate()
    return merged
