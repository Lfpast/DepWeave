"""Compact Nested Index — machine-oriented routing records.

Replaces verbose prose summaries with structured routing records:
  theme_id=work label=employment cnt_e=45 cnt_f=120 hot_rel=works_at|ceo_of

Each record is a fixed-schema dict with inverted indices for O(1) lookup.
No natural language. Expand to readable form only when explicitly requested.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pixelmem.memory import PixelMemUnit, BLACK
from pixelmem.shard_manager import ShardManager
from pixelmem.decoder import reconstruct_chunks


@dataclass(frozen=True)
class IndexRecord:
    """One routing record — compact metadata for a group of facts."""
    id: str
    parent_id: str  # theme id or "" for top-level
    label: str  # short keyword label
    entity_names: frozenset[str] = field(default_factory=frozenset)
    relation_types: frozenset[str] = field(default_factory=frozenset)
    shard_refs: tuple[int, ...] = field(default_factory=tuple)
    chunk_refs: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    n_entities: int = 0
    n_facts: int = 0
    # Lexical anchors — most frequent terms for BM25
    anchors: tuple[str, ...] = field(default_factory=tuple)

    def compact_str(self) -> str:
        """One-line machine-readable representation."""
        hot = "|".join(sorted(self.relation_types)[:3])
        return (
            f"id={self.id} label={self.label} "
            f"cnt_e={self.n_entities} cnt_f={self.n_facts} "
            f"hot_rel={hot}"
        )

    def expand(self) -> str:
        """Expanded human-readable form (for debugging only)."""
        ents = ", ".join(sorted(self.entity_names)[:8])
        if len(self.entity_names) > 8:
            ents += f" +{len(self.entity_names) - 8}"
        rels = ", ".join(sorted(self.relation_types))
        return (
            f"[{self.id}] {self.label}\n"
            f"  entities: {ents}\n"
            f"  relations: {rels}\n"
            f"  facts: {self.n_facts}, shards: {list(self.shard_refs)}"
        )


# Relation → theme mapping (same as nested_index.py RELATION_CATEGORIES)
_THEME_MAP = {
    # Conversation memory relations
    "works_at": "work", "employed_by": "work", "ceo_of": "work",
    "manages": "work", "reports_to": "work", "founded_by": "work",
    "member_of": "work", "role": "work",
    "lives_in": "places", "born_in": "places", "located_in": "places",
    "traveled_to": "places", "moved_to": "places",
    "has_skill": "skills", "studied_at": "skills", "degree": "skills",
    "certified_in": "skills",
    "knows": "social", "married_to": "social", "collaborates_with": "social",
    "partner_of": "social", "likes": "social",
    "works_on": "projects", "created": "projects", "owns": "projects",
    "uses": "projects",
    "prefers": "preferences", "favorite": "preferences",
    "always_uses": "preferences",
    # Workflow memory relations — code structure
    "imports": "code_deps",
    "calls": "code_deps",
    "depends_on": "code_deps",
    "extends": "code_deps",
    "contains_function": "code_structure",
    "contains_method": "code_structure",
    "contains_class": "code_structure",
    "contains_file": "code_structure",
    "contains_dir": "code_structure",
    "parameters": "code_structure",
    "decorated_by": "code_structure",
    "exports": "code_structure",
    "defines_constant": "code_config",
    "configures": "code_config",
    "configures_section": "code_config",
    "tests": "code_tests",
    "docstring": "code_docs",
    "type": "code_meta",
    "h1": "code_docs", "h2": "code_docs", "h3": "code_docs", "h4": "code_docs",
    "total_files": "code_meta",
}

_THEME_LABELS = {
    "work": "employment",
    "places": "locations",
    "skills": "education",
    "social": "relationships",
    "projects": "projects",
    "preferences": "preferences",
    "code_deps": "dependencies",
    "code_structure": "file_structure",
    "code_config": "configuration",
    "code_tests": "testing",
    "code_docs": "documentation",
    "code_meta": "file_metadata",
    "other": "general",
}


class CompactIndex:
    """Flat inverted index over structured routing records.

    O(1) lookup by entity name or relation type.
    No tree navigation — just direct index hits.
    """

    def __init__(self):
        self.records: list[IndexRecord] = []
        # Inverted indices
        self._entity_to_records: dict[str, list[int]] = defaultdict(list)
        self._relation_to_records: dict[str, list[int]] = defaultdict(list)
        self._theme_to_records: dict[str, list[int]] = defaultdict(list)
        self._id_to_idx: dict[str, int] = {}

    def add_record(self, record: IndexRecord) -> None:
        idx = len(self.records)
        self.records.append(record)
        self._id_to_idx[record.id] = idx
        for ent in record.entity_names:
            self._entity_to_records[ent].append(idx)
        for rel in record.relation_types:
            self._relation_to_records[rel].append(idx)
        self._theme_to_records[record.parent_id or record.id].append(idx)

    def lookup_entities(self, names: list[str]) -> list[IndexRecord]:
        """O(1) lookup: find records containing any of the given entities."""
        hits = set()
        for name in names:
            canonical = name.strip().lower()
            for idx in self._entity_to_records.get(canonical, []):
                hits.add(idx)
        return [self.records[i] for i in sorted(hits)]

    def lookup_relations(self, rels: list[str]) -> list[IndexRecord]:
        """O(1) lookup: find records containing any of the given relations."""
        hits = set()
        for rel in rels:
            canonical = rel.strip().lower()
            for idx in self._relation_to_records.get(canonical, []):
                hits.add(idx)
        return [self.records[i] for i in sorted(hits)]

    def lookup_theme(self, theme: str) -> list[IndexRecord]:
        """Get all records for a theme."""
        return [self.records[i] for i in self._theme_to_records.get(theme, [])]

    def get_record(self, record_id: str) -> Optional[IndexRecord]:
        idx = self._id_to_idx.get(record_id)
        return self.records[idx] if idx is not None else None

    def all_entities(self) -> set[str]:
        return set(self._entity_to_records.keys())

    def all_relations(self) -> set[str]:
        return set(self._relation_to_records.keys())

    def themes(self) -> list[str]:
        return sorted(self._theme_to_records.keys())

    def token_cost(self) -> int:
        """Estimate tokens if entire index is serialized as compact strings."""
        text = "\n".join(r.compact_str() for r in self.records)
        return len(text) // 4

    def compact_summary(self) -> str:
        """Ultra-compact index summary for LLM consumption."""
        themes = defaultdict(lambda: {"cnt_e": 0, "cnt_f": 0, "rels": set()})
        for r in self.records:
            t = r.parent_id or r.id
            themes[t]["cnt_e"] += r.n_entities
            themes[t]["cnt_f"] += r.n_facts
            themes[t]["rels"].update(r.relation_types)
        lines = []
        for theme in sorted(themes):
            d = themes[theme]
            label = _THEME_LABELS.get(theme, theme)
            hot = "|".join(sorted(d["rels"])[:4])
            lines.append(f"{label}({d['cnt_e']}e,{d['cnt_f']}f) {hot}")
        return " | ".join(lines)

    # ── Persistence ─────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        data = {
            "records": [
                {
                    "id": r.id,
                    "parent_id": r.parent_id,
                    "label": r.label,
                    "entity_names": sorted(r.entity_names),
                    "relation_types": sorted(r.relation_types),
                    "shard_refs": list(r.shard_refs),
                    "chunk_refs": list(r.chunk_refs),
                    "n_entities": r.n_entities,
                    "n_facts": r.n_facts,
                    "anchors": list(r.anchors),
                }
                for r in self.records
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> "CompactIndex":
        with open(path) as f:
            data = json.load(f)
        idx = cls()
        for rd in data["records"]:
            idx.add_record(IndexRecord(
                id=rd["id"],
                parent_id=rd["parent_id"],
                label=rd["label"],
                entity_names=frozenset(rd["entity_names"]),
                relation_types=frozenset(rd["relation_types"]),
                shard_refs=tuple(rd["shard_refs"]),
                chunk_refs=tuple(tuple(x) for x in rd.get("chunk_refs", [])),
                n_entities=rd["n_entities"],
                n_facts=rd["n_facts"],
                anchors=tuple(rd.get("anchors", [])),
            ))
        return idx


def build_compact_index(mgr: ShardManager) -> CompactIndex:
    """Build a CompactIndex from an existing ShardManager.

    Groups facts by theme (from relation types), then by primary entity
    within each theme. Produces flat records with inverted indices.
    """
    index = CompactIndex()

    # Collect all facts grouped by theme → primary entity
    theme_entity_facts: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {
            "entities": set(), "relations": set(),
            "shard_refs": set(), "chunk_refs": [], "n_facts": 0,
        })
    )

    for si, shard in enumerate(mgr.shards):
        for i in range(shard.n):
            subj = shard.idx_to_entity.get(i)
            if not subj:
                continue
            for j in range(shard.n):
                rgb = tuple(int(x) for x in shard.relation[i, j])
                if rgb == BLACK:
                    continue
                obj = shard.idx_to_entity.get(j, f"e{j}")
                rel = shard.color_to_relation.get(rgb, "unknown")
                theme = _THEME_MAP.get(rel, "other")

                group = theme_entity_facts[theme][subj]
                group["entities"].add(subj)
                group["entities"].add(obj)
                group["relations"].add(rel)
                group["shard_refs"].add(si)
                group["n_facts"] += 1

    # Build records: merge small entity groups within each theme
    for theme, entity_groups in theme_entity_facts.items():
        # Sort by fact count descending
        sorted_groups = sorted(
            entity_groups.items(), key=lambda x: -x[1]["n_facts"]
        )

        merged: list[dict] = []
        current: Optional[dict] = None
        max_per_record = 50  # max entities per record

        for primary_ent, group_data in sorted_groups:
            if current is None or len(current["entities"]) >= max_per_record:
                if current:
                    merged.append(current)
                current = {
                    "primary": primary_ent,
                    "entities": set(), "relations": set(),
                    "shard_refs": set(), "n_facts": 0,
                }
            current["entities"].update(group_data["entities"])
            current["relations"].update(group_data["relations"])
            current["shard_refs"].update(group_data["shard_refs"])
            current["n_facts"] += group_data["n_facts"]

        if current and current["entities"]:
            merged.append(current)

        # Create records
        for mi, m in enumerate(merged):
            # Build lexical anchors from most common entity name tokens
            all_tokens = []
            for ent in m["entities"]:
                all_tokens.extend(ent.replace("_", " ").split())
            token_freq = defaultdict(int)
            for t in all_tokens:
                if len(t) > 2:
                    token_freq[t] += 1
            anchors = sorted(token_freq, key=token_freq.get, reverse=True)[:10]

            record = IndexRecord(
                id=f"{theme}_{mi}",
                parent_id=theme,
                label=_THEME_LABELS.get(theme, theme),
                entity_names=frozenset(m["entities"]),
                relation_types=frozenset(m["relations"]),
                shard_refs=tuple(sorted(m["shard_refs"])),
                n_entities=len(m["entities"]),
                n_facts=m["n_facts"],
                anchors=tuple(anchors),
            )
            index.add_record(record)

    return index
