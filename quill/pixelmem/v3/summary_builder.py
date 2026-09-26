"""Build V3 summaries from ShardManager KG data.

Iterates shards once, collects all facts per entity, then builds
EntitySummary and SourceSummary objects in batch. Supports full
rebuild and incremental updates for individual entities.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from pixelmem.memory import PixelMemUnit, BLACK
from pixelmem.shard_manager import ShardManager
from pixelmem.v3.summary_objects import (
    EntitySummary,
    SourceSummary,
    TaskSummary,
    summary_to_tokens,
)
from pixelmem.v3.summary_store import SummaryStore

_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenise(text: str) -> List[str]:
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


# ---------------------------------------------------------------------------
# Fact record used during the collection pass
# ---------------------------------------------------------------------------

class _FactRecord:
    """Lightweight container for a decoded fact (avoids decoder import)."""
    __slots__ = ("subject", "relation", "object", "condition", "shard_idx")

    def __init__(
        self,
        subject: str,
        relation: str,
        object_: str,
        condition: str,
        shard_idx: int,
    ) -> None:
        self.subject = subject
        self.relation = relation
        self.object = object_
        self.condition = condition
        self.shard_idx = shard_idx


# ---------------------------------------------------------------------------
# SummaryBuilder
# ---------------------------------------------------------------------------

class SummaryBuilder:
    """Build V3 summaries from a ShardManager's knowledge graph.

    Usage::

        builder = SummaryBuilder(mgr)
        store = builder.build_all()
        store.save("summaries.json")
    """

    def __init__(self, mgr: ShardManager) -> None:
        self.mgr = mgr

    # ------------------------------------------------------------------
    # Full rebuild
    # ------------------------------------------------------------------

    def build_all(self) -> SummaryStore:
        """Iterate all shards, build entity + source summaries, return store.

        Single-pass collection: reads each shard's relation matrix once,
        collects facts keyed by entity, then builds summaries in batch.
        """
        store = SummaryStore()

        # Phase 1: collect all facts per entity (single pass over shards)
        entity_facts: Dict[str, List[_FactRecord]] = defaultdict(list)
        entity_lanes: Dict[str, str] = {}

        for shard_idx, shard in enumerate(self.mgr.shards):
            facts = self._extract_facts(shard, shard_idx)

            # Accumulate per entity
            for fact in facts:
                entity_facts[fact.subject].append(fact)
                entity_facts[fact.object].append(fact)

            # Track lane from shard topic
            lane = getattr(shard, "_topic", "") or ""
            if lane:
                for fact in facts:
                    entity_lanes.setdefault(fact.subject, lane)
                    entity_lanes.setdefault(fact.object, lane)

            # Build source summaries for this shard
            source_summaries = self.build_source_summaries(shard_idx)
            for ss in source_summaries:
                store.add_source_summary(ss)

        # Phase 2: build entity summaries in batch
        for entity_name, facts in entity_facts.items():
            summary = self._build_entity_from_facts(
                entity_name, facts, entity_lanes.get(entity_name, "")
            )
            store.add_entity_summary(summary)

        # Phase 3: rebuild BM25 index
        store.rebuild_bm25()

        return store

    # ------------------------------------------------------------------
    # Per-entity summary
    # ------------------------------------------------------------------

    def build_entity_summary(self, entity: str) -> EntitySummary:
        """Scan all shards for *entity* and build its EntitySummary."""
        facts: List[_FactRecord] = []
        lane = ""

        for shard_idx, shard in enumerate(self.mgr.shards):
            canonical = entity.strip().lower()
            if canonical not in shard.entity_to_idx:
                continue

            shard_facts = self._extract_facts(shard, shard_idx)
            for f in shard_facts:
                if f.subject == canonical or f.object == canonical:
                    facts.append(f)

            topic = getattr(shard, "_topic", "")
            if topic and not lane:
                lane = topic

        return self._build_entity_from_facts(entity.strip().lower(), facts, lane)

    def _build_entity_from_facts(
        self,
        entity: str,
        facts: List[_FactRecord],
        lane: str,
    ) -> EntitySummary:
        """Build an EntitySummary from pre-collected facts."""
        # Relation counts: count each unique relation type
        relation_counts: Dict[str, int] = defaultdict(int)
        relation_targets: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for f in facts:
            # Facts where entity is subject
            if f.subject == entity:
                relation_counts[f.relation] += 1
                relation_targets[f.relation].append((f.object, f.condition))
            # Facts where entity is object (inverse direction)
            elif f.object == entity:
                inv_rel = f"inv_{f.relation}"
                relation_counts[inv_rel] += 1
                relation_targets[inv_rel].append((f.subject, f.condition))

        # Top relations: pick the most frequent (relation, target, condition) triples
        scored_triples: List[Tuple[int, str, str, str]] = []
        for rel, targets in relation_targets.items():
            for target, cond in targets:
                scored_triples.append((relation_counts[rel], rel, target, cond))
        scored_triples.sort(key=lambda x: x[0], reverse=True)
        top_relations = [
            (rel, target, cond) for _, rel, target, cond in scored_triples[:5]
        ]

        # Salience
        salience = self._compute_salience(entity, facts)

        # Text digest
        text_digest = self._generate_text_digest(entity, facts)

        # BM25 tokens: entity name + relation names + top targets
        token_parts = [entity]
        token_parts.extend(relation_counts.keys())
        for rel, target, cond in top_relations:
            token_parts.append(target)
            if cond:
                token_parts.append(cond)
        bm25_tokens = _tokenise(" ".join(token_parts))

        return EntitySummary(
            entity_name=entity,
            aliases=[],
            lane=lane,
            relation_counts=dict(relation_counts),
            top_relations=top_relations,
            first_seen="",
            last_seen="",
            salience=salience,
            text_digest=text_digest,
            bm25_tokens=bm25_tokens,
        )

    # ------------------------------------------------------------------
    # Source summaries
    # ------------------------------------------------------------------

    def build_source_summaries(self, shard_idx: int) -> List[SourceSummary]:
        """Build one SourceSummary per chunk in the specified shard.

        Uses reconstruct_chunks to get chunk groupings from the tapes.
        """
        from pixelmem.decoder import reconstruct_chunks

        shard = self.mgr.shards[shard_idx]
        chunks = reconstruct_chunks(shard)
        lane = getattr(shard, "_topic", "") or ""
        summaries: List[SourceSummary] = []

        for chunk_idx, chunk in enumerate(chunks):
            if not chunk:
                continue

            source_id = f"shard_{shard_idx}:chunk_{chunk_idx}"
            entities: Set[str] = set()
            relations: Set[str] = set()
            conditions: Set[str] = set()

            for triple in chunk:
                entities.add(triple.subject)
                entities.add(triple.object)
                relations.add(triple.relation)
                if triple.condition:
                    conditions.add(triple.condition)

            # Build BM25 tokens from chunk content
            token_parts = list(entities) + list(relations) + list(conditions)
            bm25_tokens = _tokenise(" ".join(token_parts))

            # Text digest: compact summary of the chunk
            ent_list = sorted(entities)[:6]
            rel_list = sorted(relations)[:4]
            digest = (
                f"chunk with {len(chunk)} facts about {', '.join(ent_list)}"
                f" via {', '.join(rel_list)}"
            )

            summaries.append(SourceSummary(
                source_id=source_id,
                lane=lane,
                timestamp="",
                entities=sorted(entities),
                relations=sorted(relations),
                conditions=sorted(conditions),
                n_facts=len(chunk),
                text_digest=digest,
                bm25_tokens=bm25_tokens,
            ))

        return summaries

    # ------------------------------------------------------------------
    # Digest and salience helpers
    # ------------------------------------------------------------------

    def _generate_text_digest(
        self, entity: str, facts: List[_FactRecord]
    ) -> str:
        """Rule-based ~30 token digest.

        Example: "person with 5 relations: works_at acme, lives_in sf, has_skill python"
        """
        # Count unique relation types where entity is subject
        subj_rels: Dict[str, List[str]] = defaultdict(list)
        for f in facts:
            if f.subject == entity:
                subj_rels[f.relation].append(f.object)

        n_rels = len(subj_rels)

        # Pick top relations by frequency
        sorted_rels = sorted(subj_rels.items(), key=lambda x: len(x[1]), reverse=True)
        snippets: List[str] = []
        for rel, targets in sorted_rels[:3]:
            # Pick the first target as representative
            snippets.append(f"{rel} {targets[0]}")

        if snippets:
            detail = ", ".join(snippets)
            return f"{entity} with {n_rels} relations: {detail}"
        else:
            # Entity only appears as object
            n_total = len(facts)
            return f"{entity} referenced in {n_total} facts"

    def _compute_salience(
        self, entity: str, facts: List[_FactRecord]
    ) -> float:
        """Compute salience score in [0, 1].

        Based on:
        - fact_count: more facts = more salient (log-scaled, saturates ~50)
        - relation_diversity: more diverse relations = more salient
        - shard_spread: appearing in more shards indicates importance
        """
        import math

        if not facts:
            return 0.0

        fact_count = len(facts)
        unique_relations: Set[str] = set()
        unique_shards: Set[int] = set()
        for f in facts:
            unique_relations.add(f.relation)
            unique_shards.add(f.shard_idx)

        # Fact count score: log-scaled, saturates around 50 facts
        fact_score = min(1.0, math.log1p(fact_count) / math.log1p(50))

        # Relation diversity score: more unique relations = higher
        diversity_score = min(1.0, len(unique_relations) / 10.0)

        # Shard spread score: appearing in multiple shards
        spread_score = min(1.0, len(unique_shards) / 5.0)

        # Weighted combination
        salience = 0.5 * fact_score + 0.3 * diversity_score + 0.2 * spread_score
        return round(min(1.0, salience), 4)

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def incremental_update(
        self, store: SummaryStore, entities: List[str]
    ) -> None:
        """Update only the specified entities in an existing store.

        Rebuilds each entity's summary from scratch by scanning relevant
        shards, then replaces the entry in the store. Finishes with a
        BM25 rebuild.
        """
        for entity in entities:
            canonical = entity.strip().lower()
            summary = self.build_entity_summary(canonical)
            store.add_entity_summary(summary)

        store.rebuild_bm25()

    # ------------------------------------------------------------------
    # Internal: extract facts from a shard
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_facts(shard: PixelMemUnit, shard_idx: int) -> List[_FactRecord]:
        """Extract all non-black facts from a shard's relation matrix.

        Returns lightweight _FactRecord objects (avoids full DecodedTriple
        construction for efficiency).
        """
        facts: List[_FactRecord] = []
        n = shard.n

        for i in range(n):
            for j in range(n):
                rgb = tuple(int(x) for x in shard.relation[i, j])
                if rgb == (0, 0, 0):
                    continue

                subj = shard.idx_to_entity.get(i, f"entity_{i}")
                obj = shard.idx_to_entity.get(j, f"entity_{j}")
                relation = shard.color_to_relation.get(rgb, f"rel_{rgb}")
                cond_rgb = tuple(int(x) for x in shard.condition[i, j])
                condition = shard.color_to_condition.get(cond_rgb, "")

                facts.append(_FactRecord(
                    subject=subj,
                    relation=relation,
                    object_=obj,
                    condition=condition,
                    shard_idx=shard_idx,
                ))

        return facts
