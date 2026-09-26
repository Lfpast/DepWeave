"""Cross-lane routing and merging for PixelMem V3.

When a query touches both conversation-domain facts (e.g. "Alice works at
Acme") and workflow-domain facts (e.g. "main.py imports utils"), the
CrossLaneRouter identifies which shards belong to which domain, scans
both, discovers join keys (shared entities), and merges the results.

Typical usage::

    from pixelmem.v3.cross_lane import CrossLaneRouter

    router = CrossLaneRouter(mgr, summary_store)
    result = router.cross_query("What does Alice's project import?",
                                 entities=["alice"])
    print(result.domain)          # DomainType.MIXED
    print(result.join_keys)       # {"alice"}
    print(result.merged_facts)    # [Fact(...), ...]
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pixelmem.memory import PixelMemUnit
from pixelmem.shard_manager import ShardManager
from pixelmem.v3.evidence_bundle import EvidenceBundle, Evidence
from pixelmem.v3.retrieval_algebra import Fact, scan_entity, union
from pixelmem.v3.summary_store import SummaryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

class DomainType(Enum):
    """High-level domain type for a set of shards or a query."""
    CONVERSATION = "conversation"
    WORKFLOW = "workflow"
    MIXED = "mixed"


# Relation types that indicate workflow / code content
_WORKFLOW_RELATIONS = frozenset({
    "imports", "calls", "extends", "depends_on", "contains_function",
    "contains_class", "configures", "tests", "defines_constant",
    "code_deps", "code_structure", "has_method", "has_parameter",
})

# Relation types that indicate conversational / personal content
_CONVERSATION_RELATIONS = frozenset({
    "works_at", "lives_in", "born_in", "studied_at", "likes", "prefers",
    "has_skill", "knows", "partner_of", "collaborates_with", "manages",
    "attended", "visited", "bought", "is", "is_a",
})


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CrossLaneResult:
    """Result of a cross-lane query.

    Attributes:
        domain: Which domain(s) the query spans.
        conversation_facts: Facts from conversation-domain shards.
        workflow_facts: Facts from workflow-domain shards.
        join_keys: Entities that appear in both domains (bridges).
        merged_facts: Union of conversation and workflow facts.
    """
    domain: DomainType = DomainType.CONVERSATION
    conversation_facts: list[Fact] = field(default_factory=list)
    workflow_facts: list[Fact] = field(default_factory=list)
    join_keys: set[str] = field(default_factory=set)
    merged_facts: list[Fact] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CrossLaneRouter
# ---------------------------------------------------------------------------

class CrossLaneRouter:
    """Route queries across conversation and workflow domains.

    On construction, classifies every shard by its dominant relation types.
    At query time, scans both domains and merges results via shared entities.

    Parameters
    ----------
    mgr : ShardManager
        The shard manager containing all shards.
    summary_store : SummaryStore
        Summary store for BM25-based entity discovery.
    """

    def __init__(self, mgr: ShardManager, summary_store: SummaryStore) -> None:
        self._mgr = mgr
        self._summary_store = summary_store
        # Per-shard domain classification
        self._shard_domains: dict[int, DomainType] = {}
        self._classify_shards()

    # ------------------------------------------------------------------
    # Shard classification
    # ------------------------------------------------------------------

    def _classify_shards(self) -> None:
        """Classify each shard by its dominant relation types.

        Scans the relation colour map of every shard and counts how many
        relation names fall into the workflow vs conversation buckets.
        Shards with relations from both are labelled MIXED.
        """
        self._shard_domains.clear()

        for idx, shard in enumerate(self._mgr.shards):
            wf_count = 0
            conv_count = 0

            for rel_name in shard.relation_color_map:
                canon = rel_name.strip().lower().replace(" ", "_")
                if canon in _WORKFLOW_RELATIONS:
                    wf_count += 1
                elif canon in _CONVERSATION_RELATIONS:
                    conv_count += 1

            if wf_count > 0 and conv_count > 0:
                self._shard_domains[idx] = DomainType.MIXED
            elif wf_count > 0:
                self._shard_domains[idx] = DomainType.WORKFLOW
            else:
                self._shard_domains[idx] = DomainType.CONVERSATION

        logger.debug(
            "Classified %d shards: %s",
            len(self._shard_domains),
            {d.value: sum(1 for v in self._shard_domains.values() if v == d)
             for d in DomainType},
        )

    # ------------------------------------------------------------------
    # Query routing
    # ------------------------------------------------------------------

    def route(self, query: str, entities: list[str]) -> DomainType:
        """Determine which domain a query belongs to.

        Checks the summary store for each entity and examines which
        domain(s) they appear in.

        Parameters
        ----------
        query : str
            The query text (used for BM25 hints if entities are sparse).
        entities : list[str]
            Known entities from the query.

        Returns
        -------
        DomainType
        """
        has_workflow = False
        has_conversation = False

        for entity in entities:
            canonical = entity.strip().lower()
            # Check which shard domains this entity appears in
            for idx, shard in enumerate(self._mgr.shards):
                if canonical in shard.entity_to_idx:
                    domain = self._shard_domains.get(idx, DomainType.CONVERSATION)
                    if domain == DomainType.WORKFLOW:
                        has_workflow = True
                    elif domain == DomainType.CONVERSATION:
                        has_conversation = True
                    else:
                        has_workflow = True
                        has_conversation = True

        if has_workflow and has_conversation:
            return DomainType.MIXED
        if has_workflow:
            return DomainType.WORKFLOW
        return DomainType.CONVERSATION

    # ------------------------------------------------------------------
    # Cross-domain query
    # ------------------------------------------------------------------

    def cross_query(
        self,
        query: str,
        entities: list[str],
        max_per_domain: int = 20,
    ) -> CrossLaneResult:
        """Scan both domains and merge results via shared entities.

        Parameters
        ----------
        query : str
            The original query text.
        entities : list[str]
            Seed entities extracted from the query.
        max_per_domain : int
            Maximum facts to collect per domain.

        Returns
        -------
        CrossLaneResult
        """
        conv_facts: list[Fact] = []
        wf_facts: list[Fact] = []

        # Scan each entity across all shards, partition by domain
        for entity in entities:
            canonical = entity.strip().lower()
            all_facts = scan_entity(self._mgr, canonical)

            for fact in all_facts:
                shard_idx = fact.shard_idx
                domain = self._shard_domains.get(shard_idx, DomainType.CONVERSATION)

                if domain == DomainType.WORKFLOW:
                    wf_facts.append(fact)
                elif domain == DomainType.CONVERSATION:
                    conv_facts.append(fact)
                else:
                    # MIXED shard: add to both
                    conv_facts.append(fact)
                    wf_facts.append(fact)

        # Trim to max_per_domain
        conv_facts = conv_facts[:max_per_domain]
        wf_facts = wf_facts[:max_per_domain]

        # Find join keys: entities that appear in both domains
        conv_entities: set[str] = set()
        for f in conv_facts:
            conv_entities.add(f.subject)
            conv_entities.add(f.object)

        wf_entities: set[str] = set()
        for f in wf_facts:
            wf_entities.add(f.subject)
            wf_entities.add(f.object)

        join_keys = conv_entities & wf_entities

        # Merge via union (deduplicate)
        merged = union(conv_facts, wf_facts)

        # Determine overall domain
        if conv_facts and wf_facts:
            domain = DomainType.MIXED
        elif wf_facts:
            domain = DomainType.WORKFLOW
        else:
            domain = DomainType.CONVERSATION

        logger.debug(
            "Cross-query: %d conv facts, %d wf facts, %d join keys, domain=%s",
            len(conv_facts), len(wf_facts), len(join_keys), domain.value,
        )

        return CrossLaneResult(
            domain=domain,
            conversation_facts=conv_facts,
            workflow_facts=wf_facts,
            join_keys=join_keys,
            merged_facts=merged,
        )

    # ------------------------------------------------------------------
    # Bridge discovery
    # ------------------------------------------------------------------

    def find_bridges(self, entity: str) -> list[tuple[str, str]]:
        """Find entities that bridge the conversation and workflow domains.

        Starting from *entity*, scans its neighbourhood and returns pairs
        ``(bridge_entity, connecting_relation)`` where the bridge entity
        appears in the opposite domain from *entity*.

        Parameters
        ----------
        entity : str
            The seed entity.

        Returns
        -------
        list[tuple[str, str]]
            Each item is ``(bridge_entity_name, relation_via)`` connecting
            the seed entity to the other domain.
        """
        canonical = entity.strip().lower()
        facts = scan_entity(self._mgr, canonical)

        if not facts:
            return []

        # Determine seed entity's primary domain
        seed_domains: set[DomainType] = set()
        for f in facts:
            d = self._shard_domains.get(f.shard_idx, DomainType.CONVERSATION)
            seed_domains.add(d)

        # For each neighbour, check if it lives in the opposite domain
        bridges: list[tuple[str, str]] = []
        seen: set[str] = set()

        for fact in facts:
            # Identify the neighbour
            neighbour = fact.object if fact.subject == canonical else fact.subject
            if neighbour in seen or neighbour == canonical:
                continue
            seen.add(neighbour)

            # Scan the neighbour to find its domains
            neighbour_facts = scan_entity(self._mgr, neighbour, limit=10)
            neighbour_domains: set[DomainType] = set()
            for nf in neighbour_facts:
                nd = self._shard_domains.get(nf.shard_idx, DomainType.CONVERSATION)
                neighbour_domains.add(nd)

            # Is it a bridge? (appears in a domain the seed doesn't)
            if neighbour_domains - seed_domains:
                bridges.append((neighbour, fact.relation))

        logger.debug(
            "find_bridges(%s): found %d bridges", entity, len(bridges)
        )
        return bridges

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def shard_domains(self) -> dict[int, str]:
        """Return a mapping from shard index to domain name."""
        return {k: v.value for k, v in self._shard_domains.items()}

    @property
    def stats(self) -> dict:
        """Summary statistics."""
        counts: dict[str, int] = defaultdict(int)
        for d in self._shard_domains.values():
            counts[d.value] += 1
        return {
            "total_shards": len(self._shard_domains),
            "conversation_shards": counts.get("conversation", 0),
            "workflow_shards": counts.get("workflow", 0),
            "mixed_shards": counts.get("mixed", 0),
        }
