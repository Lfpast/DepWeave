"""V3 Read Pipeline — plan → route → execute → bundle → answer.

The LLM does NOT browse indexes or pick entities.
Tools do routing, filtering, graph traversal, and computation.
The LLM only does final reasoning and answer wording.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Optional

from pixelmem.shard_manager import ShardManager

from pixelmem.v3.query_planner import QueryPlanner, RetrievalPlan, PlanStep, QueryMode
from pixelmem.v3 import retrieval_algebra as ra
from pixelmem.v3.evidence_bundle import EvidenceBundle, Evidence, Fact
from pixelmem.v3.summary_store import SummaryStore
from pixelmem.v3.cache import PlanCache
from pixelmem.v3.metrics import Metrics


# Map operator names to retrieval_algebra functions.
# Every public function in retrieval_algebra is available for plan dispatch.
_OPERATOR_MAP = {
    # Entity resolution
    "resolve": ra.resolve,
    "resolve_pronoun": ra.resolve_pronoun,
    # Scan operators
    "scan_entity": ra.scan_entity,
    "scan_relation": ra.scan_relation,
    "scan_pair": ra.scan_pair,
    "scan_submatrix": ra.scan_submatrix,
    # Filter operators
    "filter_relation": ra.filter_relation,
    "filter_condition": ra.filter_condition,
    "filter_date_range": ra.filter_date_range,
    "filter_shard": ra.filter_shard,
    # Aggregation
    "count_facts": ra.count_facts,
    "sum_numeric": ra.sum_numeric,
    "date_diff": ra.date_diff,
    "most_recent": ra.most_recent,
    # Graph traversal
    "neighbors": ra.neighbors,
    "path_between": ra.path_between,
    "connected_component": ra.connected_component,
    # Workflow-specific
    "topo_sort": ra.topo_sort,
    "find_subclasses": ra.find_subclasses,
    "find_overrides": ra.find_overrides,
    "resolve_imports": ra.resolve_imports,
    "search_docstrings": ra.search_docstrings,
    # Chunk operators
    "get_chunks": ra.get_chunks,
    "chunk_search": ra.chunk_search,
    # Set operators
    "union": ra.union,
    "intersect": ra.intersect,
    "difference": ra.difference,
    "deduplicate": ra.deduplicate,
    # Projection
    "project": ra.project,
    "group_by": ra.group_by,
}


class ReadPipeline:
    """V3 read path: plan → route → execute → bundle.

    Usage:
        reader = ReadPipeline(mgr, summary_store)
        bundle = reader.query("Which file contains ShardManager?")
        print(bundle.answer_context(budget_tokens=500))
    """

    def __init__(
        self,
        mgr: ShardManager,
        summary_store: SummaryStore,
        cache_size: int = 256,
        cache_ttl: float = 300,
    ):
        self.mgr = mgr
        self.summary_store = summary_store
        self.planner = QueryPlanner(summary_store)
        self.cache = PlanCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self.metrics = Metrics()

    def query(
        self,
        query_text: str,
        budget_tokens: int = 2000,
        use_cache: bool = True,
        expand: bool = False,
    ) -> EvidenceBundle:
        """Full V3 read path.

        Steps:
          1. Cache check
          2. Plan generation (QueryPlanner)
          3. Summary-guided routing (entity validation)
          4. Plan execution (_execute_plan)
          5. Evidence scoring (_score_relevance)
          6. Bundle creation with interned strings
          7. Budget enforcement (_enforce_budget)
          8. Cache result
          9. Record metrics
        """
        with self.metrics.timed("read") as event:
            # 1. Cache check
            if use_cache:
                cached = self.cache.get(query_text)
                if cached is not None:
                    event.cache_hit = True
                    event.n_facts = len(cached.evidences)
                    event.query_mode = cached.mode
                    return cached

            # 2. Generate retrieval plan
            plan = self.planner.plan(query_text)
            event.query_mode = plan.mode.value
            event.plan_id = plan.plan_id

            # 3. Summary-guided entity validation
            validated_entities: list[str] = []
            for ent in plan.entities:
                if self.summary_store.get_entity(ent):
                    validated_entities.append(ent)
                else:
                    # Try fuzzy match via summary search
                    hits = self.summary_store.search_entities(ent, top_k=1)
                    if hits:
                        validated_entities.append(hits[0][1].entity_name)
                    else:
                        # Keep the original -- algebra resolve() handles unknowns
                        validated_entities.append(ent)

            # Update plan entities with validated ones
            if validated_entities:
                plan.entities = validated_entities

            # 4. Execute plan
            evidences = self._execute_plan(plan)
            event.n_operators = len(plan.steps)

            # 5. Score relevance for each evidence
            for ev in evidences:
                ev.relevance = self._score_relevance(
                    ev.fact, plan.query, plan.mode.value,
                )

            # 6. Build evidence bundle with interned strings
            operators_used = list(dict.fromkeys(s.operator for s in plan.steps))
            bundle = EvidenceBundle(
                query=query_text,
                mode=plan.mode.value,
                metadata={
                    "plan_id": plan.plan_id,
                    "operators_used": operators_used,
                    "tool_calls": len(plan.steps),
                    "latency_ms": 0.0,
                    "confidence": plan.confidence,
                    "entities": plan.entities,
                },
            )

            for ev in evidences:
                bundle.add_evidence(ev)

            bundle.deduplicate()

            # 7. Budget enforcement
            bundle = self._enforce_budget(bundle, budget_tokens)

            # 8. Record metrics
            event.n_facts = len(bundle.evidences)
            event.n_entities_scanned = len(set(
                e.fact.subject for e in bundle.evidences
            ) | set(e.fact.object for e in bundle.evidences))
            event.tokens_in = bundle.estimate_tokens()

            # 9. Cache result
            if use_cache:
                plan_hash = hashlib.md5(str(plan.steps).encode()).hexdigest()[:8]
                self.cache.put(query_text, bundle, plan_hash=plan_hash)

        return bundle

    def _execute_plan(self, plan: RetrievalPlan) -> list[Evidence]:
        """Execute plan steps in dependency order.

        Step outputs are stored in an intermediate results table indexed
        by step number.  Later steps reference earlier outputs via ``$N``
        placeholders in their args.  The executor dispatches each operator
        name to the corresponding retrieval_algebra function.
        """
        step_results: dict[int, object] = {}  # step_index -> result
        all_evidences: list[Evidence] = []

        # Operators that need ShardManager as first positional arg
        _MGR_OPS = {
            "resolve", "scan_entity", "scan_relation", "scan_pair",
            "scan_submatrix", "neighbors", "path_between",
            "connected_component", "topo_sort", "find_subclasses",
            "find_overrides", "resolve_imports", "search_docstrings",
            "get_chunks", "chunk_search",
        }

        # Set operators that take two fact lists (a, b)
        _PAIR_OPS = {"union", "intersect", "difference"}

        for i, step in enumerate(plan.steps):
            op_name = step.operator
            op_func = _OPERATOR_MAP.get(op_name)

            if op_func is None:
                step_results[i] = []
                continue

            try:
                # Resolve $N placeholders in args
                args = self._resolve_step_args(step, step_results)

                if op_name in _MGR_OPS:
                    # Inject mgr as first positional arg
                    result = op_func(self.mgr, **args)

                elif op_name in _PAIR_OPS:
                    # Two-list operators: a, b
                    a = args.get("a", [])
                    b = args.get("b", [])
                    result = op_func(a, b)

                elif op_name == "date_diff":
                    result = op_func(
                        args.get("date_a", ""),
                        args.get("date_b", ""),
                    )

                elif op_name == "filter_date_range":
                    facts_input = self._collect_dep_facts(step, step_results)
                    result = op_func(
                        facts_input,
                        args.get("start", "0000-01-01"),
                        args.get("end", "9999-12-31"),
                    )

                elif op_name == "filter_relation":
                    facts_input = self._collect_dep_facts(step, step_results)
                    result = op_func(facts_input, args.get("relation", ""))

                elif op_name == "filter_condition":
                    facts_input = self._collect_dep_facts(step, step_results)
                    result = op_func(facts_input, args.get("pattern", ""))

                elif op_name == "filter_shard":
                    facts_input = self._collect_dep_facts(step, step_results)
                    result = op_func(facts_input, args.get("shard_idx", 0))

                elif op_name == "count_facts":
                    facts_input = self._collect_dep_facts(step, step_results)
                    result = op_func(facts_input, args.get("group_by", "relation"))

                elif op_name == "sum_numeric":
                    facts_input = self._collect_dep_facts(step, step_results)
                    result = op_func(facts_input)

                elif op_name == "project":
                    facts_input = self._collect_dep_facts(step, step_results)
                    result = op_func(facts_input, args.get("fields", ["subject", "object"]))

                elif op_name == "group_by":
                    facts_input = self._collect_dep_facts(step, step_results)
                    result = op_func(facts_input, args.get("key", "relation"))

                elif op_name in ("most_recent", "deduplicate"):
                    facts_input = self._collect_dep_facts(step, step_results)
                    result = op_func(facts_input)

                else:
                    # Unknown operator -- try direct call with args
                    result = op_func(**args)

                step_results[i] = result

                # Collect Evidence from steps that produce facts
                facts = self._coerce_to_ra_facts(result)
                for f in facts:
                    ev_fact = Fact(
                        subject=f.subject,
                        relation=f.relation,
                        object=f.object,
                        condition=f.condition,
                        shard_idx=f.shard_idx,
                        chunk_idx=f.chunk_idx,
                        confidence=f.confidence,
                    )
                    prov = f"step_{i}:{op_name}"
                    if f.shard_idx >= 0:
                        prov += f":shard_{f.shard_idx}"
                    if f.chunk_idx >= 0:
                        prov += f":chunk_{f.chunk_idx}"
                    all_evidences.append(Evidence(
                        fact=ev_fact,
                        relevance=0.0,  # scored later in query()
                        provenance=prov,
                    ))

            except Exception:
                step_results[i] = []

        return all_evidences

    def _resolve_step_args(
        self, step: PlanStep, step_results: dict[int, object],
    ) -> dict[str, object]:
        """Replace ``$N`` placeholders in step args with actual outputs.

        If a resolved value is a list of strings (e.g. from ``resolve``),
        and the downstream expects a scalar entity name, takes the first.
        """
        resolved: dict[str, object] = {}
        for k, v in step.args.items():
            if isinstance(v, str) and v.startswith("$"):
                try:
                    ref_idx = int(v[1:])
                    dep_output = step_results.get(ref_idx, v)
                    if isinstance(dep_output, (list, tuple)) and dep_output:
                        if isinstance(dep_output[0], str):
                            resolved[k] = dep_output[0]
                        else:
                            resolved[k] = dep_output
                    elif isinstance(dep_output, set) and dep_output:
                        resolved[k] = sorted(dep_output)[0]
                    else:
                        resolved[k] = dep_output
                except (ValueError, IndexError):
                    resolved[k] = v
            else:
                resolved[k] = v
        return resolved

    def _collect_dep_facts(
        self, step: PlanStep, step_results: dict[int, object],
    ) -> list[ra.Fact]:
        """Collect ra.Fact lists from all dependency steps, flattening nested lists."""
        facts: list[ra.Fact] = []
        for dep_idx in step.depends_on:
            dep = step_results.get(dep_idx, [])
            if isinstance(dep, list):
                for item in dep:
                    if isinstance(item, ra.Fact):
                        facts.append(item)
                    elif isinstance(item, list):
                        for sub in item:
                            if isinstance(sub, ra.Fact):
                                facts.append(sub)
        return facts

    @staticmethod
    def _coerce_to_ra_facts(result: object) -> list[ra.Fact]:
        """Extract ra.Fact objects from various result types."""
        if isinstance(result, list):
            facts: list[ra.Fact] = []
            for item in result:
                if isinstance(item, ra.Fact):
                    facts.append(item)
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, ra.Fact):
                            facts.append(sub)
            return facts
        return []

    def _score_relevance(self, fact: Fact, query: str, mode: str) -> float:
        """BM25-style relevance scoring."""
        q_tokens = set(re.findall(r'\w+', query.lower()))
        fact_tokens = set()
        for field in [fact.subject, fact.relation, fact.object, fact.condition]:
            fact_tokens.update(re.findall(r'\w+', field.lower()))

        if not q_tokens:
            return 0.5

        overlap = len(q_tokens & fact_tokens)
        score = overlap / len(q_tokens)

        # Boost for direct entity mention
        for token in q_tokens:
            if token in fact.subject or token in fact.object:
                score += 0.2

        return min(1.0, score)

    def _enforce_budget(
        self, bundle: EvidenceBundle, budget: int,
    ) -> EvidenceBundle:
        """Keep top-k evidences within token budget.

        Evidence items are added in relevance order until the budget
        is exhausted.
        """
        if bundle.estimate_tokens() <= budget:
            return bundle

        sorted_ev = sorted(bundle.evidences, key=lambda e: e.relevance, reverse=True)

        trimmed = EvidenceBundle(
            query=bundle.query,
            mode=bundle.mode,
            metadata=dict(bundle.metadata),
        )

        used_tokens = 0
        for ev in sorted_ev:
            line = ev.to_compact_str()
            est = max(1, len(line) // 4)
            if used_tokens + est > budget:
                break
            trimmed.add_evidence(ev)
            used_tokens += est

        trimmed.metadata["budget_used_pct"] = round(
            used_tokens / max(1, budget) * 100, 1,
        )
        return trimmed

    def rebuild_index(self) -> None:
        """Rebuild summary store after writes."""
        from pixelmem.v3.summary_builder import SummaryBuilder
        builder = SummaryBuilder(self.mgr)
        self.summary_store = builder.build_all()
        self.planner = QueryPlanner(self.summary_store)

    def stats(self) -> dict:
        """Pipeline statistics."""
        return {
            "summary_store": self.summary_store.stats,
            "cache": self.cache.stats(),
            "metrics": self.metrics.summary(),
        }
