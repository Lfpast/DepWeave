"""Structured retrieval plan generation for PixelMem V3.

Classifies queries into one of 20 modes and generates a step-by-step
retrieval plan (sequence of operators with dependencies) that the
ReadPipeline can execute deterministically.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pixelmem.v3.summary_store import SummaryStore


# =====================================================================
#  QueryMode: 20 retrieval modes
# =====================================================================

class QueryMode(Enum):
    """Twenty retrieval modes covering fact, temporal, relational,
    graph, workflow, and general queries."""

    FACT_LOOKUP = "fact_lookup"
    TEMPORAL_LOOKUP = "temporal_lookup"
    TEMPORAL_DIFF = "temporal_diff"
    MULTI_SESSION_AGGREGATE = "multi_session_aggregate"
    PREFERENCE_LOOKUP = "preference_lookup"
    RECOMMENDATION_BRIDGE = "recommendation_bridge"
    ASSISTANT_CONTENT_LOOKUP = "assistant_content_lookup"
    UPDATE_OR_CONFLICT = "update_or_conflict"
    COUNT_QUERY = "count_query"
    RELATION_QUERY = "relation_query"
    PATH_QUERY = "path_query"
    NEIGHBORHOOD_QUERY = "neighborhood_query"
    COMPARISON_QUERY = "comparison_query"
    EXISTENCE_CHECK = "existence_check"
    LIST_QUERY = "list_query"
    WORKFLOW_LOOKUP = "workflow_lookup"
    WORKFLOW_DEPENDENCY = "workflow_dependency"
    SUBCLASS_LOOKUP = "subclass_lookup"
    CROSS_DOMAIN = "cross_domain"
    GENERAL = "general"


# =====================================================================
#  PlanStep / RetrievalPlan dataclasses
# =====================================================================

@dataclass
class PlanStep:
    """One step in a retrieval plan.

    Attributes:
        operator:    Name of the retrieval_algebra function to call.
        args:        Keyword arguments for the operator.  Values that
                     start with ``$`` refer to outputs of earlier steps
                     (e.g. ``"$0"`` means "use the output of step 0").
        depends_on:  Indices of steps whose output this step consumes.
        description: Human-readable description of what this step does.
    """

    operator: str
    args: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)
    description: str = ""


@dataclass
class RetrievalPlan:
    """Complete retrieval plan produced by the QueryPlanner.

    Attributes:
        plan_id:       Unique identifier for this plan instance.
        query:         The original query text.
        mode:          Classified query mode.
        steps:         Ordered list of PlanStep objects.
        entities:      Entities extracted from the query.
        relations:     Relation hints extracted from the query.
        temporal_refs: Temporal references (dates, relative phrases).
        confidence:    Classification confidence in [0, 1].
        estimated_cost: Rough cost estimate (number of matrix reads).
    """

    plan_id: str = ""
    query: str = ""
    mode: QueryMode = QueryMode.GENERAL
    steps: List[PlanStep] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    temporal_refs: List[str] = field(default_factory=list)
    confidence: float = 0.0
    estimated_cost: int = 1


# =====================================================================
#  Classification patterns (compiled once)
# =====================================================================

# -- New V3 modes (checked first) ------------------------------------

_WORKFLOW_LOOKUP_RE = re.compile(
    r"\b(which file|what function|what class|what module|\.py|\.js|\.ts"
    r"|defined in|declared in|located in|where is .+ defined)\b",
    re.IGNORECASE,
)

_WORKFLOW_DEPENDENCY_RE = re.compile(
    r"\b(import(?:s|ed)?|depends? on|dependenc(?:y|ies)|require[ds]?"
    r"|used by|uses)\b",
    re.IGNORECASE,
)

_SUBCLASS_RE = re.compile(
    r"\b(subclass(?:es)?|inherit(?:s|ed|ance)?|extend(?:s|ed)?|is[_ ]a\b"
    r"|child class|derived from|parent class|base class)\b",
    re.IGNORECASE,
)

_PATH_QUERY_RE = re.compile(
    r"\b(path between|how (?:is|are) .+ connected|connection between"
    r"|link between|reachable from|chain from)\b",
    re.IGNORECASE,
)

_NEIGHBORHOOD_RE = re.compile(
    r"\b(neighbor(?:s|hood)?|related to|adjacent to|connected to"
    r"|what (?:is|are) .+ (?:related|linked|connected) to)\b",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\b(compar(?:e|ing|ison)|difference(?:s)? between|differ(?:s|ent)?"
    r"|vs\.?|versus|similarities|contrast)\b",
    re.IGNORECASE,
)

_EXISTENCE_CHECK_RE = re.compile(
    r"\b(does .+ have|is there|do(?:es)? .+ exist|any .+ with"
    r"|has .+ been|are there)\b",
    re.IGNORECASE,
)

_LIST_QUERY_RE = re.compile(
    r"\b(list (?:all|every|each)|show (?:all|every|each)|enumerate"
    r"|give me (?:all|every)|what are (?:all|the) )\b",
    re.IGNORECASE,
)

# -- Existing V2 modes -----------------------------------------------

_TEMPORAL_DIFF_RE = re.compile(
    r"\b(how long|duration|time between|since when|days? since"
    r"|weeks? since|months? since|elapsed"
    r"|days? ago|weeks? ago|months? ago|years? ago"
    r"|how (?:many|much) (?:more|less).*(?:than|vs|compared))\b",
    re.IGNORECASE,
)

_TEMPORAL_LOOKUP_RE = re.compile(
    r"\b(when|what (?:date|time|day)|last time|first time"
    r"|most recent|latest|earliest|before|after"
    r"|\d{4}[-/]\d{1,2}|yesterday|today|last (?:week|month|year))\b",
    re.IGNORECASE,
)

_COUNT_RE = re.compile(
    r"\b(how many|count(?:ed)?|number of|total (?:of |number )"
    r"|tally|sum of)\b",
    re.IGNORECASE,
)

_PREFERENCE_RE = re.compile(
    r"\b(prefer(?:s|red|ence)?|fav(?:ou?rite)?|like(?:s)? (?:to|best)"
    r"|dislike|hate|always use|never use)\b",
    re.IGNORECASE,
)

_RECOMMENDATION_RE = re.compile(
    r"\b(recommend|suggest|should I|what would|best option|advice"
    r"|alternative to)\b",
    re.IGNORECASE,
)

_ASSISTANT_CONTENT_RE = re.compile(
    r"\b(you (?:said|wrote|told|mentioned|recommended|generated)"
    r"|your (?:response|answer|recommendation|suggestion)"
    r"|previous (?:response|answer))\b",
    re.IGNORECASE,
)

_UPDATE_RE = re.compile(
    r"\b(update|conflict|contradiction|changed|correction|overrid(?:e|den)"
    r"|replace[ds]?|no longer|instead of|actually)\b",
    re.IGNORECASE,
)

_MULTI_SESSION_RE = re.compile(
    r"\b(across (?:all |every )?sessions?|overall|aggregate|combined"
    r"|total across|all (?:conversations?|threads?|sessions?))\b",
    re.IGNORECASE,
)

_RELATION_RE = re.compile(
    r"\b(what (?:is|are) the relation|how (?:does|do) .+ relate"
    r"|relationship between|relation between)\b",
    re.IGNORECASE,
)

_CROSS_DOMAIN_RE = re.compile(
    r"\b(cross[- ]domain|bridge between|connect .+ (?:and|with) .+"
    r"|overlap between|intersection of .+ and)\b",
    re.IGNORECASE,
)

# -- Entity extraction helpers ----------------------------------------

_FILE_PATH_RE = re.compile(r"\b[\w./\\-]+\.(?:py|js|ts|json|yaml|yml|toml|sh|go|rs|c|cpp|h)\b")
_QUOTED_RE = re.compile(r"""[\"']([^\"']+)[\"']""")
_PERSON_N_RE = re.compile(r"\bperson[_ ]?\d+\b", re.IGNORECASE)
_CAPITALIZED_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_CAMEL_CASE_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")  # ShardManager, PixelMemUnit
_PRONOUN_RE = re.compile(r"\b(I|my|me|mine|myself|we|our)\b")

# -- Temporal extraction -----------------------------------------------

_DATE_ISO_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
_DATE_RELATIVE_RE = re.compile(
    r"\b(yesterday|today|tomorrow|last (?:week|month|year|monday|tuesday"
    r"|wednesday|thursday|friday|saturday|sunday)"
    r"|(?:this|next) (?:week|month|year)"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2})\b",
    re.IGNORECASE,
)

# -- Relation keyword hints -------------------------------------------

_RELATION_KEYWORDS = [
    "imports", "extends", "inherits", "treats", "causes", "uses",
    "contains", "defines", "calls", "returns", "creates", "depends_on",
    "belongs_to", "located_in", "prefers", "likes", "dislikes",
    "visited", "bought", "works_at", "lives_in", "member_of",
    "has_method", "subclass_of", "is_a", "alias",
]

_RELATION_KW_RE = re.compile(
    r"\b(" + "|".join(_RELATION_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


# =====================================================================
#  QueryPlanner
# =====================================================================

class QueryPlanner:
    """Generate structured retrieval plans from natural-language queries.

    Usage::

        planner = QueryPlanner(summary_store=store)
        plan = planner.plan("What files does encoder.py import?")
        # plan.mode == QueryMode.WORKFLOW_DEPENDENCY
        # plan.steps == [PlanStep("resolve", ...), ...]

    Parameters
    ----------
    summary_store : SummaryStore, optional
        If provided, used for entity existence checks during planning.
    """

    def __init__(self, summary_store: Optional[SummaryStore] = None) -> None:
        self.summary_store = summary_store

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    def plan(self, query: str) -> RetrievalPlan:
        """Classify, extract, and build a retrieval plan for *query*.

        Steps:
          1. Classify the query mode.
          2. Extract entities, relations, and temporal references.
          3. Select and specialize the plan template.
          4. Return the fully populated RetrievalPlan.
        """
        mode, confidence, reason = self._classify(query)
        entities = self._extract_entities(query)
        relations = self._extract_relations(query)
        temporal_refs = self._extract_temporal(query)

        # Optionally verify entities against SummaryStore
        if self.summary_store is not None:
            verified: List[str] = []
            for ent in entities:
                if self.summary_store.get_entity(ent) is not None:
                    verified.append(ent)
                else:
                    # Keep it anyway -- the algebra's resolve() handles unknowns
                    verified.append(ent)
            entities = verified

        steps = self._build_template(mode, entities, relations, temporal_refs, query)
        estimated_cost = max(1, len(steps) * max(1, len(entities)))

        plan_id = hashlib.md5(
            f"{query}:{mode.value}:{','.join(entities)}".encode()
        ).hexdigest()[:12]

        return RetrievalPlan(
            plan_id=plan_id,
            query=query,
            mode=mode,
            steps=steps,
            entities=entities,
            relations=relations,
            temporal_refs=temporal_refs,
            confidence=confidence,
            estimated_cost=estimated_cost,
        )

    # -----------------------------------------------------------------
    #  Classification
    # -----------------------------------------------------------------

    def _classify(self, query: str) -> Tuple[QueryMode, float, str]:
        """Classify *query* into one of 20 modes.

        Returns (mode, confidence, reason_string).

        Patterns are checked in a deliberate priority order so that
        more specific modes win over general ones.
        """
        q = query.strip()

        # -- V3 new modes (most specific first) -----------------------

        if _SUBCLASS_RE.search(q):
            return (QueryMode.SUBCLASS_LOOKUP, 0.90, "subclass/inheritance keyword")

        if _WORKFLOW_DEPENDENCY_RE.search(q) and _FILE_PATH_RE.search(q):
            return (QueryMode.WORKFLOW_DEPENDENCY, 0.92, "dependency + file path")

        if _WORKFLOW_DEPENDENCY_RE.search(q):
            return (QueryMode.WORKFLOW_DEPENDENCY, 0.85, "dependency keyword")

        if _WORKFLOW_LOOKUP_RE.search(q):
            return (QueryMode.WORKFLOW_LOOKUP, 0.88, "workflow/file lookup keyword")

        if _PATH_QUERY_RE.search(q):
            return (QueryMode.PATH_QUERY, 0.90, "path between entities")

        if _COMPARISON_RE.search(q):
            return (QueryMode.COMPARISON_QUERY, 0.85, "comparison keyword")

        if _NEIGHBORHOOD_RE.search(q):
            return (QueryMode.NEIGHBORHOOD_QUERY, 0.85, "neighborhood keyword")

        if _EXISTENCE_CHECK_RE.search(q):
            return (QueryMode.EXISTENCE_CHECK, 0.88, "existence check pattern")

        if _LIST_QUERY_RE.search(q):
            return (QueryMode.LIST_QUERY, 0.87, "list/enumerate keyword")

        # -- Existing modes -------------------------------------------

        if _TEMPORAL_DIFF_RE.search(q):
            return (QueryMode.TEMPORAL_DIFF, 0.88, "temporal diff keyword")

        if _COUNT_RE.search(q):
            return (QueryMode.COUNT_QUERY, 0.90, "count keyword")

        if _MULTI_SESSION_RE.search(q):
            return (QueryMode.MULTI_SESSION_AGGREGATE, 0.82, "multi-session keyword")

        if _PREFERENCE_RE.search(q):
            return (QueryMode.PREFERENCE_LOOKUP, 0.88, "preference keyword")

        if _RECOMMENDATION_RE.search(q):
            return (QueryMode.RECOMMENDATION_BRIDGE, 0.80, "recommendation keyword")

        if _ASSISTANT_CONTENT_RE.search(q):
            return (QueryMode.ASSISTANT_CONTENT_LOOKUP, 0.82, "assistant content ref")

        if _UPDATE_RE.search(q):
            return (QueryMode.UPDATE_OR_CONFLICT, 0.80, "update/conflict keyword")

        if _TEMPORAL_LOOKUP_RE.search(q):
            return (QueryMode.TEMPORAL_LOOKUP, 0.82, "temporal keyword")

        if _RELATION_RE.search(q):
            return (QueryMode.RELATION_QUERY, 0.85, "relation query keyword")

        if _CROSS_DOMAIN_RE.search(q):
            return (QueryMode.CROSS_DOMAIN, 0.78, "cross-domain keyword")

        # -- Heuristic fallbacks --------------------------------------

        if _FILE_PATH_RE.search(q):
            return (QueryMode.WORKFLOW_LOOKUP, 0.70, "contains file path")

        # Default
        return (QueryMode.GENERAL, 0.50, "no specific pattern matched")

    # -----------------------------------------------------------------
    #  Entity / Relation / Temporal extraction
    # -----------------------------------------------------------------

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entity mentions from *query*.

        Recognises:
          - File paths (e.g. ``encoder.py``)
          - ``person_N`` patterns
          - Quoted strings
          - Capitalized multi-word phrases
          - First-person pronouns mapped to ``user``
        """
        entities: List[str] = []
        seen: set = set()

        def _add(e: str) -> None:
            norm = e.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                entities.append(norm)

        # File paths
        for m in _FILE_PATH_RE.finditer(query):
            _add(m.group(0))

        # person_N
        for m in _PERSON_N_RE.finditer(query):
            _add(m.group(0))

        # Quoted strings
        for m in _QUOTED_RE.finditer(query):
            _add(m.group(1))

        # Capitalized phrases (e.g. "Alice Smith")
        for m in _CAPITALIZED_RE.finditer(query):
            val = m.group(1)
            # Skip common English words
            if val.lower() not in {"which", "what", "where", "who", "how", "when",
                                    "the", "find", "list", "show", "get", "does", "did", "is"}:
                _add(val)

        # CamelCase identifiers (e.g. "ShardManager", "PixelMemUnit")
        for m in _CAMEL_CASE_RE.finditer(query):
            _add(m.group(1))

        # Also extract bare keywords that might be entity names
        # e.g. "numpy", "jwt", "redis" — lowercase important terms > 3 chars
        for word in re.findall(r'\b[a-z]\w{3,}\b', query):
            if word not in {"which", "what", "where", "does", "file", "class",
                            "function", "contain", "import", "have", "that",
                            "from", "with", "this", "about", "many", "some",
                            "their", "there", "here", "than", "then", "when",
                            "only", "also", "just", "like", "into", "over"}:
                _add(word)

        # Pronoun -> user
        if _PRONOUN_RE.search(query):
            _add("user")

        return entities

    def _extract_relations(self, query: str) -> List[str]:
        """Extract relation keyword hints from the query."""
        relations: List[str] = []
        seen: set = set()
        for m in _RELATION_KW_RE.finditer(query):
            r = m.group(1).lower()
            if r not in seen:
                seen.add(r)
                relations.append(r)
        return relations

    def _extract_temporal(self, query: str) -> List[str]:
        """Extract temporal references (dates and relative phrases)."""
        refs: List[str] = []
        seen: set = set()

        for m in _DATE_ISO_RE.finditer(query):
            v = m.group(0)
            if v not in seen:
                seen.add(v)
                refs.append(v)

        for m in _DATE_RELATIVE_RE.finditer(query):
            v = m.group(0).lower()
            if v not in seen:
                seen.add(v)
                refs.append(v)

        return refs

    # -----------------------------------------------------------------
    #  Plan templates per mode
    # -----------------------------------------------------------------

    def _build_template(
        self,
        mode: QueryMode,
        entities: List[str],
        relations: List[str],
        temporal_refs: List[str],
        query: str,
    ) -> List[PlanStep]:
        """Select and specialize a plan template for the given mode."""

        builder = getattr(self, f"_template_{mode.value}", None)
        if builder is not None:
            return builder(entities, relations, temporal_refs, query)
        return self._template_general(entities, relations, temporal_refs, query)

    # -- FACT_LOOKUP ---------------------------------------------------

    def _template_fact_lookup(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        steps: List[PlanStep] = []
        for i, ent in enumerate(entities or ["user"]):
            steps.append(PlanStep(
                "resolve", {"surface": ent}, [], f"resolve '{ent}'",
            ))
            scan_idx = len(steps)
            steps.append(PlanStep(
                "scan_entity", {"entity": f"${scan_idx - 1}"}, [scan_idx - 1],
                f"scan facts for '{ent}'",
            ))
        if relations:
            last = len(steps) - 1
            steps.append(PlanStep(
                "filter_relation", {"relation": relations[0]}, [last],
                f"filter by relation '{relations[0]}'",
            ))
        steps.append(PlanStep(
            "deduplicate", {}, [len(steps) - 1], "deduplicate results",
        ))
        return steps

    # -- TEMPORAL_LOOKUP -----------------------------------------------

    def _template_temporal_lookup(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "user"
        steps = [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0], f"scan facts for '{ent}'"),
            PlanStep("most_recent", {}, [1], "sort by date"),
        ]
        if temporal_refs:
            steps.append(PlanStep(
                "filter_condition", {"pattern": temporal_refs[0]}, [1],
                f"filter by temporal ref '{temporal_refs[0]}'",
            ))
        return steps

    # -- TEMPORAL_DIFF -------------------------------------------------

    def _template_temporal_diff(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "user"
        steps = [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0], f"scan facts for '{ent}'"),
            PlanStep("most_recent", {}, [1], "sort by date"),
        ]
        if len(temporal_refs) >= 2:
            steps.append(PlanStep(
                "date_diff", {"date_a": temporal_refs[0], "date_b": temporal_refs[1]},
                [], f"compute date diff {temporal_refs[0]}..{temporal_refs[1]}",
            ))
        return steps

    # -- MULTI_SESSION_AGGREGATE ---------------------------------------

    def _template_multi_session_aggregate(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "user"
        return [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0", "limit": 200}, [0],
                     f"scan all facts for '{ent}'"),
            PlanStep("count_facts", {"group_by": "relation"}, [1],
                     "aggregate counts by relation"),
        ]

    # -- PREFERENCE_LOOKUP ---------------------------------------------

    def _template_preference_lookup(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "user"
        pref_rels = "prefers|likes|dislikes|fav|preference|always_use"
        return [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0],
                     f"scan facts for '{ent}'"),
            PlanStep("filter_condition", {"pattern": pref_rels}, [1],
                     "filter preference relations"),
            PlanStep("most_recent", {}, [2], "prefer most recent preferences"),
        ]

    # -- RECOMMENDATION_BRIDGE -----------------------------------------

    def _template_recommendation_bridge(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "user"
        return [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0],
                     f"scan facts for '{ent}'"),
            PlanStep("filter_relation", {"relation": "recommend"}, [1],
                     "filter recommendations"),
            PlanStep("deduplicate", {}, [2], "deduplicate"),
        ]

    # -- ASSISTANT_CONTENT_LOOKUP --------------------------------------

    def _template_assistant_content_lookup(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        return [
            PlanStep("resolve", {"surface": "assistant"}, [],
                     "resolve assistant entity"),
            PlanStep("scan_entity", {"entity": "$0"}, [0],
                     "scan assistant-generated facts"),
            PlanStep("most_recent", {}, [1], "sort by recency"),
        ]

    # -- UPDATE_OR_CONFLICT --------------------------------------------

    def _template_update_or_conflict(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "user"
        return [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0],
                     f"scan facts for '{ent}'"),
            PlanStep("most_recent", {}, [1], "sort by date (newest first)"),
            PlanStep("deduplicate", {}, [2], "keep latest version of each fact"),
        ]

    # -- COUNT_QUERY ---------------------------------------------------

    def _template_count_query(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "user"
        group = "relation"
        if relations:
            group = "object"
        return [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0],
                     f"scan facts for '{ent}'"),
            PlanStep("count_facts", {"group_by": group}, [1],
                     f"count grouped by {group}"),
        ]

    # -- RELATION_QUERY ------------------------------------------------

    def _template_relation_query(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        if len(entities) >= 2:
            return [
                PlanStep("resolve", {"surface": entities[0]}, [],
                         f"resolve '{entities[0]}'"),
                PlanStep("resolve", {"surface": entities[1]}, [],
                         f"resolve '{entities[1]}'"),
                PlanStep("scan_pair", {"entity_a": "$0", "entity_b": "$1"},
                         [0, 1], "scan pair for direct relations"),
            ]
        rel = relations[0] if relations else "related_to"
        return [
            PlanStep("scan_relation", {"relation": rel}, [],
                     f"scan all facts with relation '{rel}'"),
            PlanStep("deduplicate", {}, [0], "deduplicate"),
        ]

    # -- PATH_QUERY ----------------------------------------------------

    def _template_path_query(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        a = entities[0] if len(entities) >= 1 else "user"
        b = entities[1] if len(entities) >= 2 else "user"
        return [
            PlanStep("resolve", {"surface": a}, [], f"resolve '{a}'"),
            PlanStep("resolve", {"surface": b}, [], f"resolve '{b}'"),
            PlanStep("path_between", {"entity_a": "$0", "entity_b": "$1",
                                      "max_hops": 3}, [0, 1],
                     f"BFS path from '{a}' to '{b}'"),
        ]

    # -- NEIGHBORHOOD_QUERY --------------------------------------------

    def _template_neighborhood_query(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "user"
        return [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("neighbors", {"entity": "$0", "depth": 1}, [0],
                     f"1-hop neighbors of '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0],
                     f"scan facts for '{ent}'"),
        ]

    # -- COMPARISON_QUERY ----------------------------------------------

    def _template_comparison_query(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        a = entities[0] if len(entities) >= 1 else "user"
        b = entities[1] if len(entities) >= 2 else "user"
        return [
            PlanStep("resolve", {"surface": a}, [], f"resolve '{a}'"),
            PlanStep("resolve", {"surface": b}, [], f"resolve '{b}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0],
                     f"scan facts for '{a}'"),
            PlanStep("scan_entity", {"entity": "$1"}, [1],
                     f"scan facts for '{b}'"),
            PlanStep("difference", {"a": "$2", "b": "$3"}, [2, 3],
                     f"facts in '{a}' not in '{b}'"),
            PlanStep("difference", {"a": "$3", "b": "$2"}, [2, 3],
                     f"facts in '{b}' not in '{a}'"),
            PlanStep("union", {"a": "$4", "b": "$5"}, [4, 5],
                     "combine unique differences"),
        ]

    # -- EXISTENCE_CHECK -----------------------------------------------

    def _template_existence_check(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "user"
        steps = [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0", "limit": 5}, [0],
                     f"check if '{ent}' has facts"),
        ]
        if relations:
            steps.append(PlanStep(
                "filter_relation", {"relation": relations[0]}, [1],
                f"filter by '{relations[0]}'",
            ))
        return steps

    # -- LIST_QUERY ----------------------------------------------------

    def _template_list_query(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        if relations:
            return [
                PlanStep("scan_relation", {"relation": relations[0]}, [],
                         f"scan all '{relations[0]}' facts"),
                PlanStep("project", {"fields": ["subject", "object"]}, [0],
                         "project to subject/object"),
            ]
        ent = entities[0] if entities else "user"
        return [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0", "limit": 100}, [0],
                     f"list all facts for '{ent}'"),
            PlanStep("deduplicate", {}, [1], "deduplicate"),
        ]

    # -- WORKFLOW_LOOKUP -----------------------------------------------

    def _template_workflow_lookup(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "unknown_file"
        return [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0],
                     f"scan facts for '{ent}'"),
            PlanStep("deduplicate", {}, [1], "deduplicate results"),
        ]

    # -- WORKFLOW_DEPENDENCY -------------------------------------------

    def _template_workflow_dependency(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "unknown_file"
        return [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0],
                     f"scan facts for '{ent}'"),
            PlanStep("filter_relation", {"relation": "imports"}, [1],
                     "filter import edges"),
            PlanStep("topo_sort", {"relation": "imports"}, [2],
                     "topological sort of dependencies"),
        ]

    # -- SUBCLASS_LOOKUP -----------------------------------------------

    def _template_subclass_lookup(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        ent = entities[0] if entities else "object"
        return [
            PlanStep("resolve", {"surface": ent}, [], f"resolve '{ent}'"),
            PlanStep("find_subclasses", {"class_name": "$0"}, [0],
                     f"find subclasses of '{ent}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0],
                     f"scan parent class facts"),
        ]

    # -- CROSS_DOMAIN --------------------------------------------------

    def _template_cross_domain(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        a = entities[0] if len(entities) >= 1 else "user"
        b = entities[1] if len(entities) >= 2 else "user"
        return [
            PlanStep("resolve", {"surface": a}, [], f"resolve '{a}'"),
            PlanStep("resolve", {"surface": b}, [], f"resolve '{b}'"),
            PlanStep("scan_entity", {"entity": "$0"}, [0],
                     f"scan facts for '{a}'"),
            PlanStep("scan_entity", {"entity": "$1"}, [1],
                     f"scan facts for '{b}'"),
            PlanStep("intersect", {"a": "$2", "b": "$3"}, [2, 3],
                     "find overlapping facts"),
            PlanStep("union", {"a": "$2", "b": "$3"}, [2, 3],
                     "combine all facts"),
        ]

    # -- GENERAL -------------------------------------------------------

    def _template_general(
        self, entities: List[str], relations: List[str],
        temporal_refs: List[str], query: str,
    ) -> List[PlanStep]:
        tokens = [t.lower() for t in query.split() if len(t) > 2]
        steps: List[PlanStep] = []
        if entities:
            for i, ent in enumerate(entities[:3]):
                base = len(steps)
                steps.append(PlanStep(
                    "resolve", {"surface": ent}, [], f"resolve '{ent}'",
                ))
                steps.append(PlanStep(
                    "scan_entity", {"entity": f"${base}"}, [base],
                    f"scan facts for '{ent}'",
                ))
        else:
            steps.append(PlanStep(
                "chunk_search", {"query_tokens": tokens[:10]}, [],
                "BM25 chunk search over query tokens",
            ))
        steps.append(PlanStep(
            "deduplicate", {}, [len(steps) - 1], "deduplicate results",
        ))
        return steps

    # -----------------------------------------------------------------
    #  Self-test
    # -----------------------------------------------------------------

    def run_self_test(self) -> Dict[str, Any]:
        """Test classification against 20+ example queries.

        Returns a dict with ``total``, ``passed``, ``failed``, and
        ``details`` (list of dicts per test case).
        """
        test_cases: List[Tuple[str, QueryMode]] = [
            # WORKFLOW_LOOKUP
            ("Which file defines the ShardManager class?", QueryMode.WORKFLOW_LOOKUP),
            ("What function is in encoder.py?", QueryMode.WORKFLOW_LOOKUP),
            # WORKFLOW_DEPENDENCY
            ("What does encoder.py import?", QueryMode.WORKFLOW_DEPENDENCY),
            ("What are the dependencies of shard_manager.py?", QueryMode.WORKFLOW_DEPENDENCY),
            # SUBCLASS_LOOKUP
            ("What subclasses does BaseEncoder have?", QueryMode.SUBCLASS_LOOKUP),
            ("Which classes inherit from PixelMemUnit?", QueryMode.SUBCLASS_LOOKUP),
            # PATH_QUERY
            ("What is the path between Alice and Bob?", QueryMode.PATH_QUERY),
            ("How is encoder connected to decoder?", QueryMode.PATH_QUERY),
            # NEIGHBORHOOD_QUERY
            ("What is related to encoder.py?", QueryMode.NEIGHBORHOOD_QUERY),
            ("Show neighbors of Alice", QueryMode.NEIGHBORHOOD_QUERY),
            # COMPARISON_QUERY
            ("Compare encoder.py vs decoder.py", QueryMode.COMPARISON_QUERY),
            ("What is the difference between Alice and Bob?", QueryMode.COMPARISON_QUERY),
            # EXISTENCE_CHECK
            ("Does the user have a preference for Python?", QueryMode.EXISTENCE_CHECK),
            ("Is there a relation between Alice and Bob?", QueryMode.EXISTENCE_CHECK),
            # LIST_QUERY
            ("List all entities that use imports", QueryMode.LIST_QUERY),
            ("Show all preferences", QueryMode.LIST_QUERY),
            # TEMPORAL_DIFF
            ("How long since my last visit to Paris?", QueryMode.TEMPORAL_DIFF),
            # COUNT_QUERY
            ("How many files import encoder.py?", QueryMode.COUNT_QUERY),
            # PREFERENCE_LOOKUP
            ("What is my favourite programming language?", QueryMode.PREFERENCE_LOOKUP),
            # TEMPORAL_LOOKUP
            ("When did I last visit Tokyo?", QueryMode.TEMPORAL_LOOKUP),
            # UPDATE_OR_CONFLICT
            ("Actually, I moved to Berlin", QueryMode.UPDATE_OR_CONFLICT),
            # GENERAL
            ("Tell me about the project", QueryMode.GENERAL),
        ]

        results: List[Dict[str, Any]] = []
        passed = 0

        for query_text, expected_mode in test_cases:
            plan = self.plan(query_text)
            ok = plan.mode == expected_mode
            if ok:
                passed += 1
            results.append({
                "query": query_text,
                "expected": expected_mode.value,
                "got": plan.mode.value,
                "confidence": plan.confidence,
                "passed": ok,
                "entities": plan.entities,
                "steps": len(plan.steps),
            })

        return {
            "total": len(test_cases),
            "passed": passed,
            "failed": len(test_cases) - passed,
            "details": results,
        }
