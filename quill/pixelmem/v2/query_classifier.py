"""Query Mode Classifier — cheap heuristic-first classification.

Classifies queries into modes that determine retrieval strategy:
  - fact_lookup: "What degree did I graduate with?"
  - temporal_lookup: "When did I start working at Acme?"
  - temporal_diff: "How many weeks ago did I attend the sale?"
  - multi_session_aggregate: "How many weeks to watch all MCU + Star Wars?"
  - preference_lookup: "What kind of hotels do I prefer?"
  - recommendation_bridge: "Can you suggest a hotel for my trip?"
  - assistant_content_lookup: "What color was the Plesiosaur?"
  - update_or_conflict_check: "Did Rachel move recently?"
  - count_query: "How many Korean restaurants have I tried?"
  - relation_query: "What is the relationship between X and Y?"

Heuristic patterns handle ~80% of queries. LLM fallback only when
confidence < threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryMode(Enum):
    FACT_LOOKUP = "fact_lookup"
    TEMPORAL_LOOKUP = "temporal_lookup"
    TEMPORAL_DIFF = "temporal_diff"
    MULTI_SESSION_AGGREGATE = "multi_session_aggregate"
    PREFERENCE_LOOKUP = "preference_lookup"
    RECOMMENDATION_BRIDGE = "recommendation_bridge"
    ASSISTANT_CONTENT_LOOKUP = "assistant_content_lookup"
    UPDATE_OR_CONFLICT = "update_or_conflict_check"
    COUNT_QUERY = "count_query"
    RELATION_QUERY = "relation_query"
    GENERAL = "general"


@dataclass
class ClassificationResult:
    mode: QueryMode
    confidence: float  # 0.0 - 1.0
    extracted_entities: list[str] = field(default_factory=list)
    extracted_relations: list[str] = field(default_factory=list)
    temporal_refs: list[str] = field(default_factory=list)
    reasoning: str = ""  # short explanation

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "confidence": round(self.confidence, 2),
            "entities": self.extracted_entities,
            "relations": self.extracted_relations,
            "temporal_refs": self.temporal_refs,
        }


# ── Pattern definitions ─────────────────────────────────────────

_TEMPORAL_DIFF_PATTERNS = [
    re.compile(r'how (?:many|long|much time).*(?:ago|since|before)', re.I),
    re.compile(r'how (?:many )?(?:weeks?|days?|months?|years?).*(?:ago|since)', re.I),
    re.compile(r'(?:weeks?|days?|months?) ago', re.I),
    re.compile(r'how much (?:more|less).*(?:spent|paid|cost)', re.I),
    re.compile(r'how much (?:more|less) did', re.I),
    re.compile(r'(?:compared|comparison) to', re.I),
    re.compile(r'(?:more|less) .* (?:than|vs|versus|compared)', re.I),
]

_TEMPORAL_LOOKUP_PATTERNS = [
    re.compile(r'\bwhen did\b', re.I),
    re.compile(r'\bwhat (?:date|time|day)\b', re.I),
    re.compile(r'\bhow long (?:is|does|did)\b', re.I),
    re.compile(r'\bwhat time\b', re.I),
]

_COUNT_PATTERNS = [
    re.compile(r'\bhow many\b', re.I),
    re.compile(r'\bhow much\b.*(?:total|altogether|in total)', re.I),
    re.compile(r'\bcount\b|\btotal\b', re.I),
]

_AGGREGATE_PATTERNS = [
    re.compile(r'\btotal\b.*\band\b', re.I),
    re.compile(r'\ball\b.*\band\b.*(?:how|total)', re.I),
    re.compile(r'\bcombined\b|\boverall\b|\baltogether\b', re.I),
    re.compile(r'(?:how (?:many|long|much)).*\band\b', re.I),
    re.compile(r'\ball (?:the )?.*(?:movies?|films?|books?|shows?)\b.*\band\b', re.I),
]

_PREFERENCE_PATTERNS = [
    re.compile(r'\bprefer\b|\bfavorite\b|\busually\b|\balways\b', re.I),
    re.compile(r'\bwhat (?:kind|type|sort)\b.*\bdo I\b', re.I),
    re.compile(r'\bwhat do I (?:like|prefer|enjoy)\b', re.I),
]

_RECOMMENDATION_PATTERNS = [
    re.compile(r'\b(?:can you |could you )?(?:suggest|recommend)\b', re.I),
    re.compile(r'\bwhat (?:should|would|could) (?:I|you)\b', re.I),
    re.compile(r'\bany (?:suggestions?|recommendations?)\b', re.I),
]

_ASSISTANT_CONTENT_PATTERNS = [
    re.compile(r'\bprevious (?:chat|conversation)\b', re.I),
    re.compile(r'\byou (?:told|said|mentioned|recommended|suggested)\b', re.I),
    re.compile(r'\bremind me of\b', re.I),
    re.compile(r'\bwhat (?:color|name|title)\b.*\b(?:story|book|recipe)\b', re.I),
    re.compile(r'\bshift rotation\b|\bschedule\b.*\byou\b', re.I),
]

_UPDATE_PATTERNS = [
    re.compile(r'\bdid .* (?:move|change|update|switch)\b', re.I),
    re.compile(r'\b(?:recent|latest|current|new)\b.*(?:address|job|status)', re.I),
    re.compile(r'\bstill\b|\banymore\b|\bnow\b', re.I),
]

_RELATION_PATTERNS = [
    re.compile(r'\brelationship (?:between|of)\b', re.I),
    re.compile(r'\bhow (?:are|is) .* (?:related|connected)\b', re.I),
    re.compile(r'\bwhat (?:is|was) .* relationship\b', re.I),
]

# Entity extraction
_ENTITY_PATTERNS = [
    re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'),  # "Alice Smith"
    re.compile(r'\b\w+_\d+\b'),  # "person_14"
    re.compile(r'"([^"]+)"'),  # quoted strings
]

_TEMPORAL_REFS = re.compile(
    r'\b(?:\d{4}-\d{2}-\d{2}|\d+\s*(?:weeks?|days?|months?|years?)'
    r'|yesterday|today|tomorrow|last (?:week|month|year|night)'
    r'|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*)\b',
    re.I,
)

_PRONOUN_MAP = {"i", "my", "me", "mine", "myself"}


def classify(
    query: str,
    confidence_threshold: float = 0.7,
) -> ClassificationResult:
    """Classify a query into a mode.

    Uses heuristic patterns first. Falls back to LLM only if
    confidence < threshold (not implemented here — caller decides).
    """
    result = _classify_heuristic(query)

    # Extract entities
    for pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(query):
            ent = match.group(0).strip('"').strip()
            if ent.lower() not in _PRONOUN_MAP and len(ent) > 1:
                result.extracted_entities.append(ent.lower())

    # Pronoun → user
    q_lower = query.lower()
    if any(f" {p} " in f" {q_lower} " for p in _PRONOUN_MAP):
        if "user" not in result.extracted_entities:
            result.extracted_entities.append("user")

    # Extract temporal references
    result.temporal_refs = [m.group(0) for m in _TEMPORAL_REFS.finditer(query)]

    return result


def _classify_heuristic(query: str) -> ClassificationResult:
    """Pattern-based classification with confidence scoring."""

    # Check patterns in priority order (most specific first)
    checks = [
        (_TEMPORAL_DIFF_PATTERNS, QueryMode.TEMPORAL_DIFF, 0.9),
        (_AGGREGATE_PATTERNS, QueryMode.MULTI_SESSION_AGGREGATE, 0.8),
        (_COUNT_PATTERNS, QueryMode.COUNT_QUERY, 0.85),
        (_TEMPORAL_LOOKUP_PATTERNS, QueryMode.TEMPORAL_LOOKUP, 0.85),
        (_RECOMMENDATION_PATTERNS, QueryMode.RECOMMENDATION_BRIDGE, 0.85),
        (_PREFERENCE_PATTERNS, QueryMode.PREFERENCE_LOOKUP, 0.8),
        (_ASSISTANT_CONTENT_PATTERNS, QueryMode.ASSISTANT_CONTENT_LOOKUP, 0.8),
        (_UPDATE_PATTERNS, QueryMode.UPDATE_OR_CONFLICT, 0.75),
        (_RELATION_PATTERNS, QueryMode.RELATION_QUERY, 0.85),
    ]

    for patterns, mode, conf in checks:
        for pattern in patterns:
            if pattern.search(query):
                return ClassificationResult(
                    mode=mode, confidence=conf,
                    reasoning=f"matched: {pattern.pattern[:40]}",
                )

    # Default: fact lookup with lower confidence
    return ClassificationResult(
        mode=QueryMode.FACT_LOOKUP, confidence=0.5,
        reasoning="no strong pattern match, defaulting to fact_lookup",
    )


# ── Batch classification ────────────────────────────────────────

def classify_batch(queries: list[str]) -> list[ClassificationResult]:
    """Classify multiple queries."""
    return [classify(q) for q in queries]


# ── Test cases ──────────────────────────────────────────────────

_TEST_CASES = [
    ("How many weeks ago did I attend the Nordstrom sale?", QueryMode.TEMPORAL_DIFF),
    ("What degree did I graduate with?", QueryMode.FACT_LOOKUP),
    ("Where does Alice work?", QueryMode.FACT_LOOKUP),
    ("How many Korean restaurants have I tried?", QueryMode.COUNT_QUERY),
    ("How much more did I spend in Hawaii vs Tokyo?", QueryMode.TEMPORAL_DIFF),
    ("Can you suggest a hotel for my trip?", QueryMode.RECOMMENDATION_BRIDGE),
    ("What kind of hotels do I prefer?", QueryMode.PREFERENCE_LOOKUP),
    ("What color was the Plesiosaur in the children's book?", QueryMode.ASSISTANT_CONTENT_LOOKUP),
    ("Did Rachel move recently?", QueryMode.UPDATE_OR_CONFLICT),
    ("What is the relationship between org_6 and person_14?", QueryMode.RELATION_QUERY),
    ("How many weeks to watch all MCU movies and Star Wars?", QueryMode.MULTI_SESSION_AGGREGATE),
    ("When did I start my new job?", QueryMode.TEMPORAL_LOOKUP),
    ("Remind me of the restaurant in Bandung", QueryMode.ASSISTANT_CONTENT_LOOKUP),
]


def run_self_test() -> dict:
    """Run built-in test cases, return accuracy."""
    correct = 0
    results = []
    for query, expected_mode in _TEST_CASES:
        result = classify(query)
        ok = result.mode == expected_mode
        correct += int(ok)
        results.append({
            "query": query[:50],
            "expected": expected_mode.value,
            "got": result.mode.value,
            "confidence": result.confidence,
            "ok": ok,
        })
    accuracy = correct / len(_TEST_CASES)
    return {"accuracy": accuracy, "correct": correct, "total": len(_TEST_CASES), "results": results}
