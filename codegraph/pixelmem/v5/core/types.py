"""Shared data types for V5.

These types are intentionally decoupled from ``pixelmem.v4``. Default plugins
in ``pixelmem.v5.plugins.python_deps`` translate between V5 Primitives and
V4 ``Triple`` objects; nothing in V5 core knows about V4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Primitive:
    """One atomic quadruple — the unit V5 plugins exchange.

    V5's Primitive is a superset of V4's Triple: same four slots, plus an
    optional ``provenance`` dict for debugging and failure classification.
    """

    subject: str
    relation: str
    object: str
    condition: str = ""
    provenance: Optional[dict] = None

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.subject, self.relation, self.object, self.condition)


@dataclass
class EvidenceBundle:
    """Output of the Derivation stage, consumed by PromptTemplate.

    - ``strong``: high-confidence derived facts (shown to the LLM as "confirmed")
    - ``ambiguous``: low-confidence facts (shown only when strong evidence is absent)
    - ``raw_primitives``: subset of primitives included verbatim in the prompt
    - ``ordering_hint``: optional sequence (e.g., topo sort) the LLM may confirm/fix
    """

    strong: list[Primitive] = field(default_factory=list)
    ambiguous: list[Primitive] = field(default_factory=list)
    raw_primitives: list[Primitive] = field(default_factory=list)
    ordering_hint: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Example:
    """One (input, expected_output) pair from a TaskCard."""

    input: dict
    expected_output: Any
    qid: Optional[str] = None


@dataclass
class TaskSpec:
    """In-memory form of a parsed TaskCard — the contract handed to plugins.

    ``input_schema`` and ``query`` are free-form dicts so different domains can
    declare their own shape without V5 core dictating one.
    """

    domain: str
    description: str
    input_schema: dict
    query: dict
    eval_metric: str = "exact_match"
    eval_threshold: float = 0.80
    few_shot: list[Example] = field(default_factory=list)
    holdout: list[Example] = field(default_factory=list)
    options: dict = field(default_factory=dict)


@dataclass
class PipelineStats:
    """Per-query instrumentation returned alongside the prediction."""

    n_primitives: int = 0
    n_strong_edges: int = 0
    n_ambiguous_edges: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    extraction_errors: int = 0
    parse_error: bool = False
    wallclock_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


# Function signatures used by the pipeline. LLMCaller is the only side
# V5 lets the outside world plug in.
LLMFn = Callable[[str], tuple[str, int, int]]
"""``(prompt) -> (completion_text, tokens_in, tokens_out)``.

Deterministic callers (testing) and real API callers share this signature.
"""
