"""Plugin protocols — the contract V5 enforces on every domain.

All plugins are ``typing.Protocol`` — duck-typed. A plugin does NOT need to
inherit from anything; it just needs the right methods.

Benchmark-specific implementations live under ``benchmarks/`` and are wired in
by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from quill.types import EvidenceBundle, Primitive, PipelineStats, TaskSpec


# ---------------------------------------------------------------------------
# Stage 2: Extractor — documents -> primitive quadruples
# ---------------------------------------------------------------------------


@runtime_checkable
class Extractor(Protocol):
    """Turns a document set into primitive quadruples.

    Implementations must be **pure** in the sense that repeated calls on the
    same input produce identical output (the V5 test harness relies on this
    for reproducible failure classification).
    """

    def extract(self, documents: dict[str, str], **kwargs: Any) -> list[Primitive]:
        """Extract primitives.

        Args:
            documents: ``{document_id: text}`` — document_id is whatever the
                domain uses to name a unit (file path, clause id, turn id).
            **kwargs: domain-specific options; core ignores unknown keys.

        Returns:
            List of Primitives. Must not raise on malformed input; instead
            return fewer primitives and surface the count via
            ``extraction_errors`` on the returned stats (if the extractor
            exposes one).
        """
        ...


# ---------------------------------------------------------------------------
# Stage 3: Resolver — strip false positives from extracted primitives
# ---------------------------------------------------------------------------


@runtime_checkable
class Resolver(Protocol):
    """Optional subtractive filter over extracted primitives.

    A resolver may REMOVE primitives but must never invent new ones. This
    keeps the trust boundary tight: the extractor is the source of truth.
    """

    def resolve(
        self,
        primitives: list[Primitive],
        documents: dict[str, str],
    ) -> list[Primitive]:
        ...


# ---------------------------------------------------------------------------
# Stage 4: DerivationRule — chain primitives into derived facts
# ---------------------------------------------------------------------------


@dataclass
class DerivationRule:
    """A single pattern -> derived-fact rule.

    The pattern is a list of "fact shapes" that share variables. Variables
    are strings starting with ``?``. Constants are literal strings.

    Example::

        DerivationRule(
            name="import_defines_dep",
            pattern=[
                ("?X", "imports_symbol", "?Y", "?c1"),
                ("?Y", "defined_in", "?Z", "?c2"),
            ],
            derived=("?X", "depends_on", "?Z", "resolved"),
            confidence=0.90,
        )
    """

    name: str
    pattern: list[tuple[str, str, str, str]]
    derived: tuple[str, str, str, str]
    confidence: float = 0.9
    notes: str = ""


@runtime_checkable
class DerivationEngine(Protocol):
    """Applies a set of DerivationRules to produce an EvidenceBundle."""

    def derive(
        self,
        primitives: list[Primitive],
        rules: list[DerivationRule],
        task: TaskSpec,
    ) -> EvidenceBundle:
        ...


# ---------------------------------------------------------------------------
# Stage 5: PromptTemplate — evidence -> LLM-ready text
# ---------------------------------------------------------------------------


@runtime_checkable
class PromptTemplate(Protocol):
    """Builds the single LLM prompt sent per query.

    Implementations must respect the token budget in
    ``task.options.get("prompt_token_budget", 500)`` — the harness uses this
    to classify ``prompt_parse_fail`` vs ``budget_exceeded`` failures.
    """

    def build(
        self,
        task: TaskSpec,
        query_input: dict,
        evidence: EvidenceBundle,
    ) -> str:
        ...

    def parse(self, completion: str, task: TaskSpec) -> Any:
        """Parse the LLM completion into the Task's declared output type.

        Should raise ``ValueError`` (or subclass) on unparseable output —
        the pipeline catches this and records ``parse_error=True`` in stats.
        """
        ...


# ---------------------------------------------------------------------------
# LLMCaller — only external dependency V5 accepts
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMCaller(Protocol):
    """Wraps whatever LLM backend the user provides.

    The core never talks to any API directly; everything goes through here.
    """

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        """Return ``(completion_text, tokens_in, tokens_out)``.

        Callers should clamp ``prompt`` length themselves if needed; the
        pipeline will not truncate.
        """
        ...


# ---------------------------------------------------------------------------
# PluginSet — the bundle passed to V5Pipeline
# ---------------------------------------------------------------------------


@dataclass
class PluginSet:
    """A complete plugin bundle for one task family.

    ``resolver`` is optional — many domains won't need subtractive filtering.
    ``derivation_engine`` has a sensible default (the one in
    ``quill.derivation.DefaultDerivationEngine``), so most callers
    only need to supply rules.
    """

    extractor: Extractor
    prompt_template: PromptTemplate
    derivation_rules: list[DerivationRule] = field(default_factory=list)
    resolver: Optional[Resolver] = None
    derivation_engine: Optional[DerivationEngine] = None
    name: str = "unnamed"

    def validate(self) -> None:
        """Structural check that required plugins satisfy their Protocols."""
        if not isinstance(self.extractor, Extractor):
            raise TypeError(
                f"{self.name}: extractor must implement Extractor protocol"
            )
        if not isinstance(self.prompt_template, PromptTemplate):
            raise TypeError(
                f"{self.name}: prompt_template must implement PromptTemplate protocol"
            )
        if self.resolver is not None and not isinstance(self.resolver, Resolver):
            raise TypeError(
                f"{self.name}: resolver must implement Resolver protocol"
            )
        if self.derivation_engine is not None and not isinstance(
            self.derivation_engine, DerivationEngine
        ):
            raise TypeError(
                f"{self.name}: derivation_engine must implement DerivationEngine protocol"
            )


# PipelineStats re-exported so plugin authors don't need to dig into types.py
__all__ = [
    "Extractor",
    "Resolver",
    "DerivationRule",
    "DerivationEngine",
    "PromptTemplate",
    "LLMCaller",
    "PluginSet",
    "PipelineStats",
]
