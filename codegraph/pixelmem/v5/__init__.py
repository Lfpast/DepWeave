"""PixelMem V5 — plugin-based pipeline with LLM-driven tool synthesis.

V5 is isolated from V4. Nothing in ``pixelmem/v5/core/`` imports V4.
V4 is consumed only by the default python-deps plugin set under
``pixelmem/v5/plugins/python_deps/`` as a backwards-compatible adapter.

Public API::

    from pixelmem.v5 import V5Pipeline, TaskCard, TestHarness

See ``docs/v5_plan.md`` for the overall architecture.
"""

from pixelmem.v5.core.pipeline import V5Pipeline
from pixelmem.v5.core.types import (
    Primitive,
    EvidenceBundle,
    TaskSpec,
    Example,
    PipelineStats,
)
from pixelmem.v5.task_card import TaskCard
from pixelmem.v5.harness import TestHarness, EvalReport, FailureCase

__all__ = [
    "V5Pipeline",
    "Primitive",
    "EvidenceBundle",
    "TaskSpec",
    "Example",
    "PipelineStats",
    "TaskCard",
    "TestHarness",
    "EvalReport",
    "FailureCase",
]
