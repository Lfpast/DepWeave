"""PixelMem V5 — plugin-based pipeline with LLM-driven tool synthesis.

V5 is isolated from V4. Nothing in ``quill/core/`` imports V4.
V4 is consumed only by the default python-deps plugin set under
``quill/plugins/python_deps/`` as a backwards-compatible adapter.

Public API::

    from quill import V5Pipeline, TaskCard, TestHarness

See ``docs/v5_plan.md`` for the overall architecture.
"""

from quill.core.pipeline import V5Pipeline
from quill.core.types import (
    Primitive,
    EvidenceBundle,
    TaskSpec,
    Example,
    PipelineStats,
)
from quill.task_card import TaskCard
from quill.harness import TestHarness, EvalReport, FailureCase

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
