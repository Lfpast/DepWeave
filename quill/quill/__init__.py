"""Quill's task-card pipeline and optional tool synthesis.

The generic pipeline is independent of PixelMem. Benchmark adapters live in
``benchmarks/`` and use the unified ``pixelmem`` package when needed.
"""

from quill.pipeline import V5Pipeline
from quill.types import (
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
