"""Quill's task-card pipeline and optional tool synthesis.

The generic pipeline is independent of PixelMem. Benchmark adapters live in
``benchmarks/`` and use the unified ``pixelmem`` package when needed.
"""

from shortmem.pipeline import V5Pipeline
from shortmem.types import (
    Primitive,
    EvidenceBundle,
    TaskSpec,
    Example,
    PipelineStats,
)
from shortmem.task_card import TaskCard
from shortmem.harness import TestHarness, EvalReport, FailureCase

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
