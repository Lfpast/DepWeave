"""V5 core — plugin protocols, pipeline host, shared types.

This subpackage must not import anything from ``pixelmem.v4``.
Any V4 coupling lives in ``quill.plugins`` only.
"""

from quill.core.plugins import (
    Extractor,
    Resolver,
    DerivationRule,
    PromptTemplate,
    LLMCaller,
    PluginSet,
)
from quill.core.pipeline import V5Pipeline
from quill.core.types import (
    Primitive,
    EvidenceBundle,
    TaskSpec,
    Example,
    PipelineStats,
)

__all__ = [
    "Extractor",
    "Resolver",
    "DerivationRule",
    "PromptTemplate",
    "LLMCaller",
    "PluginSet",
    "V5Pipeline",
    "Primitive",
    "EvidenceBundle",
    "TaskSpec",
    "Example",
    "PipelineStats",
]
