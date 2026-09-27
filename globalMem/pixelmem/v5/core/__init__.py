"""V5 core — plugin protocols, pipeline host, shared types.

This subpackage must not import anything from ``pixelmem.v4``.
Any V4 coupling lives in ``pixelmem.v5.plugins`` only.
"""

from pixelmem.v5.core.plugins import (
    Extractor,
    Resolver,
    DerivationRule,
    PromptTemplate,
    LLMCaller,
    PluginSet,
)
from pixelmem.v5.core.pipeline import V5Pipeline
from pixelmem.v5.core.types import (
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
