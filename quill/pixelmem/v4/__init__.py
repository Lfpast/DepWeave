"""PixelMem V4 — graph-native retrieval over primitive quadruples.

Redesigned retrieval layer:
  - Primitive quadruples stored simply in PNG matrices
  - Internal alias namespace for deduplication
  - Derived file-level dependency graph (cached)
  - Reconstruction layer hides internal aliases from LLM
  - Graph-first dependency finding

Core PixelMem storage (PNG matrices, quadruples) is unchanged.
"""

from .alias_namespace import AliasNamespace
from .primitive_extractor import extract_primitives
from .dependency_graph import DependencyGraph
from .symbol_resolver import SymbolResolver
from .retrieval import RetrievalPipeline

__all__ = [
    "AliasNamespace",
    "extract_primitives",
    "DependencyGraph",
    "RetrievalPipeline",
]
