"""Code-graph plugin set for V5 — a Codebase-Memory-style symbol graph that
rides PixelMem's quadruple substrate.

Build a rich code graph (``calls``, ``inherits``, ``has_method``, ``uses``,
imports, type refs) as ``Primitive`` quadruples, navigate it with
token-budgeted ego views (``CodeGraphIndex``), and run it through ``V5Pipeline``
with the ``build_code_graph_plugins`` factory.

Public API::

    from pixelmem.v5.plugins.code_graph import (
        CodeGraphExtractor, CodeGraphIndex, build_code_graph_plugins,
    )
"""

from pixelmem.v5.plugins.code_graph.extractor import CodeGraphExtractor
from pixelmem.v5.plugins.code_graph.index import CodeGraphIndex, render_ego
from pixelmem.v5.plugins.code_graph.plugins import (
    CALL_FILE_DEP_RULE,
    CodeGraphDerivationEngine,
    CodeNavPrompt,
    build_code_graph_plugins,
)

__all__ = [
    "CodeGraphExtractor",
    "CodeGraphIndex",
    "render_ego",
    "CodeGraphDerivationEngine",
    "CodeNavPrompt",
    "CALL_FILE_DEP_RULE",
    "build_code_graph_plugins",
]
