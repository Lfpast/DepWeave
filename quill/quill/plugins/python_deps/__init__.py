"""Default python-deps plugin set — wraps V4 components.

This is the single place where V5 depends on V4. The adapters translate
between V5's `Primitive` / `EvidenceBundle` and V4's `Triple` / internal
objects, so the rest of V5 stays V4-agnostic.
"""

from quill.plugins.python_deps.default import build_python_deps_plugins

__all__ = ["build_python_deps_plugins"]
