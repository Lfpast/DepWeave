"""Natural-label reconstruction: hide internal aliases from the LLM.

Converts internal aliases like ``a(1).py`` into human-readable
disambiguated descriptions:

    - "the a.py that defines helper"
    - "the a.py from package pkg_b"
    - "the __init__.py that imports model.py"

Priority:
    1. Symbol-based: "the a.py that defines <distinguishing_symbol>"
    2. Package-based: "the a.py from <parent_dir>"
    3. Dependency-based: "the __init__.py that imports <dep>"
    4. Fallback: just the basename (no alias shown)
    5. Debug mode: show internal alias
"""

from __future__ import annotations

from typing import Optional

from pixelmem.alias_namespace import AliasNamespace
from pixelmem.dependency_graph import DependencyGraph


class NaturalLabeler:
    """Convert internal aliases to LLM-facing natural descriptions.

    Usage::

        labeler = NaturalLabeler(namespace, graph, debug=False)
        labeler.label("a(1).py")     # -> "the a.py that defines other"
        labeler.label("main.py")     # -> "main.py" (no dup, no change)
        labeler.label("__init__.py") # -> "__init__.py" (or disambiguated)
    """

    def __init__(
        self,
        ns: AliasNamespace,
        graph: Optional[DependencyGraph] = None,
        debug: bool = False,
    ) -> None:
        self._ns = ns
        self._graph = graph
        self._debug = debug
        self._cache: dict[str, str] = {}

    def label(self, file_alias: str) -> str:
        """Get the natural label for a file alias.

        If the file has no duplicate basename, returns the basename unchanged.
        If duplicated, produces a disambiguated description.
        In debug mode, always returns the raw internal alias.
        """
        if self._debug:
            return file_alias

        if file_alias in self._cache:
            return self._cache[file_alias]

        result = self._resolve(file_alias)
        self._cache[file_alias] = result
        return result

    def label_symbol(self, sym_alias: str) -> str:
        """Get the natural label for a symbol alias.

        ``"helper@a(1).py"`` -> ``"helper"`` (if unique)
        ``"helper@a(1).py"`` -> ``"helper (in the a.py that defines other)"``
        """
        if self._debug:
            return sym_alias

        if "@" not in sym_alias:
            return sym_alias

        name, file_alias = sym_alias.split("@", 1)

        if not self._ns.has_duplicate_symbol(name):
            return name

        file_label = self.label(file_alias)
        return f"{name} (in {file_label})"

    def label_list(self, aliases: list[str]) -> list[str]:
        """Label a list of file aliases."""
        return [self.label(a) for a in aliases]

    def _resolve(self, file_alias: str) -> str:
        """Resolve a file alias to a natural description."""
        try:
            entry = self._ns.file_entry(file_alias)
        except KeyError:
            return file_alias

        bn = entry.basename

        # No duplicate? Just use basename.
        if not self._ns.has_duplicate_basename(bn):
            return bn

        # Strategy 1: Symbol-based ("the a.py that defines helper")
        label = self._try_symbol_based(file_alias, entry)
        if label:
            return label

        # Strategy 2: Package-based ("a.py from pkg_b")
        label = self._try_package_based(file_alias, entry)
        if label:
            return label

        # Strategy 3: Dependency-based ("the __init__.py that imports model")
        label = self._try_dependency_based(file_alias, entry)
        if label:
            return label

        # Fallback: basename (may be ambiguous but better than exposing alias)
        return bn

    def _try_symbol_based(self, file_alias: str, entry) -> Optional[str]:
        """Try to disambiguate by a unique symbol defined in this file."""
        if not entry.symbols:
            return None

        # Find a symbol that's unique to this file (not in other files with same basename)
        siblings = self._ns.aliases_for_basename(entry.basename)
        sibling_syms: set[str] = set()
        for sib in siblings:
            if sib != file_alias:
                sib_entry = self._ns.file_entry(sib)
                sibling_syms.update(sib_entry.symbols)

        unique_syms = [s for s in entry.symbols if s not in sibling_syms]
        if unique_syms:
            # Pick shortest unique symbol
            best = min(unique_syms, key=len)
            return f"the {entry.basename} that defines {best}"

        return None

    def _try_package_based(self, file_alias: str, entry) -> Optional[str]:
        """Try to disambiguate by parent directory."""
        if not entry.parent_dir:
            return None

        # Check if parent_dir is unique among siblings
        siblings = self._ns.aliases_for_basename(entry.basename)
        sibling_parents = set()
        for sib in siblings:
            if sib != file_alias:
                sibling_parents.add(self._ns.file_entry(sib).parent_dir)

        if entry.parent_dir not in sibling_parents:
            return f"{entry.basename} (in {entry.parent_dir}/)"

        return None

    def _try_dependency_based(self, file_alias: str, entry) -> Optional[str]:
        """Try to disambiguate by what this file imports."""
        if not self._graph:
            return None

        deps = self._graph.dependencies_of(file_alias)
        if not deps:
            # Try reverse: what depends on this file?
            rev = self._graph.dependents_of(file_alias)
            if rev:
                dep_name = self._ns.file_entry(rev[0].source).basename
                return f"the {entry.basename} imported by {dep_name}"
            return None

        # Pick the most distinctive dependency
        dep_name = self._ns.file_entry(deps[0].target).basename
        return f"the {entry.basename} that imports {dep_name}"

    def format_file_table(self, file_aliases: list[str]) -> str:
        """Format a file table for LLM consumption.

        Returns a numbered list with natural labels:
            1. main.py
            2. the a.py that defines helper
            3. the a.py from package pkg_b
        """
        lines = []
        for i, alias in enumerate(file_aliases):
            lines.append(f"  {i+1}. {self.label(alias)}")
        return "\n".join(lines)
