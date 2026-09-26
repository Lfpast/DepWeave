"""Canonical alias namespace for deduplication before PixelMem storage.

Assigns deterministic internal aliases for files and symbols so that
duplicate basenames (multiple ``__init__.py``) and duplicate symbol
names (``helper`` in two files) get unique storage identifiers.

Aliases are for **internal** graph reasoning only — they must NOT
appear in the final LLM-facing prompt.

File alias scheme:
    - First occurrence of ``a.py`` stays ``a.py``
    - Second occurrence becomes ``a(1).py``
    - Third becomes ``a(2).py``
    - Ordering is deterministic: sorted by full path.

Symbol alias scheme:
    - ``helper`` in ``a.py``    → ``helper@a.py``
    - ``helper`` in ``a(1).py`` → ``helper@a(1).py``
    - ``Model`` in ``model.py`` → ``Model@model.py``
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class FileEntry:
    """Metadata for a single file in the namespace."""
    original_path: str    # full original path
    basename: str         # e.g. "a.py"
    alias: str            # e.g. "a(1).py" or just "a.py"
    parent_dir: str       # immediate parent directory name
    package_path: str     # dotted package path (e.g. "pkg.sub")
    symbols: list[str] = field(default_factory=list)  # defined function/class names


@dataclass
class SymbolEntry:
    """Metadata for a single symbol (function/class)."""
    name: str             # original name (e.g. "helper")
    alias: str            # internal alias (e.g. "helper@a(1).py")
    file_alias: str       # file it belongs to (internal alias)
    kind: str             # "function", "class", "method", "constant"


class AliasNamespace:
    """Deterministic alias assignment for files and symbols.

    Usage::

        ns = AliasNamespace()
        ns.register_files(file_paths)
        ns.register_symbols(file_alias, symbols)

        # Lookup
        ns.file_alias_for("pkg/sub/a.py")   # -> "a(1).py"
        ns.symbol_alias_for("helper", "a(1).py")  # -> "helper@a(1).py"
        ns.original_path_for("a(1).py")     # -> "pkg/sub/a.py"

    Alias assignment is **stable** for a given set of file paths:
    paths are sorted before numbering, so the same input always
    produces the same aliases.
    """

    def __init__(self) -> None:
        # File bookkeeping
        self._path_to_alias: dict[str, str] = {}
        self._alias_to_entry: dict[str, FileEntry] = {}
        self._basename_groups: dict[str, list[str]] = defaultdict(list)

        # Symbol bookkeeping
        self._symbol_key_to_entry: dict[str, SymbolEntry] = {}  # "name@file_alias"
        self._name_to_aliases: dict[str, list[str]] = defaultdict(list)

        # Repo alias (optional)
        self._repo_alias: str = "repo"

    # ------------------------------------------------------------------
    # File registration
    # ------------------------------------------------------------------

    def register_files(self, paths: list[str]) -> None:
        """Register all file paths and assign aliases.

        Must be called once with ALL files before any symbol registration.
        Paths are sorted for deterministic alias assignment.
        """
        # Group by basename
        groups: dict[str, list[str]] = defaultdict(list)
        for p in sorted(paths):
            bn = os.path.basename(p) if "/" in p or "\\" in p else p
            groups[bn].append(p)

        for bn, group_paths in sorted(groups.items()):
            for idx, path in enumerate(group_paths):
                if len(group_paths) == 1:
                    alias = bn
                else:
                    alias = bn if idx == 0 else f"{_stem(bn)}({idx}){_ext(bn)}"

                parent = _parent_dir(path)
                pkg = _package_path(path)
                entry = FileEntry(
                    original_path=path,
                    basename=bn,
                    alias=alias,
                    parent_dir=parent,
                    package_path=pkg,
                )
                self._path_to_alias[path] = alias
                self._alias_to_entry[alias] = entry
                self._basename_groups[bn].append(alias)

    def register_symbols(
        self,
        file_alias: str,
        symbols: list[tuple[str, str]],
    ) -> None:
        """Register symbols (functions/classes) for a file.

        Args:
            file_alias: Internal file alias (e.g. ``"a(1).py"``).
            symbols: List of ``(name, kind)`` where kind is
                ``"function"``, ``"class"``, ``"method"``, or ``"constant"``.
        """
        if file_alias not in self._alias_to_entry:
            raise ValueError(f"Unknown file alias: {file_alias!r}")

        entry = self._alias_to_entry[file_alias]

        for name, kind in symbols:
            sym_alias = f"{name}@{file_alias}"
            sym_entry = SymbolEntry(
                name=name,
                alias=sym_alias,
                file_alias=file_alias,
                kind=kind,
            )
            self._symbol_key_to_entry[sym_alias] = sym_entry
            self._name_to_aliases[name].append(sym_alias)
            entry.symbols.append(name)

    # ------------------------------------------------------------------
    # File lookups
    # ------------------------------------------------------------------

    def file_alias_for(self, path: str) -> str:
        """Get the internal alias for a full file path."""
        return self._path_to_alias[path]

    def original_path_for(self, alias: str) -> str:
        """Get the original full path from an internal alias."""
        return self._alias_to_entry[alias].original_path

    def file_entry(self, alias: str) -> FileEntry:
        """Get full FileEntry for an alias."""
        return self._alias_to_entry[alias]

    def all_file_aliases(self) -> list[str]:
        """All registered file aliases, sorted."""
        return sorted(self._alias_to_entry.keys())

    def has_duplicate_basename(self, basename: str) -> bool:
        """True if this basename appears more than once."""
        return len(self._basename_groups.get(basename, [])) > 1

    def aliases_for_basename(self, basename: str) -> list[str]:
        """All aliases sharing a basename (e.g. both __init__.py)."""
        return list(self._basename_groups.get(basename, []))

    def duplicate_basenames(self) -> dict[str, list[str]]:
        """Return only basenames that have duplicates."""
        return {
            bn: aliases
            for bn, aliases in self._basename_groups.items()
            if len(aliases) > 1
        }

    # ------------------------------------------------------------------
    # Symbol lookups
    # ------------------------------------------------------------------

    def symbol_alias_for(self, name: str, file_alias: str) -> str:
        """Get the internal alias for a symbol in a specific file."""
        key = f"{name}@{file_alias}"
        if key not in self._symbol_key_to_entry:
            raise KeyError(f"Symbol {name!r} not registered in {file_alias!r}")
        return key

    def symbol_entry(self, sym_alias: str) -> SymbolEntry:
        """Get full SymbolEntry for a symbol alias."""
        return self._symbol_key_to_entry[sym_alias]

    def symbols_in_file(self, file_alias: str) -> list[SymbolEntry]:
        """All symbols registered in a file."""
        return [
            e for e in self._symbol_key_to_entry.values()
            if e.file_alias == file_alias
        ]

    def find_symbol(self, name: str) -> list[SymbolEntry]:
        """Find all symbols with a given name (may be in multiple files)."""
        return [
            self._symbol_key_to_entry[a]
            for a in self._name_to_aliases.get(name, [])
        ]

    def has_duplicate_symbol(self, name: str) -> bool:
        """True if this symbol name appears in multiple files."""
        return len(self._name_to_aliases.get(name, [])) > 1

    # ------------------------------------------------------------------
    # Repo-level
    # ------------------------------------------------------------------

    @property
    def repo_alias(self) -> str:
        return self._repo_alias

    def set_repo_alias(self, name: str) -> None:
        self._repo_alias = name

    # ------------------------------------------------------------------
    # Debug / inspection
    # ------------------------------------------------------------------

    def dump(self) -> dict:
        """Full namespace state for debugging."""
        return {
            "files": {
                alias: {
                    "original_path": e.original_path,
                    "basename": e.basename,
                    "parent_dir": e.parent_dir,
                    "package_path": e.package_path,
                    "symbols": e.symbols,
                }
                for alias, e in sorted(self._alias_to_entry.items())
            },
            "duplicate_basenames": self.duplicate_basenames(),
            "symbols": {
                alias: {
                    "name": e.name,
                    "file_alias": e.file_alias,
                    "kind": e.kind,
                }
                for alias, e in sorted(self._symbol_key_to_entry.items())
            },
            "duplicate_symbols": {
                name: aliases
                for name, aliases in self._name_to_aliases.items()
                if len(aliases) > 1
            },
        }

    def __repr__(self) -> str:
        n_files = len(self._alias_to_entry)
        n_syms = len(self._symbol_key_to_entry)
        n_dup_files = sum(1 for g in self._basename_groups.values() if len(g) > 1)
        return f"AliasNamespace({n_files} files, {n_syms} symbols, {n_dup_files} dup basenames)"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _stem(filename: str) -> str:
    """``'a.py'`` -> ``'a'``, ``'__init__.py'`` -> ``'__init__'``."""
    dot = filename.rfind(".")
    return filename[:dot] if dot > 0 else filename


def _ext(filename: str) -> str:
    """``'a.py'`` -> ``'.py'``, ``'Makefile'`` -> ``''``."""
    dot = filename.rfind(".")
    return filename[dot:] if dot > 0 else ""


def _parent_dir(path: str) -> str:
    """Immediate parent directory name, or ``''``."""
    normed = path.replace("\\", "/")
    parts = normed.rstrip("/").rsplit("/", 2)
    return parts[-2] if len(parts) >= 2 else ""


def _package_path(path: str) -> str:
    """Convert ``'pkg/sub/mod.py'`` to ``'pkg.sub'``."""
    normed = path.replace("\\", "/")
    parts = normed.rstrip("/").split("/")
    if len(parts) <= 1:
        return ""
    # Drop filename, join dirs
    return ".".join(parts[:-1])
