"""Compiler-inspired symbol resolution for dependency analysis.

Three techniques that fix false edges:

1. **Symbol table with provenance**: Track where each symbol is
   *defined* vs *re-exported*. When __init__.py has
   ``from .submodule import X``, record X as re-exported, not defined.

2. **Re-export tracing**: When file A does ``from pkg import X`` and
   pkg/__init__.py re-exports X from submodule.py, the true edge is
   A -> submodule.py, not A -> __init__.py.

3. **Scope-aware resolution**: Distinguish stdlib ``types`` from local
   ``types.py`` by checking if the import path matches a known stdlib
   module AND the imported symbol is not defined in the local file.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional

from pixelmem.alias_namespace import AliasNamespace


# ── Known stdlib top-level modules ────────────────────────────────

STDLIB_MODULES: frozenset[str] = frozenset({
    "abc", "argparse", "ast", "asyncio", "base64", "bisect",
    "builtins", "calendar", "collections", "configparser",
    "contextlib", "copy", "csv", "ctypes", "dataclasses",
    "datetime", "decimal", "difflib", "email", "enum",
    "functools", "gc", "glob", "hashlib", "heapq", "html",
    "http", "importlib", "inspect", "io", "itertools", "json",
    "keyword", "logging", "math", "multiprocessing", "numbers",
    "operator", "os", "pathlib", "pickle", "platform", "pprint",
    "queue", "random", "re", "secrets", "select", "shelve",
    "shutil", "signal", "socket", "sqlite3", "ssl", "stat",
    "statistics", "string", "struct", "subprocess", "sys",
    "tempfile", "textwrap", "threading", "time", "timeit",
    "tkinter", "token", "tokenize", "trace", "traceback",
    "turtle", "types", "typing", "unicodedata", "unittest",
    "urllib", "uuid", "warnings", "wave", "weakref", "xml",
    "xmlrpc", "zipfile", "zlib",
})


# ── Symbol provenance ─────────────────────────────────────────────


@dataclass
class SymbolProvenance:
    """Where a symbol came from."""
    name: str
    defined_in: Optional[str] = None      # file alias where it's actually defined
    re_exported_by: Optional[str] = None   # file alias that re-exports it (e.g. __init__.py)
    source_import: Optional[str] = None    # the import line that defines/re-exports it


@dataclass
class FileSymbolTable:
    """Symbol table for a single file."""
    file_alias: str
    defined: dict[str, SymbolProvenance] = field(default_factory=dict)     # symbols defined here
    re_exported: dict[str, SymbolProvenance] = field(default_factory=dict)  # symbols re-exported from elsewhere
    imported_modules: list[str] = field(default_factory=list)               # raw module imports


class SymbolResolver:
    """Compiler-style symbol resolution across a set of files.

    Usage::

        resolver = SymbolResolver(namespace)
        resolver.build_tables(files, file_contents)

        # Where is symbol X actually defined?
        provenance = resolver.resolve("FairseqEncoder")
        # -> SymbolProvenance(name="FairseqEncoder", defined_in="fairseq_model.py",
        #                     re_exported_by="__init__.py")

        # Is this import stdlib or local?
        resolver.is_stdlib_import("from types import SimpleNamespace")
        # -> True (types is stdlib AND SimpleNamespace is not defined locally)

        # True dependency target for an import
        resolver.resolve_import_target("composite_encoder.py",
                                       "from fairseq.models import FairseqEncoder")
        # -> "fairseq_model.py" (traces through __init__.py re-export)
    """

    def __init__(self, ns: AliasNamespace) -> None:
        self._ns = ns
        self._tables: dict[str, FileSymbolTable] = {}
        self._module_map: dict[str, str] = {}  # module_path -> file_alias
        self._global_symbols: dict[str, list[SymbolProvenance]] = {}  # name -> provenances

    def build_tables(
        self,
        files: list[str],
        file_contents: dict[str, str],
    ) -> None:
        """Build symbol tables for all files.

        Pass 1: Collect definitions (def/class at module level).
        Pass 2: Collect re-exports (from .X import Y in __init__.py).
        Pass 3: Build global symbol index.
        """
        self._module_map = _build_module_map(files, self._ns)

        # Pass 1: definitions
        for path in files:
            alias = self._ns.file_alias_for(path)
            code = file_contents.get(path, "")
            table = FileSymbolTable(file_alias=alias)

            try:
                tree = ast.parse(code)
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        prov = SymbolProvenance(name=node.name, defined_in=alias)
                        table.defined[node.name] = prov
                    elif isinstance(node, ast.ClassDef):
                        prov = SymbolProvenance(name=node.name, defined_in=alias)
                        table.defined[node.name] = prov
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                prov = SymbolProvenance(name=target.id, defined_in=alias)
                                table.defined[target.id] = prov
            except SyntaxError:
                # Regex fallback
                for line in code.split("\n"):
                    m = re.match(r"^\s*(?:def|class)\s+(\w+)", line.strip())
                    if m:
                        prov = SymbolProvenance(name=m.group(1), defined_in=alias)
                        table.defined[m.group(1)] = prov

            self._tables[alias] = table

        # Pass 2: re-exports (from .X import Y)
        for path in files:
            alias = self._ns.file_alias_for(path)
            code = file_contents.get(path, "")
            table = self._tables[alias]

            for line in code.split("\n"):
                stripped = line.strip()
                m = re.match(r"^from\s+(\.{1,3}[\w.]*)\s+import\s+([\w, *]+)", stripped)
                if not m:
                    continue

                from_mod = m.group(1).lstrip(".")
                names = [n.strip() for n in m.group(2).split(",")]

                # Find the source file
                source_alias = self._resolve_module(from_mod)
                if not source_alias or source_alias == alias:
                    continue

                for name in names:
                    name = name.strip()
                    if not name or name == "*":
                        # Wildcard: re-export everything from source
                        source_table = self._tables.get(source_alias)
                        if source_table:
                            for sym_name, sym_prov in source_table.defined.items():
                                table.re_exported[sym_name] = SymbolProvenance(
                                    name=sym_name,
                                    defined_in=sym_prov.defined_in,
                                    re_exported_by=alias,
                                    source_import=stripped,
                                )
                        continue

                    # Check if name is defined in source
                    source_table = self._tables.get(source_alias)
                    if source_table and name in source_table.defined:
                        table.re_exported[name] = SymbolProvenance(
                            name=name,
                            defined_in=source_alias,
                            re_exported_by=alias,
                            source_import=stripped,
                        )
                    elif source_table and name in source_table.re_exported:
                        # Chained re-export: __init__ re-exports from sub __init__
                        orig = source_table.re_exported[name]
                        table.re_exported[name] = SymbolProvenance(
                            name=name,
                            defined_in=orig.defined_in,
                            re_exported_by=alias,
                            source_import=stripped,
                        )
                    else:
                        # Not found in source definitions — might be defined
                        # dynamically or in a file we don't have
                        table.re_exported[name] = SymbolProvenance(
                            name=name,
                            defined_in=source_alias,
                            re_exported_by=alias,
                            source_import=stripped,
                        )

        # Pass 3: global index
        self._global_symbols.clear()
        for alias, table in self._tables.items():
            for name, prov in table.defined.items():
                self._global_symbols.setdefault(name, []).append(prov)

    # ── Public API ────────────────────────────────────────────────

    def resolve(self, symbol_name: str) -> list[SymbolProvenance]:
        """Find where a symbol is actually defined (across all files)."""
        return self._global_symbols.get(symbol_name, [])

    def resolve_import_target(
        self,
        src_alias: str,
        import_line: str,
    ) -> Optional[str]:
        """Resolve an import line to its TRUE dependency target.

        If the import goes through an __init__.py re-export, traces
        through to the actual defining file.

        Returns:
            The file alias of the true dependency, or None.
        """
        # Parse the import
        m = re.match(r"^from\s+(\.{0,3}[\w.]*)\s+import\s+([\w, *]+)", import_line.strip())
        if m:
            from_mod = m.group(1).lstrip(".")
            is_relative = m.group(1).startswith(".")
            names = [n.strip() for n in m.group(2).split(",")]

            # Check stdlib first
            if not is_relative and self._is_stdlib_module(from_mod):
                # Verify the symbol is not defined locally
                for name in names:
                    name = name.strip()
                    if name and name != "*" and name in self._global_symbols:
                        # Symbol IS defined locally — not stdlib
                        break
                else:
                    return None  # All symbols are stdlib, no local dep

            # Find the direct target module
            target_alias = self._resolve_module(from_mod)
            if not target_alias:
                # Try with name as module
                for name in names:
                    name = name.strip()
                    if name and name != "*":
                        target_alias = self._resolve_module(name)
                        if target_alias:
                            break
            if not target_alias or target_alias == src_alias:
                return None

            # Check if target is an __init__.py that re-exports the symbol
            target_table = self._tables.get(target_alias)
            if target_table:
                for name in names:
                    name = name.strip()
                    if not name or name == "*":
                        continue
                    # Is this symbol re-exported by the target?
                    if name in target_table.re_exported:
                        prov = target_table.re_exported[name]
                        if prov.defined_in and prov.defined_in != src_alias:
                            # Trace to the actual source!
                            return prov.defined_in
                    # Is this symbol defined in the target?
                    if name in target_table.defined:
                        return target_alias

            return target_alias

        # import X / import X.Y.Z as alias
        m = re.match(r"^import\s+([\w.]+)", import_line.strip())
        if m:
            mod = m.group(1).split(" as ")[0].strip()
            if self._is_stdlib_module(mod.split(".")[0]):
                return None
            target = self._resolve_module(mod)
            if target and target != src_alias:
                return target

        return None

    def is_stdlib_import(self, import_line: str) -> bool:
        """Check if an import line references a stdlib module."""
        m = re.match(r"^from\s+([\w.]+)\s+import", import_line.strip())
        if m:
            top_mod = m.group(1).split(".")[0]
            if top_mod in STDLIB_MODULES:
                # Check if any imported name is defined locally
                m2 = re.match(r"^from\s+[\w.]+\s+import\s+([\w, *]+)", import_line.strip())
                if m2:
                    for name in m2.group(1).split(","):
                        name = name.strip()
                        if name and name != "*" and name in self._global_symbols:
                            return False  # locally defined symbol — not stdlib
                return True

        m = re.match(r"^import\s+([\w.]+)", import_line.strip())
        if m:
            top_mod = m.group(1).split(".")[0]
            return top_mod in STDLIB_MODULES

        return False

    def get_file_table(self, file_alias: str) -> Optional[FileSymbolTable]:
        """Get the symbol table for a file."""
        return self._tables.get(file_alias)

    # ── Internal ──────────────────────────────────────────────────

    def _resolve_module(self, mod_path: str) -> Optional[str]:
        """Resolve a module path to a file alias."""
        if not mod_path:
            return None

        # Direct lookup
        target = self._module_map.get(mod_path)
        if target:
            return target

        # Try last segment
        last = mod_path.split(".")[-1]
        target = self._module_map.get(last)
        if target:
            return target

        # Try suffix matching
        parts = mod_path.split(".")
        for i in range(1, len(parts)):
            target = self._module_map.get(".".join(parts[i:]))
            if target:
                return target

        return None

    def _is_stdlib_module(self, top_module: str) -> bool:
        """Check if a top-level module name is stdlib."""
        return top_module in STDLIB_MODULES


# ── Shared module map builder ─────────────────────────────────────

def _build_module_map(
    files: list[str],
    ns: AliasNamespace,
) -> dict[str, str]:
    """Map Python module paths to file aliases (same as primitive_extractor)."""
    mod_map: dict[str, str] = {}
    for path in sorted(files):
        alias = ns.file_alias_for(path)
        entry = ns.file_entry(alias)
        bn = entry.basename
        stem = bn.rsplit(".", 1)[0] if "." in bn else bn

        mod_map[bn] = alias
        mod_map[stem] = alias

        if bn == "__init__.py" and entry.parent_dir:
            if entry.parent_dir not in mod_map:
                mod_map[entry.parent_dir] = alias

        normed = path.replace("\\", "/").replace(".py", "").replace("/", ".")
        parts = normed.split(".")
        for i in range(len(parts)):
            key = ".".join(parts[i:])
            if key not in mod_map:
                mod_map[key] = alias

        last_seg = parts[-1] if parts else ""
        if last_seg == "__init__" and len(parts) >= 2:
            last_seg = parts[-2]
        if last_seg and last_seg not in mod_map:
            mod_map[last_seg] = alias

    return mod_map
