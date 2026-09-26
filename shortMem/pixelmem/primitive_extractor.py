"""Extract primitive quadruples from source files using canonical aliases.

Produces simple, regular facts for PixelMem storage:
  - (repo, contains_file, main.py, repo_level)
  - (main.py, contains_function, run@main.py, symbol_level)
  - (helper@a.py, defined_in, a.py, symbol_level)
  - (main.py, imports_symbol, helper@a(1).py, symbol_level)
  - (main.py, imports_module, os, external)

No verbose natural-language summaries. No absolute paths as entities.
All entities use the canonical aliases from AliasNamespace.
"""

from __future__ import annotations

import ast
import re
from typing import Optional

from pixelmem.triple_extractor import Triple
from pixelmem.alias_namespace import AliasNamespace


# ------------------------------------------------------------------
# Relation vocabulary
# ------------------------------------------------------------------

# Repo level
REL_CONTAINS_FILE = "contains_file"

# File level
REL_CONTAINS_FUNCTION = "contains_function"
REL_CONTAINS_CLASS = "contains_class"
REL_CONTAINS_METHOD = "contains_method"
REL_FILE_TYPE = "file_type"

# Symbol level
REL_DEFINED_IN = "defined_in"
REL_IMPORTS_SYMBOL = "imports_symbol"
REL_IMPORTS_MODULE = "imports_module"
REL_CALLS = "calls"
REL_EXTENDS = "extends"
REL_DEFINES_CONSTANT = "defines_constant"

# Condition / channel tags
COND_REPO = "repo_level"
COND_FILE = "file_level"
COND_SYMBOL = "symbol_level"
COND_EXTERNAL = "external"
COND_INTERNAL = "internal"
COND_INFERRED = "inferred"


# ------------------------------------------------------------------
# Main API
# ------------------------------------------------------------------


def extract_primitives(
    files: list[str],
    file_contents: dict[str, str],
    ns: AliasNamespace,
    language: str = "python",
) -> list[Triple]:
    """Extract primitive quadruples from all files.

    Args:
        files: List of original file paths.
        file_contents: ``{path: source_code}`` dict.
        ns: Pre-built ``AliasNamespace`` with files registered.
        language: Programming language (currently ``"python"`` supported).

    Returns:
        List of ``Triple(subject, relation, object, condition)`` ready
        for PixelMem encoding.
    """
    triples: list[Triple] = []

    # Repo-level: contains_file
    for path in files:
        alias = ns.file_alias_for(path)
        triples.append(Triple(ns.repo_alias, REL_CONTAINS_FILE, alias, COND_REPO))

    # Per-file extraction
    for path in files:
        code = file_contents.get(path, "")
        alias = ns.file_alias_for(path)

        if language == "python":
            file_triples = _extract_python(path, code, alias, ns)
        else:
            file_triples = _extract_generic(path, code, alias, ns)

        triples.extend(file_triples)

    # Cross-file resolution: link imports to actual file symbols
    resolution_triples = _resolve_cross_file_imports(files, file_contents, ns)
    triples.extend(resolution_triples)

    return triples


# ------------------------------------------------------------------
# Python extraction (AST-based)
# ------------------------------------------------------------------


def _extract_python(
    path: str,
    code: str,
    file_alias: str,
    ns: AliasNamespace,
) -> list[Triple]:
    """Extract primitives from a Python file using AST + regex."""
    triples: list[Triple] = []

    # File type
    triples.append(Triple(file_alias, REL_FILE_TYPE, "python", COND_FILE))

    # Try AST parsing
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fall back to regex
        return triples + _extract_python_regex(code, file_alias, ns)

    symbols: list[tuple[str, str]] = []

    for node in ast.iter_child_nodes(tree):
        # Top-level functions
        if isinstance(node, ast.FunctionDef):
            sym_alias = f"{node.name}@{file_alias}"
            symbols.append((node.name, "function"))
            triples.append(Triple(
                file_alias, REL_CONTAINS_FUNCTION, sym_alias, COND_SYMBOL
            ))
            triples.append(Triple(
                sym_alias, REL_DEFINED_IN, file_alias, COND_SYMBOL
            ))

        # Top-level classes
        elif isinstance(node, ast.ClassDef):
            sym_alias = f"{node.name}@{file_alias}"
            symbols.append((node.name, "class"))
            triples.append(Triple(
                file_alias, REL_CONTAINS_CLASS, sym_alias, COND_SYMBOL
            ))
            triples.append(Triple(
                sym_alias, REL_DEFINED_IN, file_alias, COND_SYMBOL
            ))

            # Class bases (extends)
            for base in node.bases:
                base_name = _get_name(base)
                if base_name and base_name not in ("object",):
                    triples.append(Triple(
                        sym_alias, REL_EXTENDS, base_name, COND_SYMBOL
                    ))

            # Methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_alias = f"{node.name}.{item.name}@{file_alias}"
                    triples.append(Triple(
                        sym_alias, REL_CONTAINS_METHOD, method_alias, COND_SYMBOL
                    ))

        # Import statements
        elif isinstance(node, ast.Import):
            for alias_node in node.names:
                mod = alias_node.name
                triples.append(Triple(
                    file_alias, REL_IMPORTS_MODULE, mod, COND_EXTERNAL
                ))

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            is_relative = (node.level or 0) > 0
            cond = COND_INTERNAL if is_relative else COND_EXTERNAL

            if module:
                triples.append(Triple(
                    file_alias, REL_IMPORTS_MODULE, module, cond
                ))

            for alias_node in (node.names or []):
                name = alias_node.name
                if name == "*":
                    continue
                # Store as imports_symbol with the module context
                full_ref = f"{module}.{name}" if module else name
                triples.append(Triple(
                    file_alias, REL_IMPORTS_SYMBOL, full_ref, cond
                ))

        # Top-level assignments (constants)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    sym_alias = f"{target.id}@{file_alias}"
                    symbols.append((target.id, "constant"))
                    triples.append(Triple(
                        file_alias, REL_DEFINES_CONSTANT, sym_alias, COND_SYMBOL
                    ))
                    triples.append(Triple(
                        sym_alias, REL_DEFINED_IN, file_alias, COND_SYMBOL
                    ))

    # Register discovered symbols in the namespace
    ns.register_symbols(file_alias, symbols)

    return triples


def _extract_python_regex(
    code: str,
    file_alias: str,
    ns: AliasNamespace,
) -> list[Triple]:
    """Fallback regex extraction when AST fails."""
    triples: list[Triple] = []
    symbols: list[tuple[str, str]] = []

    for line in code.split("\n"):
        stripped = line.strip()

        # Functions
        m = re.match(r"^def\s+(\w+)\s*\(", stripped)
        if m:
            name = m.group(1)
            sym_alias = f"{name}@{file_alias}"
            symbols.append((name, "function"))
            triples.append(Triple(file_alias, REL_CONTAINS_FUNCTION, sym_alias, COND_SYMBOL))
            triples.append(Triple(sym_alias, REL_DEFINED_IN, file_alias, COND_SYMBOL))

        # Classes
        m = re.match(r"^class\s+(\w+)", stripped)
        if m:
            name = m.group(1)
            sym_alias = f"{name}@{file_alias}"
            symbols.append((name, "class"))
            triples.append(Triple(file_alias, REL_CONTAINS_CLASS, sym_alias, COND_SYMBOL))
            triples.append(Triple(sym_alias, REL_DEFINED_IN, file_alias, COND_SYMBOL))

        # Imports
        m = re.match(r"^from\s+(\.{0,3}[\w.]*)\s+import\s+([\w, *]+)", stripped)
        if m:
            module = m.group(1).lstrip(".")
            is_rel = m.group(1).startswith(".")
            cond = COND_INTERNAL if is_rel else COND_EXTERNAL
            if module:
                triples.append(Triple(file_alias, REL_IMPORTS_MODULE, module, cond))
            for name in m.group(2).split(","):
                name = name.strip()
                if name and name != "*":
                    full_ref = f"{module}.{name}" if module else name
                    triples.append(Triple(file_alias, REL_IMPORTS_SYMBOL, full_ref, cond))
            continue

        m = re.match(r"^import\s+([\w., ]+)", stripped)
        if m:
            for mod in m.group(1).split(","):
                mod = mod.strip()
                if mod and re.match(r"^[\w.]+$", mod):
                    triples.append(Triple(file_alias, REL_IMPORTS_MODULE, mod, COND_EXTERNAL))

    ns.register_symbols(file_alias, symbols)
    return triples


def _extract_generic(
    path: str,
    code: str,
    file_alias: str,
    ns: AliasNamespace,
) -> list[Triple]:
    """Generic extraction for non-Python files (regex-based)."""
    triples: list[Triple] = []
    ext = path.rsplit(".", 1)[-1] if "." in path else ""
    triples.append(Triple(file_alias, REL_FILE_TYPE, ext or "unknown", COND_FILE))

    symbols: list[tuple[str, str]] = []

    for line in code.split("\n"):
        stripped = line.strip()

        # Generic function/class patterns
        m = re.match(r"(?:export\s+)?(?:default\s+)?(?:function|def|fn)\s+(\w+)", stripped)
        if m:
            name = m.group(1)
            sym_alias = f"{name}@{file_alias}"
            symbols.append((name, "function"))
            triples.append(Triple(file_alias, REL_CONTAINS_FUNCTION, sym_alias, COND_SYMBOL))
            triples.append(Triple(sym_alias, REL_DEFINED_IN, file_alias, COND_SYMBOL))

        m = re.match(r"(?:export\s+)?(?:public\s+)?class\s+(\w+)", stripped)
        if m:
            name = m.group(1)
            sym_alias = f"{name}@{file_alias}"
            symbols.append((name, "class"))
            triples.append(Triple(file_alias, REL_CONTAINS_CLASS, sym_alias, COND_SYMBOL))
            triples.append(Triple(sym_alias, REL_DEFINED_IN, file_alias, COND_SYMBOL))

        # Generic import patterns
        m = re.match(r"(?:import|from|require|include|use|using)\s+['\"]?([\w./]+)", stripped)
        if m:
            triples.append(Triple(file_alias, REL_IMPORTS_MODULE, m.group(1), COND_EXTERNAL))

    ns.register_symbols(file_alias, symbols)
    return triples


# ------------------------------------------------------------------
# Cross-file import resolution
# ------------------------------------------------------------------


def _resolve_cross_file_imports(
    files: list[str],
    file_contents: dict[str, str],
    ns: AliasNamespace,
) -> list[Triple]:
    """Resolve imports_symbol references to actual file symbols.

    For each ``imports_symbol`` triple, try to find the symbol in another
    file's registered symbols. If found, replace the generic reference
    with the canonical symbol alias and add an ``internal`` tag.

    This is the key step that connects the symbol-level graph.
    """
    # Build module_name -> file_alias map
    module_map = _build_module_map(files, ns)

    # Build symbol_name -> [(sym_alias, file_alias)] map
    symbol_map: dict[str, list[tuple[str, str]]] = {}
    for file_alias in ns.all_file_aliases():
        for sym in ns.symbols_in_file(file_alias):
            symbol_map.setdefault(sym.name, []).append((sym.alias, file_alias))

    resolved: list[Triple] = []

    for path in files:
        code = file_contents.get(path, "")
        src_alias = ns.file_alias_for(path)

        for line in code.split("\n"):
            stripped = line.strip()

            # from X import Y
            m = re.match(r"^from\s+(\.{0,3}[\w.]*)\s+import\s+([\w, *]+)", stripped)
            if m:
                from_mod = m.group(1).lstrip(".")
                is_relative = m.group(1).startswith(".")
                names = [n.strip() for n in m.group(2).split(",")]

                for name in names:
                    if not name:
                        continue

                    if name == "*":
                        # Wildcard: treat as module-level dependency
                        # from .FlatlandModel import * => depends on FlatlandModel
                        target_alias = _resolve_module_to_file(
                            from_mod, "", module_map, is_relative,
                            src_alias, ns,
                        )
                        if target_alias and target_alias != src_alias:
                            resolved.append(Triple(
                                src_alias,
                                REL_IMPORTS_MODULE,
                                target_alias,
                                COND_INTERNAL,
                            ))
                        continue

                    # Try to find the target file
                    target_alias = _resolve_module_to_file(
                        from_mod, name, module_map, is_relative,
                        src_alias, ns,
                    )
                    if target_alias and target_alias != src_alias:
                        # Check if the imported name is a symbol in that file
                        target_syms = ns.symbols_in_file(target_alias)
                        matching = [s for s in target_syms if s.name == name]
                        if matching:
                            # Resolved: file imports a known symbol
                            resolved.append(Triple(
                                src_alias,
                                REL_IMPORTS_SYMBOL,
                                matching[0].alias,
                                COND_INTERNAL,
                            ))
                        else:
                            # Module resolved but symbol not in definitions
                            resolved.append(Triple(
                                src_alias,
                                REL_IMPORTS_SYMBOL,
                                f"{name}@{target_alias}",
                                COND_INFERRED,
                            ))
                continue

            # import X  /  import X.Y.Z as alias
            m = re.match(r"^import\s+([\w., ]+)", stripped)
            if m:
                for mod in m.group(1).split(","):
                    mod = mod.strip().split(" as ")[0].strip()  # handle `import X as Y`
                    if not mod:
                        continue
                    # Try exact, then last segment, then all suffixes
                    target = module_map.get(mod)
                    if not target:
                        target = module_map.get(mod.split(".")[-1])
                    if not target:
                        # Try intermediate segments (import a.b.c -> try b.c, c)
                        parts = mod.split(".")
                        for i in range(1, len(parts)):
                            target = module_map.get(".".join(parts[i:]))
                            if target:
                                break
                    if target and target != src_alias:
                        resolved.append(Triple(
                            src_alias,
                            REL_IMPORTS_MODULE,
                            target,
                            COND_INTERNAL,
                        ))

    return resolved


def _build_module_map(
    files: list[str],
    ns: AliasNamespace,
) -> dict[str, str]:
    """Map Python module paths to file aliases.

    Produces entries like:
        ``"pkg.sub.mod"`` -> ``"mod.py"``
        ``"sub.mod"``     -> ``"mod.py"``
        ``"mod"``         -> ``"mod.py"``

    Also registers the raw basename stem (without extension) and the
    last path segment, so that deep-path imports like
    ``import models.backbone.dino_vision_transformer`` can match
    a file named ``dino_vision_transformer.py``.
    """
    mod_map: dict[str, str] = {}
    for path in sorted(files):
        alias = ns.file_alias_for(path)
        entry = ns.file_entry(alias)
        bn = entry.basename
        # Stem from original basename (not alias — e.g. "freq" from "freq.py")
        stem = bn.rsplit(".", 1)[0] if "." in bn else bn

        # Direct matches: basename and stem
        mod_map[bn] = alias
        mod_map[stem] = alias

        # For __init__.py, also register the parent directory name
        if bn == "__init__.py" and entry.parent_dir:
            if entry.parent_dir not in mod_map:
                mod_map[entry.parent_dir] = alias

        # Package-path suffix matches from the full original path
        normed = path.replace("\\", "/").replace(".py", "").replace("/", ".")
        parts = normed.split(".")
        for i in range(len(parts)):
            key = ".".join(parts[i:])
            if key not in mod_map:
                mod_map[key] = alias

        # Also register the last segment alone (handles deep paths like
        # `import models.backbone.dino_vision_transformer` matching
        # `dino_vision_transformer.py`)
        last_seg = parts[-1] if parts else ""
        # For __init__, use parent dir instead
        if last_seg == "__init__" and len(parts) >= 2:
            last_seg = parts[-2]
        if last_seg and last_seg not in mod_map:
            mod_map[last_seg] = alias

    return mod_map


def _resolve_module_to_file(
    from_mod: str,
    name: str,
    module_map: dict[str, str],
    is_relative: bool,
    src_alias: str,
    ns: AliasNamespace,
) -> Optional[str]:
    """Try to resolve ``from <from_mod> import <name>`` to a file alias.

    For relative imports (``from .auth import OAuth2``), the ``from_mod``
    is just ``"auth"`` (dots stripped). We try matching it directly as a
    module name, which should hit the module_map entry built from the
    file's basename stem.

    For absolute imports with deep paths (``from pkg.sub.mod import X``),
    we try the full path, then progressively shorter suffixes.
    """
    candidates: list[str] = []

    if name:
        candidates.append(f"{from_mod}.{name}")  # from foo.bar import baz -> foo.bar.baz

    candidates.append(from_mod)  # from foo.bar import baz -> foo.bar (module)

    if name:
        candidates.append(name)  # from foo import bar -> bar (name is a module)

    # For deep paths, try suffix matching:
    # from pkg.sub.mod import X -> try sub.mod, mod
    if from_mod and "." in from_mod:
        parts = from_mod.split(".")
        for i in range(1, len(parts)):
            candidates.append(".".join(parts[i:]))
        # Also try the last segment alone
        candidates.append(parts[-1])

    # For relative imports, the from_mod IS the module name directly
    # (e.g. from .auth -> from_mod="auth", should match "auth" in module_map)
    # This is already covered by candidates above, but ensure we also
    # try matching with common Python naming patterns
    if is_relative and from_mod:
        # from .mask_time_state import X -> try mask_time_state
        candidates.append(from_mod.split(".")[-1])

    for c in candidates:
        if not c:
            continue
        target = module_map.get(c)
        if target and target != src_alias:
            return target

    return None


# ------------------------------------------------------------------
# AST helpers
# ------------------------------------------------------------------


def _get_name(node: ast.expr) -> Optional[str]:
    """Extract a simple name from an AST expression node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
