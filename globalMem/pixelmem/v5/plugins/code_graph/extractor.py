"""Codebase-Memory-style code-graph extractor for V5.

This is the V5 analogue of the recent "code graph" systems (RepoGraph,
CodexGraph, the Tree-Sitter/MCP *Codebase-Memory* paper): it turns a set of
source files into a rich symbol graph — not just imports, but ``calls``,
``inherits``, ``has_method``, ``uses`` and type references — so the LLM can
localize, trace call chains, and reason about dependencies.

The crucial difference is *where the graph lives*. Instead of a Neo4j database
(CodexGraph) or a raw line-level ego-graph dumped into the prompt (RepoGraph),
every edge here is a plain V5 ``Primitive`` quadruple
``(subject, relation, object, condition)``. That means the graph:

* rides the existing V5 pipeline / derivation engine unchanged,
* can be pixel-encoded and cached like any other PixelMem graph, and
* is consumed by the LLM as a *derived, ranked, compact* view (see
  ``index.CodeGraphIndex``), not as a wall of source lines.

Backend: Python's built-in :mod:`ast` (zero new dependencies; matches V4's
existing extraction path). The relation vocabulary is language-neutral, so a
Tree-Sitter backend for other languages can be swapped in behind the same
``Extractor`` protocol and emit identical relations.

Three passes keep symbol resolution honest:

* **register** — assign every definition a qualname; record class methods and
  base classes (so the whole symbol table exists before any edge resolves).
* **structure** — emit ``defines`` / ``has_method`` / ``inherits`` /
  ``annotates`` / ``decorated_by`` / imports.
* **edges** — emit ``calls`` / ``uses``, resolving targets against the table.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Optional

from pixelmem.v5.core.plugins import Extractor
from pixelmem.v5.core.types import Primitive


# ---------------------------------------------------------------------------
# Relation vocabulary — centralised so index.py / plugins.py share the exact
# strings (typo-proof).
# ---------------------------------------------------------------------------

REL_DEFINES = "defines"            # (file, defines, qualname)            cond=kind
REL_HAS_METHOD = "has_method"      # (class_qual, has_method, method_qual)
REL_INHERITS = "inherits"          # (class_qual, inherits, base)         cond=resolved|name
REL_CALLS = "calls"                # (owner_qual, calls, callee)          cond=resolved|self|unresolved
REL_USES = "uses"                  # (owner_qual, uses, symbol_qual)      cond=resolved
REL_IMPORTS_MODULE = "imports_module"   # (file, imports_module, module)
REL_IMPORTS_SYMBOL = "imports_symbol"   # (file, imports_symbol, name)    cond=from-module
REL_ANNOTATES = "annotates"        # (func_qual, annotates, typename)     cond=param|return
REL_DECORATED_BY = "decorated_by"  # (qual, decorated_by, decorator)

# Derived (produced by the derivation engine in plugins.py, never extracted):
REL_FILE_DEPENDS_ON = "file_depends_on"   # (file, file)                  cond=via_call

QUAL_SEP = "::"   # file <-> top-level symbol separator: "pkg/mod.py::func"


def make_qual(file: str, owner_qual: Optional[str], owner_is_module: bool, name: str) -> str:
    """Stable qualified name. Top-level defs are ``file::name``; nested
    defs / methods extend the owner with a dot: ``file::Class.method``."""
    if owner_is_module or owner_qual is None:
        return f"{file}{QUAL_SEP}{name}"
    return f"{owner_qual}.{name}"


# ---------------------------------------------------------------------------
# Resolution context — global symbol table built in pass 1.
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    def_file: dict[str, str] = field(default_factory=dict)              # qual -> file
    class_methods: dict[str, set[str]] = field(default_factory=dict)    # class_qual -> {method name}
    class_bases: dict[str, list[str]] = field(default_factory=dict)     # class_qual -> [base simple name]
    node_qual: dict[int, str] = field(default_factory=dict)             # id(node) -> qual
    modules: dict[str, str] = field(default_factory=dict)             # import name -> path
    imported: dict[str, dict[str, tuple[str, str | None]]] = field(default_factory=dict)
    shadows: dict[str, set[str]] = field(default_factory=dict)
    duplicate_quals: set[str] = field(default_factory=set)

    def register(self, file: str, qual: str) -> None:
        if qual in self.def_file:
            self.duplicate_quals.add(qual)
        self.def_file[qual] = file

    def resolve_export(self, module: str, name: str, seen: Optional[set[str]] = None) -> Optional[str]:
        """Resolve a definition or explicit re-export in an indexed module."""
        path = self.modules.get(module)
        if path is None:
            return None
        seen = seen or set()
        if path in seen:
            return None
        seen.add(path)
        direct = f"{path}{QUAL_SEP}{name}"
        binding = self.imported.get(path, {}).get(name)
        if direct in self.def_file and binding:
            return None
        if direct in self.def_file:
            return direct
        if binding and binding[1] is not None:
            return self.resolve_export(binding[0], binding[1], seen)
        return None

    def resolve_bare(self, path: str, owner: str, name: str) -> Optional[str]:
        if name in self.shadows.get(owner, set()):
            return None
        local = f"{path}{QUAL_SEP}{name}"
        binding = self.imported.get(path, {}).get(name)
        if local in self.def_file and binding:
            return None
        if local in self.def_file:
            return local
        if binding and binding[1] is not None:
            return self.resolve_export(*binding)
        return None

    def resolve_self_method(
        self, class_qual: str, simple: str, _seen: Optional[set] = None,
    ) -> Optional[str]:
        """Resolve ``self.simple()`` against the class MRO (own + base classes).
        Safe: ``self`` is definitely within this class hierarchy."""
        seen = _seen if _seen is not None else set()
        if class_qual in seen:
            return None
        seen.add(class_qual)
        if simple in self.class_methods.get(class_qual, set()):
            return f"{class_qual}.{simple}"
        for base_name in self.class_bases.get(class_qual, []):
            bq = (self.resolve_bare(self.def_file[class_qual], class_qual, base_name)
                  if "." not in base_name else None)
            if bq is not None:
                hit = self.resolve_self_method(bq, simple, seen)
                if hit is not None:
                    return hit
        return None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class CodeGraphExtractor(Extractor):
    """Extract a rich code graph as V5 ``Primitive`` quadruples.

    Pure / deterministic: repeated calls on the same input produce identical,
    de-duplicated, stably-ordered output (the V5 harness relies on this).
    """

    def __init__(self, language: str = "python") -> None:
        if language != "python":
            raise NotImplementedError(
                f"CodeGraphExtractor supports python only (got {language!r}); swap a "
                "Tree-Sitter backend behind this Extractor protocol for other languages."
            )
        self._language = language
        self._n_parse_errors = 0
        self.parse_error_paths: list[str] = []
        self.duplicate_quals: list[str] = []

    # -- Extractor protocol -------------------------------------------------

    def extract(self, documents: dict[str, str], **kwargs: Any) -> list[Primitive]:
        self._n_parse_errors = 0
        self.parse_error_paths = []
        self.duplicate_quals = []
        trees: dict[str, ast.AST] = {}
        for path, src in documents.items():
            try:
                trees[path] = ast.parse(src or "")
            except SyntaxError:
                self._n_parse_errors += 1
                self.parse_error_paths.append(path)

        ctx = _Ctx()
        out: list[Primitive] = []
        order = sorted(trees)
        for path in order:                       # pass 1: register defs
            self._register(path, trees[path], ctx)
        self.duplicate_quals = sorted(ctx.duplicate_quals)
        ctx.modules = _module_paths(order)
        for path in order:
            ctx.imported[path] = _import_bindings(path, trees[path])
        for path in order:                       # pass 2: structural edges
            self._emit_structure(path, trees[path], ctx, out)
        for path in order:                       # pass 3: behavioural edges
            self._collect_edges(path, trees[path], ctx, out)
        return _dedup(out)

    @property
    def n_parse_errors(self) -> int:
        return self._n_parse_errors

    # -- Pass 1: register every definition ---------------------------------

    def _register(self, path: str, tree: ast.AST, ctx: _Ctx) -> None:
        def walk(body, owner_qual, owner_is_module, class_qual):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    cq = make_qual(path, owner_qual, owner_is_module, node.name)
                    ctx.node_qual[id(node)] = cq
                    ctx.register(path, cq)
                    ctx.class_methods.setdefault(cq, set())
                    ctx.class_bases[cq] = [
                        _name_of(b) for b in node.bases
                        if _name_of(b)
                    ]
                    walk(node.body, cq, False, cq)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fq = make_qual(path, owner_qual, owner_is_module, node.name)
                    ctx.node_qual[id(node)] = fq
                    ctx.register(path, fq)
                    ctx.shadows[fq] = _local_names(node)
                    if class_qual is not None:
                        ctx.class_methods[class_qual].add(node.name)
                    walk(node.body, fq, False, None)

        walk(list(getattr(tree, "body", [])), path, True, None)

    # -- Pass 2: structural edges ------------------------------------------

    def _emit_structure(self, path: str, tree: ast.AST, ctx: _Ctx, out: list[Primitive]) -> None:
        def walk(body, class_qual):
            for node in body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._emit_imports(path, node, out)
                elif isinstance(node, ast.ClassDef):
                    cq = ctx.node_qual[id(node)]
                    out.append(Primitive(path, REL_DEFINES, cq, "class",
                                         provenance={"line": node.lineno}))
                    for base in node.bases:
                        full = _name_of(base)
                        if not full:
                            continue
                        bq = ctx.resolve_bare(path, cq, full) if "." not in full else None
                        out.append(Primitive(cq, REL_INHERITS, bq or full,
                                             "resolved" if bq else "name",
                                             provenance={"line": node.lineno}))
                    for dec in node.decorator_list:
                        dn = _name_of(dec)
                        if dn:
                            out.append(Primitive(cq, REL_DECORATED_BY, dn, ""))
                    walk(node.body, cq)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fq = ctx.node_qual[id(node)]
                    kind = "method" if class_qual is not None else "function"
                    out.append(Primitive(path, REL_DEFINES, fq, kind,
                                         provenance={"line": node.lineno,
                                                     "sig": _signature(node)}))
                    if class_qual is not None:
                        out.append(Primitive(class_qual, REL_HAS_METHOD, fq, "",
                                             provenance={"line": node.lineno}))
                    self._emit_annotations(fq, node, out)
                    for dec in node.decorator_list:
                        dn = _name_of(dec)
                        if dn:
                            out.append(Primitive(fq, REL_DECORATED_BY, dn, ""))
                    walk(node.body, None)

        walk(list(getattr(tree, "body", [])), None)

    def _emit_imports(self, path: str, node: ast.AST, out: list[Primitive]) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(Primitive(path, REL_IMPORTS_MODULE, alias.name, "",
                                     provenance={"line": node.lineno, "alias": alias.asname}))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ("." * (node.level or 0))
            out.append(Primitive(path, REL_IMPORTS_MODULE, module, "from",
                                 provenance={"line": node.lineno}))
            for alias in node.names:
                if alias.name != "*":
                    out.append(Primitive(path, REL_IMPORTS_SYMBOL, alias.name, module,
                                         provenance={"line": node.lineno, "alias": alias.asname}))

    def _emit_annotations(self, fqual: str, node: ast.AST, out: list[Primitive]) -> None:
        args = node.args  # type: ignore[attr-defined]
        for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            if a.annotation is not None:
                tn = _name_of(a.annotation)
                if tn:
                    out.append(Primitive(fqual, REL_ANNOTATES, tn, "param"))
        if getattr(node, "returns", None) is not None:
            tn = _name_of(node.returns)
            if tn:
                out.append(Primitive(fqual, REL_ANNOTATES, tn, "return"))

    # -- Pass 3: behavioural edges -----------------------------------------

    def _collect_edges(self, path: str, tree: ast.AST, ctx: _Ctx, out: list[Primitive]) -> None:
        def walk(body, owner_qual, class_qual):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    cq = ctx.node_qual.get(id(node), owner_qual)
                    for stmt in node.body:
                        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            self._region(stmt, path, cq, cq, ctx, out)
                    walk(node.body, cq, cq)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fq = ctx.node_qual.get(id(node), owner_qual)
                    for stmt in node.body:
                        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            self._region(stmt, path, fq, class_qual, ctx, out)
                    walk(node.body, fq, None)
                else:
                    self._region(node, path, owner_qual, class_qual, ctx, out)

        walk(list(getattr(tree, "body", [])), path, None)

    def _region(self, node, path, owner, class_qual, ctx, out) -> None:
        """Walk a statement subtree (NOT descending into nested defs), emitting
        ``calls`` and ``uses`` edges attributed to ``owner``."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.Call):
                self._emit_call(child, path, owner, class_qual, ctx, out)
                for arg in child.args:
                    self._region(arg, path, owner, class_qual, ctx, out)
                for kw in child.keywords:
                    self._region(kw.value, path, owner, class_qual, ctx, out)
                continue
            if isinstance(child, (ast.Name, ast.Attribute)) and \
                    isinstance(getattr(child, "ctx", None), ast.Load):
                self._emit_use(child, path, owner, class_qual, ctx, out)
            self._region(child, path, owner, class_qual, ctx, out)

    def _emit_call(self, call, path, owner, class_qual, ctx, out) -> None:
        target, cond = self._resolve_callable(call.func, path, owner, class_qual, ctx)
        if target is not None:
            out.append(Primitive(owner, REL_CALLS, target, cond,
                                 provenance={"line": getattr(call, "lineno", 0),
                                             "column": getattr(call, "col_offset", 0)}))

    def _emit_use(self, ref, path, owner, class_qual, ctx, out) -> None:
        target, cond = self._resolve_callable(ref, path, owner, class_qual, ctx)
        if target is not None and cond == "resolved" and target != owner:
            out.append(Primitive(owner, REL_USES, target, "resolved",
                                 provenance={"line": getattr(ref, "lineno", 0),
                                             "column": getattr(ref, "col_offset", 0)}))

    def _resolve_callable(self, fn, path, owner, class_qual, ctx) -> tuple[Optional[str], str]:
        """Resolve a call/use target to (qualname_or_bare_name, condition)."""
        if isinstance(fn, ast.Name):
            q = ctx.resolve_bare(path, owner, fn.id)
            return (q, "resolved") if q else (fn.id, "unresolved")
        if isinstance(fn, ast.Attribute):
            simple = fn.attr
            recv = fn.value
            # Resolve ONLY through self/cls (walking the class MRO). Resolving an
            # arbitrary receiver's method by bare name (e.g. ``"x".split()`` -> a
            # user ``split`` method) produces false edges, because we don't track
            # the receiver's type. Conservative: under-link, never mis-link.
            # Receiver-type / import-alias resolution (what CodexGraph's DB buys)
            # is future work.
            if isinstance(recv, ast.Name) and recv.id in ("self", "cls") and class_qual:
                hit = ctx.resolve_self_method(class_qual, simple)
                if hit is not None:
                    return hit, "self"
            dotted = _name_of(fn)
            if dotted:
                parts = dotted.split(".")
                binding = ctx.imported.get(path, {}).get(parts[0])
                if binding:
                    imported_module = binding[0] if binding[1] is None else f"{binding[0]}.{binding[1]}"
                    module = ".".join([imported_module, *parts[1:-1]])
                    hit = ctx.resolve_export(module, parts[-1])
                    if hit:
                        return hit, "resolved"
            return simple, "unresolved"
        return None, "unresolved"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _name_of(node: ast.AST) -> Optional[str]:
    """Best-effort dotted name for Name / Attribute / Subscript / Call nodes."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    return None


def _signature(node: ast.AST) -> str:
    args = node.args  # type: ignore[attr-defined]
    names = [a.arg for a in list(args.posonlyargs) + list(args.args)]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return "(" + ", ".join(names) + ")"


def _module_name(path: str) -> str:
    stem = path[:-3].replace("/", ".")
    return stem[:-9] if stem.endswith(".__init__") else stem


def _module_paths(paths: list[str]) -> dict[str, str]:
    """Index importable suffixes only when they identify one file."""
    candidates: dict[str, set[str]] = {}
    for path in paths:
        parts = _module_name(path).split(".")
        for i in range(len(parts)):
            candidates.setdefault(".".join(parts[i:]), set()).add(path)
    return {module: next(iter(found)) for module, found in candidates.items()
            if len(found) == 1}


def _import_bindings(path: str, tree: ast.AST) -> dict[str, tuple[str, str | None]]:
    package = _module_name(path) if path.endswith("/__init__.py") else _module_name(path).rpartition(".")[0]
    bindings: dict[str, tuple[str, str | None]] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                module = alias.name if alias.asname else local
                bindings[local] = (module, None)
        elif isinstance(node, ast.ImportFrom):
            base = package.split(".") if package else []
            if node.level:
                base = base[:max(0, len(base) - node.level + 1)]
            module = ".".join([*base, *(node.module or "").split(".")]) if node.level else (node.module or "")
            module = module.strip(".")
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = (module, alias.name)
    return bindings


def _local_names(node: ast.AST) -> set[str]:
    """Conservatively avoid binding calls shadowed by parameters or locals."""
    names = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Name) and isinstance(stmt.ctx, (ast.Store, ast.Del)):
            names.add(stmt.id)
    return names


def _dedup(prims: list[Primitive]) -> list[Primitive]:
    """Keep separate source sites for calls and uses."""
    seen: set[tuple] = set()
    out: list[Primitive] = []
    for p in prims:
        site = tuple((p.provenance or {}).get(k) for k in ("line", "column")) if p.relation in (REL_CALLS, REL_USES) else None
        key = (*p.as_tuple(), site)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


__all__ = [
    "CodeGraphExtractor", "make_qual",
    "REL_DEFINES", "REL_HAS_METHOD", "REL_INHERITS", "REL_CALLS", "REL_USES",
    "REL_IMPORTS_MODULE", "REL_IMPORTS_SYMBOL", "REL_ANNOTATES",
    "REL_DECORATED_BY", "REL_FILE_DEPENDS_ON", "QUAL_SEP",
]
