"""CodeGraphIndex — navigable, token-budgeted views over the code graph.

This is the layer that gives us Codebase-Memory's *interface* (search a symbol,
trace a call chain, find dependencies, locate usages, expand a neighbourhood)
without Codebase-Memory's *cost*. Every method returns a **derived, ranked,
compact** view — a function's signature plus its top callers/callees, an
ego-graph capped to a token budget — rather than raw source lines or a Cypher
result set. That is the whole point of doing this inside PixelMem: same
navigation ergonomics, a fraction of the tokens, and zero query-authoring turns.

The index is built purely from V5 ``Primitive`` quadruples, so it works the same
whether the primitives came straight from the extractor, were round-tripped
through the pixel cache, or were synthesised for another language.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from pixelmem.v5.core.types import Primitive
from pixelmem.v5.plugins.code_graph.extractor import (
    QUAL_SEP,
    REL_CALLS,
    REL_DECORATED_BY,
    REL_DEFINES,
    REL_HAS_METHOD,
    REL_INHERITS,
    REL_IMPORTS_MODULE,
    REL_USES,
)


def approx_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token)."""
    return (len(text) + 3) // 4


@dataclass
class _Def:
    qual: str
    file: str
    kind: str            # function | method | class
    line: int = 0
    sig: str = ""

    @property
    def simple(self) -> str:
        tail = self.qual.split(QUAL_SEP, 1)[-1]
        return tail.split(".")[-1]


@dataclass
class CodeGraphIndex:
    """Adjacency views over a code graph, built from primitives."""

    defs: dict[str, _Def] = field(default_factory=dict)          # qual -> _Def
    by_file: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    calls_out: dict[str, list[tuple[str, str, int]]] = field(
        default_factory=lambda: defaultdict(list))               # qual -> [(callee, cond, line)]
    calls_in: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    uses_in: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    methods: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    bases: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    subclasses: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    decorators: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    imports: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    # ------------------------------------------------------------------ build
    @classmethod
    def from_primitives(cls, primitives: Iterable[Primitive]) -> "CodeGraphIndex":
        idx = cls()
        for p in primitives:
            r = p.relation
            if r == REL_DEFINES:
                prov = p.provenance or {}
                d = _Def(qual=p.object, file=p.subject, kind=p.condition or "",
                         line=int(prov.get("line", 0)), sig=str(prov.get("sig", "")))
                idx.defs[p.object] = d
                idx.by_file[p.subject].append(p.object)
            elif r == REL_CALLS:
                prov = p.provenance or {}
                idx.calls_out[p.subject].append(
                    (p.object, p.condition, int(prov.get("line", 0))))
                idx.calls_in[p.object].append(p.subject)
            elif r == REL_USES:
                idx.uses_in[p.object].append(p.subject)
            elif r == REL_HAS_METHOD:
                idx.methods[p.subject].append(p.object)
            elif r == REL_INHERITS:
                idx.bases[p.subject].append(p.object)
                idx.subclasses[p.object].append(p.subject)
            elif r == REL_DECORATED_BY:
                idx.decorators[p.subject].append(p.object)
            elif r == REL_IMPORTS_MODULE:
                idx.imports[p.subject].append(p.object)
        return idx

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        return {
            "files": len(self.by_file),
            "definitions": len(self.defs),
            "functions": sum(1 for d in self.defs.values() if d.kind == "function"),
            "methods": sum(1 for d in self.defs.values() if d.kind == "method"),
            "classes": sum(1 for d in self.defs.values() if d.kind == "class"),
            "call_edges": sum(len(v) for v in self.calls_out.values()),
            "resolved_call_edges": sum(
                1 for v in self.calls_out.values() for (_, c, _) in v if c in ("resolved", "self")),
        }

    # --------------------------------------------------------------- 1. search
    def search_symbol(self, query: str, k: int = 8) -> list[dict]:
        """Rank definitions by name match against ``query`` (exact > prefix >
        substring > token-overlap). The narrowing step RepoGraph/CodexGraph rely
        on a DB query for — here it is a ranked, in-memory pass."""
        q = query.strip().lower()
        q_tokens: set[str] = set()
        for w in re.findall(r"[a-z0-9]+", q):
            q_tokens.update(_split_ident(w))
        scored: list[tuple[float, _Def]] = []
        for d in self.defs.values():
            name = d.simple.lower()
            # The module/file name is a real semantic signal (grep/RepoGraph use
            # it too): "backoff delay" should find compute_delay in backoff.py.
            file_base = d.file.split("/")[-1].rsplit(".", 1)[0].lower()
            tokens = set(_split_ident(name)) | set(_split_ident(file_base))
            score = 0.0
            if name == q:
                score = 100.0
            elif len(q) >= 3 and (name.startswith(q) or q.startswith(name)):
                score = 70.0
            elif len(q) >= 3 and q in name:
                score = 50.0
            else:
                ov = 0
                for qt in q_tokens:
                    if qt in tokens or any(
                        len(qt) >= 4 and len(nt) >= 4 and
                        (qt.startswith(nt) or nt.startswith(qt))
                        for nt in tokens
                    ):
                        ov += 1
                if ov:
                    score = 20.0 * ov
            if score:
                # Tie-breaks: name-token matches over file-only; closeness in length.
                if any(qt in set(_split_ident(name)) for qt in q_tokens):
                    score += 2.0
                score -= 0.01 * abs(len(name) - len(q))
                scored.append((score, d))
        scored.sort(key=lambda t: (-t[0], t[1].qual))
        return [self._def_card(d) for _, d in scored[:k]]

    # ------------------------------------------------------------------ 2. ego
    def ego_graph(self, symbol: str, hops: int = 1, budget_tokens: int = 220) -> dict:
        """A compact neighbourhood around ``symbol``: its signature, callers,
        callees, base/derived classes and notable uses — ranked and capped to a
        token budget. This is RepoGraph's ego-graph, emitted as a derived
        summary instead of raw lines."""
        qual = self._canonical(symbol)
        if qual is None:
            return {"symbol": symbol, "status": "not_found",
                    "did_you_mean": [c["qual"] for c in self.search_symbol(symbol, 5)]}

        d = self.defs[qual]
        callers = _dedup_keep_order(self.calls_in.get(qual, []))
        callees = [c for (c, cond, _) in self.calls_out.get(qual, [])
                   if cond in ("resolved", "self")]
        callees = _dedup_keep_order(callees)
        users = _dedup_keep_order(self.uses_in.get(qual, []))

        view: dict = {
            "symbol": qual,
            "kind": d.kind,
            "file": d.file,
            "signature": d.sig,
            "decorators": self.decorators.get(qual, []),
        }
        if d.kind == "class":
            view["inherits"] = self.bases.get(qual, [])
            view["subclasses"] = self.subclasses.get(qual, [])
            view["methods"] = [self._short(m) for m in self.methods.get(qual, [])]
        view["called_by"] = [self._short(c) for c in callers]
        view["calls"] = [self._short(c) for c in callees]
        if users:
            view["used_by"] = [self._short(u) for u in users]

        if hops >= 2:
            two = []
            for c in callees[:4]:
                grand = [self._short(g) for (g, cond, _) in self.calls_out.get(c, [])
                         if cond in ("resolved", "self")][:3]
                if grand:
                    two.append({self._short(c): grand})
            if two:
                view["calls_2hop"] = two

        return _fit_budget(view, budget_tokens)

    # --------------------------------------------------------- 3. trace calls
    def trace_call_chain(
        self, symbol: str, direction: str = "out", max_depth: int = 3, max_paths: int = 6,
    ) -> dict:
        """BFS over call edges from ``symbol``. ``direction='out'`` = what it
        eventually calls; ``'in'`` = what eventually reaches it."""
        qual = self._canonical(symbol)
        if qual is None:
            return {"symbol": symbol, "status": "not_found"}
        adj = self._out_adj if direction == "out" else self._in_adj

        paths: list[list[str]] = []
        stack: list[tuple[str, list[str]]] = [(qual, [qual])]
        seen_edges: set[tuple[str, str]] = set()
        while stack and len(paths) < max_paths:
            node, path = stack.pop()
            nxt = [n for n in adj(node) if n not in path]
            if not nxt or len(path) > max_depth:
                if len(path) > 1:
                    paths.append(path)
                continue
            extended = False
            for n in nxt:
                if (node, n) in seen_edges:
                    continue
                seen_edges.add((node, n))
                stack.append((n, path + [n]))
                extended = True
            if not extended and len(path) > 1:
                paths.append(path)

        return {
            "symbol": qual,
            "direction": direction,
            "chains": [" -> ".join(self._short(n) for n in p) for p in paths[:max_paths]],
        }

    # ----------------------------------------------------- 4. dependencies
    def find_dependencies(self, file: str, reverse: bool = False) -> dict:
        """File-level dependencies, *derived from call/use edges* (a richer
        signal than imports alone). ``reverse=True`` returns dependents."""
        target = self._match_file(file)
        if target is None:
            return {"file": file, "status": "not_found",
                    "known_files": sorted(self.by_file)[:20]}
        forward: dict[str, set[str]] = defaultdict(set)   # file -> {reasons}
        for owner in self.by_file.get(target, []):
            for (callee, cond, _) in self.calls_out.get(owner, []):
                if cond in ("resolved", "self") and callee in self.defs:
                    cf = self.defs[callee].file
                    if cf != target:
                        forward[cf].add(self._short(callee))
        if not reverse:
            return {"file": target, "depends_on": _reasons(forward),
                    "imports": _dedup_keep_order(self.imports.get(target, []))[:12]}
        # reverse: who depends on `target`
        back: dict[str, set[str]] = defaultdict(set)
        for owner, edges in self.calls_out.items():
            of = self.defs[owner].file if owner in self.defs else None
            if of is None or of == target:
                continue
            for (callee, cond, _) in edges:
                if cond in ("resolved", "self") and callee in self.defs \
                        and self.defs[callee].file == target:
                    back[of].add(self._short(callee))
        return {"file": target, "depended_on_by": _reasons(back)}

    # --------------------------------------------------------- 5. locate usage
    def locate_usage(self, symbol: str, k: int = 20) -> dict:
        """Every call/use site of ``symbol`` with owner + file:line."""
        qual = self._canonical(symbol)
        if qual is None:
            return {"symbol": symbol, "status": "not_found",
                    "did_you_mean": [c["qual"] for c in self.search_symbol(symbol, 5)]}
        sites: list[dict] = []
        for caller in self.calls_in.get(qual, []):
            line = next((ln for (c, _, ln) in self.calls_out.get(caller, [])
                         if c == qual), 0)
            sites.append({"by": self._short(caller),
                          "file": self.defs[caller].file if caller in self.defs else "?",
                          "line": line, "via": "call"})
        for user in self.uses_in.get(qual, []):
            sites.append({"by": self._short(user),
                          "file": self.defs[user].file if user in self.defs else "?",
                          "via": "use"})
        return {"symbol": qual, "n_sites": len(sites), "sites": sites[:k]}

    # ------------------------------------------------------------- internals
    def _out_adj(self, node: str) -> list[str]:
        return [c for (c, cond, _) in self.calls_out.get(node, [])
                if cond in ("resolved", "self") and c in self.defs]

    def _in_adj(self, node: str) -> list[str]:
        return [c for c in self.calls_in.get(node, []) if c in self.defs]

    def _def_card(self, d: _Def) -> dict:
        return {"qual": d.qual, "name": d.simple, "kind": d.kind,
                "file": d.file, "line": d.line, "signature": d.sig}

    def _canonical(self, symbol: str) -> Optional[str]:
        """Map a user-supplied symbol (qual, ``file::name``, or bare name) to a
        known qualname."""
        if symbol in self.defs:
            return symbol
        # bare name -> unique qual
        cands = [q for q in self.defs if q.split(QUAL_SEP, 1)[-1].split(".")[-1] == symbol]
        if len(cands) == 1:
            return cands[0]
        # suffix match (e.g. "Class.method")
        suffix = [q for q in self.defs if q.endswith(symbol)]
        if len(suffix) == 1:
            return suffix[0]
        return None

    def _match_file(self, file: str) -> Optional[str]:
        if file in self.by_file:
            return file
        cands = [f for f in self.by_file if f.endswith(file) or f.split("/")[-1] == file]
        return cands[0] if len(cands) == 1 else (cands[0] if cands else None)

    def _short(self, qual: str) -> str:
        """Render a qualname compactly: drop the file prefix, keep Class.method."""
        if QUAL_SEP in qual:
            return qual.split(QUAL_SEP, 1)[1]
        return qual


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _split_ident(name: str) -> list[str]:
    out: list[str] = []
    cur = ""
    for ch in name.replace(".", "_"):
        if ch == "_":
            if cur:
                out.append(cur.lower())
                cur = ""
        elif ch.isupper() and cur and not cur[-1].isupper():
            out.append(cur.lower())
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur.lower())
    return out


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _reasons(d: dict[str, set[str]]) -> list[dict]:
    rows = []
    for f, why in sorted(d.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rows.append({"file": f, "via": sorted(why)[:6], "n": len(why)})
    return rows


def _fit_budget(view: dict, budget_tokens: int) -> dict:
    """Trim list fields of an ego view until it fits the token budget."""
    import json

    trim_order = ["used_by", "calls_2hop", "subclasses", "calls", "called_by", "methods"]
    while approx_tokens(json.dumps(view)) > budget_tokens:
        trimmed = False
        for key in trim_order:
            v = view.get(key)
            if isinstance(v, list) and len(v) > 3:
                view[key] = v[:3]
                view[f"{key}_truncated"] = True
                trimmed = True
                break
        if not trimmed:
            break
    return view


def render_ego(view: dict) -> str:
    """Human/LLM-readable one-block rendering of an ego view (for prompts)."""
    if view.get("status") == "not_found":
        dym = view.get("did_you_mean", [])
        return f"{view['symbol']}: not found" + (f" (did you mean: {', '.join(dym)})" if dym else "")
    lines = [f"{view['kind']} {view['symbol']}{view.get('signature','')}  [{view['file']}]"]
    if view.get("inherits"):
        lines.append(f"  inherits: {', '.join(view['inherits'])}")
    if view.get("methods"):
        lines.append(f"  methods: {', '.join(view['methods'])}")
    if view.get("called_by"):
        lines.append(f"  called by: {', '.join(view['called_by'])}")
    if view.get("calls"):
        lines.append(f"  calls: {', '.join(view['calls'])}")
    if view.get("used_by"):
        lines.append(f"  used by: {', '.join(view['used_by'])}")
    return "\n".join(lines)


__all__ = ["CodeGraphIndex", "approx_tokens", "render_ego"]
