"""Source-backed evidence packets built inside the globalMem process.

The local process never imports this module: its ``pixelmem`` package is a
different implementation. MCP carries plain JSON between the two processes.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from pixelmem.v5.plugins.code_graph import CodeGraphExtractor, CodeGraphIndex
from pixelmem.v5.plugins.code_graph.extractor import (
    REL_CALLS, REL_IMPORTS_MODULE, REL_IMPORTS_SYMBOL, REL_INHERITS,
)


def snapshot_id(repo_id: str, documents: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(repo_id.encode("utf-8"))
    for path, source in sorted(documents.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass
class _Snapshot:
    repo_id: str
    snapshot_id: str
    documents: dict[str, str]
    index: CodeGraphIndex
    primitives: list
    coverage: dict
    modules: dict[str, str]

    def entity_id(self, qual: str) -> str:
        definition = self.index.defs.get(qual)
        if definition:
            return f"{self.snapshot_id}|{definition.file}|{qual}|{definition.line}"
        return f"{self.snapshot_id}|{qual}|@file|0"

    def file_id(self, path: str) -> str:
        return f"{self.snapshot_id}|{path}|@file|0"

    def card(self, qual: str) -> dict:
        d = self.index.defs[qual]
        return {"id": self.entity_id(qual), "qual": qual, "path": d.file,
                "line": d.line, "kind": d.kind, "signature": d.sig,
                "name": d.simple}


class EvidenceStore:
    def __init__(self, max_files: int = 4000) -> None:
        self.max_files = max_files
        self.snapshots: dict[str, _Snapshot] = {}
        self.repo_latest: dict[str, str] = {}

    def index_documents(self, repo_id: str, documents: dict[str, str]) -> dict:
        if not repo_id:
            raise ValueError("repo_id is required")
        py_paths = sorted(path for path in documents if path.endswith(".py"))
        omitted = py_paths[self.max_files:]
        selected: dict[str, str] = {}
        unreadable: list[str] = []
        for path in py_paths[:self.max_files]:
            value = documents[path]
            if isinstance(value, str):
                selected[path] = value
            else:
                unreadable.append(path)
        sid = snapshot_id(repo_id, {path: str(source) for path, source in documents.items()})
        reused = sid in self.snapshots
        if not reused:
            extractor = CodeGraphExtractor()
            primitives = extractor.extract(selected)
            index = CodeGraphIndex.from_primitives(primitives)
            coverage = {
                "python_files": len(py_paths), "indexed_files": len(selected) - extractor.n_parse_errors,
                "excluded_non_python": len(documents) - len(py_paths),
                "omitted_files": omitted, "unreadable_files": unreadable,
                "parse_error_files": extractor.parse_error_paths,
                "duplicate_qualnames": extractor.duplicate_quals,
            }
            self.snapshots[sid] = _Snapshot(repo_id, sid, selected, index, primitives, coverage,
                                            _module_aliases(selected))
        self.repo_latest[repo_id] = sid
        snap = self.snapshots[sid]
        return {"snapshot_id": sid, "repo_id": repo_id, "coverage": snap.coverage,
                "stats": snap.index.stats(), "reused": reused}

    def packet(self, snapshot_id: str, query: str, seeds: list[dict] | None = None,
               max_candidates: int = 16, max_edges: int = 80, hops: int = 1,
               max_gaps: int = 80) -> dict:
        snap = self.snapshots[snapshot_id]
        idx = snap.index
        internal_heads = {part for path in snap.documents for part in path.split("/")[:-1]}
        internal_heads.update(path.rsplit("/", 1)[-1][:-3] for path in snap.documents
                              if path.endswith(".py"))
        chosen: list[str] = []
        unmatched_seeds: list[dict] = []
        seed_paths = {seed.get("path") for seed in (seeds or []) if seed.get("path")}
        for seed in seeds or []:
            path = seed.get("path")
            line = seed.get("line")
            name = seed.get("name")
            matches = [q for q in idx.by_file.get(path, [])
                       if idx.defs[q].line == line and idx.defs[q].simple == name]
            if len(matches) == 1 and matches[0] not in chosen:
                chosen.append(matches[0])
            elif line is not None and len(matches) != 1:
                unmatched_seeds.append(seed)
        for path in sorted({seed["path"] for seed in (seeds or []) if seed.get("path") and seed.get("line") is None}):
            for qual in idx.by_file.get(path, [])[:2]:
                if qual not in chosen:
                    chosen.append(qual)
        for hit in idx.search_symbol(query, max_candidates):
            if hit["qual"] not in chosen:
                chosen.append(hit["qual"])
            if len(chosen) >= max_candidates:
                break
        chosen = chosen[:max_candidates]
        selected = set(chosen)
        edges: list[dict] = []
        gaps: list[dict] = []
        emitted: set[tuple] = set()
        for depth in range(max(1, min(hops, 2))):
            newly_seen: set[str] = set()
            for p in snap.primitives:
                if p.relation not in (REL_CALLS, REL_INHERITS, REL_IMPORTS_MODULE, REL_IMPORTS_SYMBOL):
                    continue
                owner = p.subject
                owner_file = owner.split("::", 1)[0]
                if owner not in selected and not (depth == 0 and owner_file in seed_paths | {idx.defs[q].file for q in chosen}):
                    continue
                source = {"path": owner_file, "line": (p.provenance or {}).get("line", 0),
                          "column": (p.provenance or {}).get("column", 0)}
                target = p.object
                status = "resolved" if target in idx.defs and p.condition in ("resolved", "self") else "unresolved"
                if p.relation in (REL_IMPORTS_MODULE, REL_IMPORTS_SYMBOL):
                    module = p.object if p.relation == REL_IMPORTS_MODULE else p.condition
                    target_path = _module_path(snap.documents, owner_file, module, snap.modules)
                    if p.relation == REL_IMPORTS_SYMBOL and target_path:
                        direct = f"{target_path}::{p.object}"
                        target = direct if direct in idx.defs else _reexport(snap, target_path, p.object)
                        if target is None:
                            child_path = _module_path(snap.documents, owner_file, f"{module}.{p.object}", snap.modules)
                            target = child_path or p.object
                        status = "resolved" if target in idx.defs or target in snap.documents else "unresolved"
                    elif target_path:
                        target = target_path
                        status = "resolved"
                    else:
                        status = "unresolved"
                if status == "unresolved":
                    if p.relation in (REL_IMPORTS_MODULE, REL_IMPORTS_SYMBOL) and not _internal_import(internal_heads, module):
                        continue
                    gaps.append({"from_id": snap.entity_id(owner) if owner in idx.defs else snap.file_id(owner_file),
                                 "relation": p.relation, "target_text": p.object,
                                 "source": source, "reason": "target_not_proven"})
                    continue
                target_id = snap.entity_id(target) if target in idx.defs else snap.file_id(target)
                source_id = snap.entity_id(owner) if owner in idx.defs else snap.file_id(owner_file)
                key = (source_id, p.relation, target_id, source["line"], source["column"])
                if key in emitted:
                    continue
                emitted.add(key)
                edges.append({"from_id": source_id, "to_id": target_id,
                              "relation": p.relation, "source": source, "status": "resolved"})
                if target in idx.defs:
                    newly_seen.add(target)
            selected.update(newly_seen)
        candidate_ids = {snap.entity_id(q) for q in chosen}
        edges.sort(key=lambda e: (e["from_id"] not in candidate_ids,
                                  e["relation"] not in (REL_CALLS, REL_INHERITS),
                                  e["source"]["path"], e["source"]["line"]))
        gaps.sort(key=lambda g: (g["from_id"] not in candidate_ids,
                                 g["relation"] != REL_CALLS,
                                 g["source"]["path"], g["source"]["line"]))
        omitted_edges = max(0, len(edges) - max_edges)
        omitted_gaps = max(0, len(gaps) - max_gaps)
        edges = edges[:max_edges]
        gaps = gaps[:max_gaps]
        if omitted_edges:
            gaps.append({"reason": "edge_limit", "omitted_count": omitted_edges})
        if omitted_gaps:
            gaps.append({"reason": "gap_limit", "omitted_count": omitted_gaps})
        for path in snap.coverage["parse_error_files"] + snap.coverage["omitted_files"] + snap.coverage["unreadable_files"]:
            gaps.append({"reason": "not_indexed", "path": path})
        for seed in unmatched_seeds:
            gaps.append({"reason": "seed_not_in_graph", "path": seed.get("path"),
                         "line": seed.get("line"), "name": seed.get("name")})
        for qual in snap.coverage["duplicate_qualnames"]:
            gaps.append({"reason": "duplicate_definition", "path": qual.split("::", 1)[0],
                         "qual": qual})
        return {"snapshot_id": snapshot_id, "coverage": snap.coverage,
                "candidates": [snap.card(q) for q in chosen], "edges": edges,
                "gaps": gaps}


def _module_path(documents: dict[str, str], owner_file: str, module: str,
                 aliases: dict[str, str] | None = None) -> str | None:
    if module.startswith("."):
        level = len(module) - len(module.lstrip("."))
        parent = owner_file.split("/")[:-1]
        parent = parent[:max(0, len(parent) - level + 1)]
        name = "/".join([*parent, module.lstrip(".").replace(".", "/")]).strip("/")
    else:
        name = module.replace(".", "/")
    for path in (f"{name}.py", f"{name}/__init__.py"):
        if path in documents:
            return path
    if aliases is not None:
        return aliases.get(name.replace("/", "."))
    suffixes = (f"/{name}.py", f"/{name}/__init__.py")
    matches = [path for path in documents if path.endswith(suffixes)]
    if len(matches) == 1:
        return matches[0]
    return None


def _internal_import(heads: set[str], module: str) -> bool:
    if module.startswith("."):
        return True
    head = module.split(".")[0]
    return head in heads


def _module_aliases(documents: dict[str, str]) -> dict[str, str]:
    found: dict[str, set[str]] = {}
    for path in documents:
        if not path.endswith(".py"):
            continue
        name = path[:-3].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[:-9]
        parts = name.split(".")
        for i in range(len(parts)):
            found.setdefault(".".join(parts[i:]), set()).add(path)
    return {name: next(iter(paths)) for name, paths in found.items() if len(paths) == 1}


def _reexport(snap: _Snapshot, path: str, name: str, seen: set[str] | None = None) -> str | None:
    seen = seen or set()
    if path in seen:
        return None
    seen.add(path)
    try:
        tree = ast.parse(snap.documents[path])
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if (alias.asname or alias.name) != name:
                continue
            module = "." * node.level + (node.module or "")
            imported_path = _module_path(snap.documents, path, module, snap.modules)
            if imported_path is None:
                continue
            direct = f"{imported_path}::{alias.name}"
            if direct in snap.index.defs:
                return direct
            return _reexport(snap, imported_path, alias.name, seen)
    return None
