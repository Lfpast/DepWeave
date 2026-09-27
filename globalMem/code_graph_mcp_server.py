"""Code-Graph MCP Server — Codebase-Memory's tool surface, PixelMem's economy.

Exposes a navigable code knowledge graph to an LLM/agent over MCP, the same way
the recent Tree-Sitter *Codebase-Memory* paper does — search a symbol, trace a
call chain, find dependencies, locate usages, expand a neighbourhood. The
difference: every tool here returns a **derived, token-budgeted** view built from
PixelMem ``Primitive`` quadruples (see ``pixelmem/v5/plugins/code_graph/``), not
raw source lines or a Cypher result set. Same ergonomics, a fraction of the
context, and no query-authoring turns.

Run standalone (requires ``fastmcp``)::

    PIXELMEM_CODE_ROOT=/path/to/repo python code_graph_mcp_server.py

Tools:
  - code_stats         : graph summary (files, defs, call edges)
  - code_search        : rank definitions by name match
  - code_ego           : compact neighbourhood (callers/callees/inherits) of a symbol
  - code_trace         : BFS call chain in/out of a symbol
  - code_dependencies  : file-level deps derived from call edges (reverse=dependents)
  - code_usage         : every call/use site of a symbol
  - code_reindex       : rebuild the graph from a directory
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastmcp import FastMCP

from pixelmem.v5.plugins.code_graph import CodeGraphExtractor, CodeGraphIndex, render_ego
from depweave_evidence import EvidenceStore

_DEFAULT_ROOT = os.environ.get("PIXELMEM_CODE_ROOT", ".")
_MAX_FILES = int(os.environ.get("PIXELMEM_CODE_MAX_FILES", "4000"))

_index: CodeGraphIndex | None = None
_root: str = _DEFAULT_ROOT
_evidence = EvidenceStore(max_files=_MAX_FILES)


def _build_index(root: str) -> tuple[CodeGraphIndex, dict]:
    base = Path(root).resolve()
    docs: dict[str, str] = {}
    for f in sorted(base.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        try:
            docs[str(f.relative_to(base))] = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if len(docs) >= _MAX_FILES:
            break
    ex = CodeGraphExtractor()
    prims = ex.extract(docs)
    idx = CodeGraphIndex.from_primitives(prims)
    meta = {"root": str(base), "files": len(docs), "primitives": len(prims),
            "parse_errors": ex.n_parse_errors}
    return idx, meta


def _get_index() -> CodeGraphIndex:
    global _index
    if _index is None:
        _index, meta = _build_index(_root)
        print(f"[code_graph] indexed {meta}", file=sys.stderr)
    return _index


mcp = FastMCP(
    "code_graph",
    instructions=(
        "A navigable code knowledge graph. Prefer this over reading whole files: "
        "1) code_search to find a symbol, 2) code_ego for its neighbourhood, "
        "3) code_trace / code_dependencies / code_usage to follow structure. "
        "Every result is a compact derived view, not raw source."
    ),
)


@mcp.tool()
def code_stats() -> str:
    """Summary of the indexed code graph (files, definitions, call edges)."""
    return json.dumps(_get_index().stats())


@mcp.tool()
def code_search(query: str, k: int = 8) -> str:
    """Find definitions whose name best matches ``query`` (ranked). Returns
    name, kind, file, line, signature for the top-k — the narrowing step before
    you expand a neighbourhood."""
    return json.dumps({"query": query, "results": _get_index().search_symbol(query, k)})


@mcp.tool()
def code_ego(symbol: str, hops: int = 1, budget_tokens: int = 220) -> str:
    """Compact neighbourhood of ``symbol``: signature, callers, callees,
    base/derived classes, notable uses — ranked and capped to a token budget.
    ``symbol`` may be a bare name, ``Class.method``, or a full qualname."""
    view = _get_index().ego_graph(symbol, hops=hops, budget_tokens=budget_tokens)
    return json.dumps({"view": view, "rendered": render_ego(view)})


@mcp.tool()
def code_trace(symbol: str, direction: str = "out", max_depth: int = 3) -> str:
    """Trace call chains from ``symbol``. ``direction='out'`` = what it
    transitively calls; ``'in'`` = what transitively reaches it."""
    return json.dumps(_get_index().trace_call_chain(symbol, direction, max_depth))


@mcp.tool()
def code_dependencies(file: str, reverse: bool = False) -> str:
    """File-level dependencies derived from call edges (richer than imports).
    ``reverse=True`` returns the files that depend on ``file``."""
    return json.dumps(_get_index().find_dependencies(file, reverse=reverse))


@mcp.tool()
def code_usage(symbol: str, k: int = 20) -> str:
    """Every call/use site of ``symbol`` with the owning symbol and file:line."""
    return json.dumps(_get_index().locate_usage(symbol, k))


@mcp.tool()
def code_reindex(root: str = "") -> str:
    """Rebuild the graph from ``root`` (defaults to the current root /
    $PIXELMEM_CODE_ROOT). Call after the codebase changes."""
    global _index, _root
    if root:
        _root = root
    _index, meta = _build_index(_root)
    return json.dumps({"status": "reindexed", **meta})


@mcp.tool()
def code_index_documents(repo_id: str, documents_json: str) -> str:
    """Index a named repository snapshot from path-to-source JSON documents.

    Returns a content-addressed snapshot ID and explicit indexing coverage.
    The client must pass this ID to code_evidence for every query.
    """
    documents = json.loads(documents_json)
    if not isinstance(documents, dict):
        raise ValueError("documents_json must be an object keyed by relative path")
    return json.dumps(_evidence.index_documents(repo_id, documents))


@mcp.tool()
def code_evidence(snapshot_id: str, query: str, seeds_json: str = "[]",
                  max_candidates: int = 16, max_edges: int = 80, hops: int = 1) -> str:
    """Return source-backed candidates, relation sites and unresolved gaps.

    Seeds are local candidates [{path, line, name}]. An edge always carries
    canonical endpoint IDs, relation type, source path and source line.
    """
    seeds = json.loads(seeds_json)
    if not isinstance(seeds, list):
        raise ValueError("seeds_json must be a list")
    return json.dumps(_evidence.packet(snapshot_id, query, seeds, max_candidates, max_edges, hops))


if __name__ == "__main__":
    mcp.run()
