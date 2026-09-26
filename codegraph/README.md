# code_graph — Codebase-Memory capability on the PixelMem V5 substrate

A Codebase-Memory-style **code knowledge graph** built entirely on PixelMem V5's
`(s, r, o, c)` Primitive substrate, derivation engine, `V5Pipeline` host, and an
MCP server — **without giving up the token-economy edge**.

Recent code-graph systems (RepoGraph, CodexGraph, the Tree-Sitter + MCP
*Codebase-Memory* paper) optimize the opposite axis: they inject richer context
(line-level ego-graphs, Neo4j symbol DBs, MCP call-chain traces) and accept the
token cost. This plugin **derives** a compact, task-shaped view instead.

## What it does

- **`extractor.py`** — `CodeGraphExtractor`: Python `ast` → a rich edge set as
  `Primitive` quadruples (`defines`, `has_method`, `inherits`, `calls`, `uses`,
  imports, `annotates`, `decorated_by`). 3-pass (register → structure → edges).
  Conservative call resolution: bare-name calls resolve iff repo-unique;
  `self`/`cls` resolve over the class MRO; arbitrary `x.m()` is left unresolved
  (under-link rather than mis-link).
- **`index.py`** — `CodeGraphIndex`: a navigation surface — `search_symbol`,
  `ego_graph`, `trace_call_chain`, `find_dependencies`, `locate_usage` — each
  returning a **token-budgeted** compact view.
- **`plugins.py`** — `CodeGraphDerivationEngine` lifts `calls` → `file_depends_on`
  + base-first order; `CodeNavPrompt` (modes: order / locate / qa);
  `build_code_graph_plugins()`.
- **`code_graph_mcp_server.py`** — MCP tools mirroring Codebase-Memory
  (`code_search` / `ego` / `trace` / `dependencies` / `usage`). Needs `fastmcp`.

## Run it (offline, AST only — no LLM/GPU)

```bash
cd codegraph
PYTHONPATH=. python experiments/exp61_code_graph_demo.py        # end-to-end demo + token economy
PYTHONPATH=. python experiments/exp62_vs_codegraph_case.py      # honest case study vs RepoGraph / CodexGraph

# MCP server (pip install fastmcp first):
PYTHONPATH=. python code_graph_mcp_server.py
```

This folder is **self-contained**: it depends only on `pixelmem/v5/core` (the
Primitive substrate, derivation, pipeline) plus the plugin itself.

## Economy (from `exp61`, scanning this folder's `pixelmem/v5`)

| Representation | Tokens to answer one localization query |
|---|---|
| raw source (13 files) | ~21,982 |
| full primitive-graph dump | ~20,976 |
| **one derived ego view (ours)** | **~63** |

→ **333–349×** smaller on this corpus; **656×/782×** on the full 30-file V5 tree.

## Status & next step

**Offline-verified** only: synthetic correctness suite (cross-file calls,
inherited `self.method` via MRO, call→file-dep derivation, base-first order, all
5 nav tools) passes, and `exp61`/`exp62` run end-to-end. **No real-LLM eval has
been run yet.** The next step is a real-LLM localization eval (RepoQA /
SWE-bench-Lite) wiring `build_code_graph_plugins(mode='locate')` to the OpenAI
caller used in `exp56`/`exp58`, reporting **accuracy and tokens** against the
full-text (`exp57`) and primitive-dump (`exp59`/`exp60`) baselines.

**Known limitation / next lever:** receiver-type + import-alias resolution (what
CodexGraph's DB buys) would recover the cross-object `x.m()` call edges we
currently leave unresolved.

Full write-up: [`docs/v5_code_graph.md`](docs/v5_code_graph.md).
