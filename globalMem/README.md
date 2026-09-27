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
  Conservative call resolution: bare calls need a local definition or explicit
  import; `self`/`cls` resolve over statically bound base classes; arbitrary `x.m()` stays
  unresolved. Distinct call sites retain their source positions.
- **`index.py`** — `CodeGraphIndex`: a navigation surface — `search_symbol`,
  `ego_graph`, `trace_call_chain`, `find_dependencies`, `locate_usage` — each
  returning a **token-budgeted** compact view.
- **`plugins.py`** — `CodeGraphDerivationEngine` lifts `calls` → `file_depends_on`
  + base-first order; `CodeNavPrompt` (modes: order / locate / qa);
  `build_code_graph_plugins()`.
- **`code_graph_mcp_server.py`** — MCP navigation tools and DepWeave's
  `code_index_documents` / `code_evidence` packet interface. Needs `fastmcp`.

## Run it (offline, AST only — no LLM/GPU)

```bash
cd /home/jackson/python/DepWeave/globalMem
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

The DepWeave integration test starts this MCP server and checks source-backed
calls, unresolved gaps, and snapshot changes. No new Qwen benchmark score is
claimed by that test. The root `experiments/exp56_*` and `exp58_*` scripts now
call this service through `depweave/runner.py`.

**Known limitation:** arbitrary receiver calls such as `x.m()` remain unresolved
without type evidence; dynamic imports and wildcard re-exports can also leave
gaps. Those gaps are returned to the local layer.

Full write-up: [`docs/v5_code_graph.md`](docs/v5_code_graph.md).
