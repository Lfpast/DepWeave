# V5 Code-Graph Plugin — Codebase-Memory capability, PixelMem economy

`pixelmem/v5/plugins/code_graph/` — a Codebase-Memory-style code knowledge graph
built **entirely on the existing V5 architecture**: the `(s, r, o, c)` Primitive
substrate, the derivation engine, the `V5Pipeline` host, and an MCP server. It is
the answer to "make PixelMem competitive with the recent code-graph projects
(RepoGraph / CodexGraph / the Tree-Sitter+MCP *Codebase-Memory* paper) without
giving up our token-economy edge."

## Why this exists

Those projects optimise the *opposite axis* from PixelMem: they inject more,
richer context (line-level ego-graphs, Neo4j symbol DBs, MCP call-chain traces)
and accept the token cost — CodexGraph and Codebase-Memory both list token
efficiency as a stated limitation. PixelMem's moat is the inverse: *derive +
rank + compress*. The two ablations (`exp59`, `exp60`) prove the lift is in the
derivation layer, not the raw graph (dumping primitives ≈ dumping text). So we
give ourselves their **interface** while keeping our **economy**.

## What it builds (the edge vocabulary)

Pure Python `ast` (zero new deps; a Tree-Sitter backend can swap in behind the
same `Extractor` protocol for other languages). Every edge is a `Primitive`:

| Relation | Example | Notes |
|---|---|---|
| `defines` | `(mod.py, defines, mod.py::Foo.bar)` cond=`method` | file → every def |
| `has_method` | `(mod.py::Foo, has_method, mod.py::Foo.bar)` | class structure |
| `inherits` | `(mod.py::Foo, inherits, base.py::Base)` cond=`resolved` | class hierarchy |
| `calls` | `(mod.py::Foo.bar, calls, util.py::helper)` cond=`resolved\|self\|unresolved` | behavioural |
| `uses` | `(mod.py::Foo.bar, uses, util.py::CONST)` | name references to known defs |
| `imports_module` / `imports_symbol` | `(mod.py, imports_symbol, helper)` | imports |
| `annotates` | `(mod.py::bar, annotates, Path)` cond=`param\|return` | type refs |
| `decorated_by` | `(mod.py::bar, decorated_by, staticmethod)` | decorators |

**Resolution is conservative on purpose.** Bare-name calls (`foo()`) resolve to a
def iff the name is repo-unique; `self.m()`/`cls.m()` resolve against the class
**MRO** (own + base classes); arbitrary `x.m()` is left *unresolved* — we don't
track `x`'s type, so resolving `"s".split()` to a user `split` method would be a
false edge. We under-link rather than mis-link. Receiver-type / import-alias
resolution (what CodexGraph's DB buys) is the named future lever.

## The derivation (this is the moat, not the graph)

`CodeGraphDerivationEngine` lifts thousands of `calls` edges into a small
`file_depends_on` graph + a base-first order — the same trick V4 uses on imports,
generalised to *call*-based deps (a strictly richer signal: catches wiring
imports miss). Declarative equivalent: `CALL_FILE_DEP_RULE`. Self-loops dropped.

## The navigation surface (`CodeGraphIndex`) — Codebase-Memory parity

Every method returns a **derived, token-budgeted, compact** view, not source:

| Method / MCP tool | Returns |
|---|---|
| `search_symbol` / `code_search` | ranked defs by fuzzy-token name match |
| `ego_graph` / `code_ego` | signature + callers + callees + inherits, capped to a token budget |
| `trace_call_chain` / `code_trace` | BFS call chains (in/out) |
| `find_dependencies` / `code_dependencies` | file deps derived from call edges (`reverse=` dependents) |
| `locate_usage` / `code_usage` | every call/use site with file:line |

MCP server: `code_graph_mcp_server.py` (`PIXELMEM_CODE_ROOT=… python code_graph_mcp_server.py`; needs `fastmcp`).

## Wiring it into the pipeline

```python
from pixelmem.v5 import V5Pipeline, TaskSpec
from pixelmem.v5.plugins.code_graph import build_code_graph_plugins

plugins = build_code_graph_plugins(mode="locate")   # order | locate | qa
pipe = V5Pipeline(plugins, task=TaskSpec(..., options={"mode": "locate"}), llm=my_llm)
name, stats = pipe.run({"question": "the fn that retries on 429"}, documents=repo)
```

## Measured economy (`exp61`, 30 V5 files, offline)

| To answer a localisation query | ~tokens |
|---|---|
| dump raw source of 30 files | ~53,966 |
| dump the full primitive graph | ~45,275 |
| **one derived ego view (our path)** | **~69** |

→ ego view is **782× smaller than source**, **656× smaller than a graph dump**.
Full `V5Pipeline` run (extract→derive→prompt→parse) builds a **448-token** locate
prompt vs ~54K to dump the source. Graph over the 30 files: 246 defs, 780 call
edges (159 confidently resolved), 0 parse errors.

## Status / next

- **Done & verified offline:** extractor, index/navigation, derivation, V5
  PluginSet, MCP server, `exp61` demo. All 8 `tests/test_v5_core.py` pass
  (isolation intact — `code_graph` imports nothing from V4).
- **Next (real-LLM eval, via `jobs/`):** wire `build_code_graph_plugins(mode="locate")`
  with the OpenAI caller from `exp56`/`exp58` on RepoQA needles / SWE-bench-Lite
  localisation, reporting **accuracy AND tokens** against the full-text (`exp57`)
  and primitive-dump (`exp59`/`exp60`) baselines. The headline to chase: *match
  RepoGraph/CodexGraph localisation at N× fewer tokens, no query-authoring turns.*
