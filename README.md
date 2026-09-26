# code_graph

Two components from the **PixelMem V5** line, each in its own folder.

PixelMem V5 is a plugin host built on an `(s, r, o, c)` *Primitive* (quadruple)
substrate: a pipeline of **extract → resolve → derive → rank → compact prompt**.
Its thesis is the *inverse* of most memory/graph systems — instead of injecting
more, richer context, it **derives** a compact, task-shaped view and wins on
**accuracy-per-token**.

| Folder | What it is | Runs standalone? |
|--------|------------|------------------|
| [`quill/`](quill/) | **CardMem / "Quill"** — the V5 compact-card method (package `quill`), evaluated on DependEval and RepoQA. | **Yes** — the `quill` package plus the vendored `pixelmem` V2–V4 substrate it builds on. `pip install -r requirements.txt` then `PYTHONPATH=. python experiments/exp58_v5_depeval_full.py` (an eval run also needs `OPENAI_API_KEY` + the datasets). |
| [`codegraph/`](codegraph/) | **code_graph plugin** — a Codebase-Memory-style symbol graph (calls / inherits / uses / imports …) on the same substrate, with token-budgeted navigation and an MCP server. | **Yes** — depends only on `pixelmem/v5/core`; `PYTHONPATH=. python experiments/exp61_code_graph_demo.py` runs offline. |

## Headline results

**CardMem (Quill)** — DependEval Task 2, gpt-4o-mini, 166 items:
**84.9 % at 322 input tokens**, beating the V4 baseline (81.3 % / 389) and far
above a primitive-dump (43.4 % / 1577) or full-text (41.6 % / 2722). The +41 pp
over a raw primitive dump comes from the **derivation + ranking** layer — not
extraction.

**code_graph** — one localization query answered by a derived **ego view of ~63
tokens** vs ~22 K to dump source (≈**333–782× smaller**, corpus-dependent), with
no measurable accuracy loss in the offline pipeline.

See each folder's `README.md` for full tables, provenance, and how to run.
