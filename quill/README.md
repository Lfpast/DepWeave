# Quill / CardMem — the PixelMem V5 compact-card method

**Quill** (a.k.a. **CardMem**) extracts task primitives, **derives + ranks** the
ones that matter, and emits a compact, ordered "card" as the LLM prompt. It wins
on **accuracy-per-token** against both a raw primitive dump and full text.

The method package is imported as **`quill`**. It is built on the underlying
PixelMem V2–V4 substrate (AST primitive extraction, shard store, dependency
graph), which is vendored here under **`pixelmem/`** so this folder runs on its
own.

## Results

### DependEval Task 2 — gpt-4o-mini, 166 items

| Method | Accuracy | Mean input tokens | Source |
|---|---|---|---|
| **Quill (CardMem)** | **84.9 %** (141/166) | **322** | `results/exp58_v5_depeval_full_summary.json` |
| PyDepCard (V4) | 81.3 % | 389 | (prior baseline) |
| Primitive-dump ablation | 43.4 % | 1577 | `results/exp59_…_summary.json` |
| Full-text baseline | 41.6 % | 2722 | `results/exp57_…_summary.json` |

### RepoQA (Python) — gpt-4o-mini, 100 needles

| Method | Accuracy | Mean input tokens | Note |
|---|---|---|---|
| **Quill (RepoQACard)** | **43 %** | **676** | 91 % cache hit → **3.3× speedup** (`exp56`) |
| Primitive-dump | 40 % | 7229 | 70 % truncated (`exp60`) |

**Key insight:** dumping raw primitives ≈ dumping text. The **+41 pp** over a
primitive dump comes from the **derivation + ranking** layer — *not* from
extraction (or from pixels). That layer is the moat.

## Layout

```
quill/                # the Quill / CardMem method  (import quill)
  core/               # Primitive substrate, plugin protocols, pipeline, derivation
  cache/              # pixel-cache (the 3.3× RepoQA speedup)
  plugins/python_deps # the DependEval extractor (adapts the pixelmem V4 AST substrate)
  synth/              # LLM-driven plugin/tool synthesis
  harness.py, task_card.py
pixelmem/             # vendored V2–V4 substrate Quill builds on (import pixelmem.*)
experiments/          # exp56/58/59/60 drivers (+ exp13/22/37 helpers)
results/              # the recorded JSON outputs behind the tables above
docs/                 # method write-ups, findings, failure analyses
requirements.txt
```

## Run it

```bash
cd quill
pip install -r requirements.txt                 # numpy, Pillow, openai, anthropic, rank-bm25, PyYAML
export OPENAI_API_KEY=sk-...                     # needed for the LLM eval calls
PYTHONPATH=. python experiments/exp58_v5_depeval_full.py    # DependEval (the 84.9% run)
PYTHONPATH=. python experiments/exp56_repoqa_python_full.py # RepoQA
```

A full evaluation run additionally needs the **DependEval / RepoQA datasets**
(the drivers read them from the paths in `experiments/exp13_dependeval.py` /
`exp37_v5_repoqa.py`). The committed `results/*.json` are the recorded outputs.

> **Verified:** the `quill` package, the vendored `pixelmem.*` substrate, and all
> experiment drivers import end-to-end from this folder alone (no dependency on
> the original PixelMem repo), and `build_python_deps_plugins()` + `V5Pipeline`
> construct successfully.

Start with [`docs/v5_three_benchmark_story.md`](docs/v5_three_benchmark_story.md)
and [`docs/v5_findings.md`](docs/v5_findings.md).
