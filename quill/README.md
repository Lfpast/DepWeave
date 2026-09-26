# Quill / CardMem

Quill extracts task facts, derives and ranks the useful ones, and gives the LLM
a compact card. The `quill` package is the task-independent pipeline. The
`pixelmem` package is one unified storage and dependency substrate; it has no
versioned subpackages.

## Layout

| Path | Purpose |
| --- | --- |
| `quill/` | Pipeline, types, plugin contracts, cache, task cards, harness, and optional synthesis (`synth/`) |
| `pixelmem/` | Pixel storage and Python dependency primitives in one package |
| `benchmarks/dependeval/` | DependEval parser, Python dependency plugins, and its standalone ordering interface |
| `benchmarks/repoqa.py` | RepoQA function extractor and candidate card |
| `experiments/` | Benchmark drivers and smoke runs |
| `results/` | Recorded outputs from earlier runs; restructuring does not regenerate them |
| `docs/` | Method and experiment history; older notes retain their original version names |

The generic pipeline imports neither PixelMem nor benchmark code. A benchmark
chooses its plugins and may use `PixelMemCache` to reuse extracted primitives.

## Recorded results

All primary runs use `gpt-4o-mini`. Accuracy is strict ordering or function-name
match under the experiment scripts' scoring rules.

| Benchmark | Quill | Comparison |
| --- | --- | --- |
| DependEval Task 2, 166 Python items | 141/166 (84.9%), 322 mean input tokens | Primitive dump: 72/166, 1,577 tokens; capped full text: 69/166, 2,722 tokens |
| RepoQA Python, 100 needles | 43/100 (43%), 676 mean input tokens | Primitive dump: 40/100, 7,229 tokens; capped full text: 5/100, 2,773 tokens |

RepoQA's cache-on and cache-off runs both scored 43/100. The recorded mean
per-query wall time was 5.22 s with cache and 17.18 s without it. See
`results/*summary.json` for the exact configurations and denominators.

## Run

```bash
cd quill
pip install -r requirements.txt
PYTHONPATH=. python -m unittest discover -s tests -v
OPENAI_API_KEY=... PYTHONPATH=. python experiments/exp58_v5_depeval_full.py
OPENAI_API_KEY=... PYTHONPATH=. python experiments/exp56_repoqa_python_full.py
```

The full experiments also need the datasets at the paths declared in their
drivers. The committed result files are historical records, not a claim that
the full benchmarks were rerun after this restructure.
