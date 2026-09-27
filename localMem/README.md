# localMem / CardMem

localMem 提供任务证据抽取、缓存与依赖推导。DepWeave 从根目录的 `depweave/`
调用这些组件；仓库图索引由 `globalMem/` 的 MCP 子进程维护。

## Layout

| Path | Purpose |
| --- | --- |
| `core/` | Pipeline, types, plugin contracts, cache, task cards, harness, and optional synthesis (`synth/`) |
| `pixelmem/` | Pixel storage and Python dependency primitives in one package |
| `../benchmarks/dependeval/` | DependEval parser, Python dependency plugins, and ordering interface |
| `../benchmarks/repoqa.py` | RepoQA function extractor and candidate card |
| `../experiments/` | Full DepWeave benchmark drivers and historical local ablations |
| `results/gpt-4o-mini/` | Historical GPT-4o-mini output, preserved locally and ignored by Git |
| `results/qwen3-4b/` | New local Qwen3-4B output, ignored by Git |
| `docs/` | Method and experiment history; older notes retain their original version names |

The generic pipeline imports neither PixelMem nor benchmark code. A benchmark
chooses its plugins and may use `PixelMemCache` to reuse extracted primitives.

## Recorded results

The following numbers are historical GPT-4o-mini runs. Accuracy is strict ordering
or function-name match under the experiment scripts' scoring rules. No Qwen3-4B
benchmark result has been generated yet.

| Benchmark | Quill | Comparison |
| --- | --- | --- |
| DependEval Task 2, 166 Python items | 141/166 (84.9%), 322 mean input tokens | Primitive dump: 72/166, 1,577 tokens; capped full text: 69/166, 2,722 tokens |
| RepoQA Python, 100 needles | 43/100 (43%), 676 mean input tokens | Primitive dump: 40/100, 7,229 tokens; capped full text: 5/100, 2,773 tokens |

RepoQA's cache-on and cache-off runs both scored 43/100. The recorded mean
per-query wall time was 5.22 s with cache and 17.18 s without it. See
`results/gpt-4o-mini/*summary.json` for the exact configurations and denominators.

## Run

从仓库根目录运行完整评测：

```bash
cd /home/jackson/python/DepWeave
python -m pip install -r requirements.txt
export QWEN3_4B_PATH=/path/to/local/Qwen3-4B
python -m experiments.exp56_repoqa_python_full
python -m experiments.exp58_v5_depeval_full
```

The model path must point to an already downloaded local checkpoint. It can
also be set with `QWEN3_4B_PATH`. Inference uses Transformers offline mode
(`local_files_only=True`), so no model is downloaded by these commands.
The full experiments also need the datasets under root `data/`. Run all
comparisons again with the same Qwen checkpoint before
interpreting differences between methods; historical GPT numbers are not
directly comparable with new Qwen numbers.
