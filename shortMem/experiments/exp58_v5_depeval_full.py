"""Exp 58: CardMem (V5) on DependEval Task 2 (Python, 3-5 file items).

Runs the unified PixelMem dependency extractor through Quill's plugin host.
The earlier V4 standalone run used GPT-4o-mini; its recorded scores are
historical and are not compared numerically with this local Qwen run.

Configuration:
  - Plugin set: ``build_python_deps_plugins()`` (PythonDependencyExtractor +
    PythonDependencyEngine + PythonOrderingPrompt) — the same V4 components,
    routed through ``V5Pipeline``.
  - Cache: ``PixelMemCache`` at ``/tmp/v5_depeval_cache``. Each
    DependEval question is a unique 3-5 file slice, so we expect ~0%
    hit rate; the cache just rides along.
  - LLM: local Qwen3-4B checkpoint.

Per-question records are written incrementally to
``results/exp58_v5_depeval_full.json`` every 10 questions. The aggregate
summary (accuracy, by-file-count breakdown, mean
tokens, cache hit rate, wall time) goes to
``results/exp58_v5_depeval_full_summary.json``.
"""

from __future__ import annotations

import json
import argparse
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import TaskCard
from core.cache import CachedExtractor, PixelMemCache
from core.pipeline import V5Pipeline
from core.plugins import PluginSet
from core.types import Primitive
from benchmarks.dependeval.python_deps import (
    PythonDependencyEngine,
    PythonDependencyExtractor,
    PythonOrderingPrompt,
)

from benchmarks.dependeval.data import DATA_PATH, parse_dependeval_content
from benchmarks.local_model import LocalQwen3, MODEL_NAME, RESULTS_DIR, add_model_arguments


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_JSON = str(RESULTS_DIR / "exp58_v5_depeval_full.json")
SUMMARY_JSON = str(RESULTS_DIR / "exp58_v5_depeval_full_summary.json")
CACHE_DIR = "/tmp/v5_depeval_cache"


# ---------------------------------------------------------------------------
# Cached extractor retaining the context needed for dependency derivation
# ---------------------------------------------------------------------------


class _DependencyCachedExtractor(CachedExtractor):
    """Use cached primitives while rebuilding per-query namespace on a hit."""

    def __init__(self, inner: PythonDependencyExtractor, cache: PixelMemCache) -> None:
        super().__init__(inner, cache)
        self._inner = inner

    def extract(self, documents: dict[str, str], **kwargs) -> list[Primitive]:
        primitives = super().extract(documents, **kwargs)
        if self.last_hit:
            # The graph also needs the namespace and resolver built by extract().
            self._inner.extract(documents, **kwargs)
        return primitives

    @property
    def last_namespace(self):
        return self._inner.last_namespace

    @property
    def last_resolver(self):
        return self._inner.last_resolver


# ---------------------------------------------------------------------------
# TaskCard / plugin wiring
# ---------------------------------------------------------------------------


def _build_card() -> TaskCard:
    """Build a single empty CardMem TaskCard for the python-dep-ordering
    domain. Holdout examples are filled in per-question by the runner;
    we don't materialize all 166 in one card so each query stays
    independent (no shared pipeline state across threads).
    """
    return TaskCard.from_dict({
        "domain": "python_dependency_ordering",
        "description": (
            "Topologically order a small set of Python source files so "
            "that base files (imported by others) come first."
        ),
        "input_schema": {
            "kind": "file_set",
            "files": "list[str]",
            "file_contents": "dict[str, str]",
        },
        "query": {"kind": "ordering", "output": "list[str]"},
        "eval": {"metric": "exact_match", "threshold": 0.80},
        "few_shot": [],
        "holdout": [],
    })


def _build_plugins(cache: PixelMemCache) -> tuple[PluginSet, _DependencyCachedExtractor]:
    """V4-wrapping plugin set with a PixelMemCache-backed extractor."""
    inner = PythonDependencyExtractor(language="python")
    probe = _DependencyCachedExtractor(inner, cache)
    engine = PythonDependencyEngine(probe)  # type: ignore[arg-type]
    plugins = PluginSet(
        name="pydepcard_v4_adapter_cached",
        extractor=probe,
        resolver=None,
        derivation_rules=[],
        derivation_engine=engine,
        prompt_template=PythonOrderingPrompt(),
    )
    return plugins, probe


# ---------------------------------------------------------------------------
# Per-question runner — routes through V5Pipeline.run()
# ---------------------------------------------------------------------------


def process_q(qi: int, item: dict, cache: PixelMemCache, llm) -> dict:
    f_raw, fc, gt_raw = parse_dependeval_content(item)
    files = [f.strip("'\"") for f in f_raw]
    gt_bn = [f.strip("'\"").split("/")[-1] for f in gt_raw]

    # documents dict keyed by file path; matches PythonDependencyExtractor's expectation.
    documents = {f: fc.get(f, "") for f in files}
    query_input = {"files": files, "file_contents": documents}

    plugins, probe = _build_plugins(cache)
    card = _build_card()
    card.spec.options["_last_doc_keys"] = list(documents.keys())
    pipe = V5Pipeline(plugins, card.spec, llm)

    # Time extraction (index) and the LLM/run separately.
    t_extract0 = time.perf_counter()
    pred: Any = None
    extract_time = 0.0
    query_time = 0.0
    in_tok = 0
    out_tok = 0
    error: str | None = None

    try:
        pipe.index(documents)
        extract_time = time.perf_counter() - t_extract0
        cache_hit = probe.last_hit

        t_query0 = time.perf_counter()
        pred, stats = pipe.run(query_input, documents=documents, reuse_index=True)
        query_time = time.perf_counter() - t_query0
        in_tok = stats.tokens_in
        out_tok = stats.tokens_out
    except Exception as e:
        error = f"{type(e).__name__}: {repr(e)[:200]}"
        cache_hit = probe.last_hit
        extract_time = extract_time or (time.perf_counter() - t_extract0)

    # Normalize prediction to basenames + case-fix against ground truth.
    if isinstance(pred, list):
        pred_list = [str(x) for x in pred]
    elif isinstance(pred, str):
        # Defensive: try to parse a JSON list out of a string completion.
        m = re.search(r"\[.*\]", pred, re.DOTALL)
        if m:
            try:
                arr = json.loads(m.group(0))
                pred_list = [str(x) for x in arr] if isinstance(arr, list) else []
            except Exception:
                pred_list = []
        else:
            pred_list = []
    else:
        pred_list = []

    # Normalize: keep just the basename, drop quotes/spaces.
    pred_basenames = [s.strip("'\" ").split("/")[-1] for s in pred_list]

    gt_map = {g.lower(): g for g in gt_bn}
    pred_fixed = [gt_map.get(p.lower(), p) for p in pred_basenames]

    exact = pred_fixed == gt_bn

    # Error classification (mirror exp21).
    if exact:
        error_type = "correct"
    elif error:
        error_type = "resolver"  # pipeline raised — bucket as resolver/plugin
    elif not pred_fixed:
        error_type = "empty"
    elif len(pred_fixed) != len(gt_bn):
        error_type = "wrong_files"
    elif set(p.lower() for p in pred_fixed) != set(g.lower() for g in gt_bn):
        error_type = "wrong_files"
    else:
        error_type = "wrong_order"

    s_marker = "OK" if exact else "X "
    print(
        f"  [{qi+1:3d}] [{s_marker}] err={error_type:11s} files={len(files)} "
        f"in_tok={in_tok:>5d} out_tok={out_tok:>3d} "
        f"hit={'Y' if cache_hit else 'N'} ext={extract_time:.2f}s "
        f"q={query_time:.2f}s"
        + (f"  ERR={error}" if error else "")
    )

    return {
        "qi": qi,
        "exact": exact,
        "predicted_order": pred_fixed,
        "gt_order": gt_bn,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "n_files": len(files),
        "extraction_time": extract_time,
        "query_time": query_time,
        "cache_hit": bool(cache_hit),
        "error_type": error_type,
        "error": error,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _write_partial(results: list[dict]) -> None:
    Path(RESULTS_JSON).parent.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(results, key=lambda r: r["qi"])
    with open(RESULTS_JSON, "w") as f:
        json.dump(sorted_results, f, indent=2, default=str)


def _summarize(results: list[dict], wall_seconds: float, cache: PixelMemCache,
               workers: int, model_path: str, max_new_tokens: int) -> dict:
    n = len(results)
    if n == 0:
        return {}
    exact = sum(r["exact"] for r in results)
    by_nfiles: dict[str, dict] = {}
    for nf in sorted(set(r["n_files"] for r in results)):
        sub = [r for r in results if r["n_files"] == nf]
        e = sum(r["exact"] for r in sub)
        by_nfiles[str(nf)] = {
            "correct": e,
            "total": len(sub),
            "accuracy": e / len(sub) if sub else 0.0,
        }

    mean_in = sum(r["input_tokens"] for r in results) / n
    mean_out = sum(r["output_tokens"] for r in results) / n
    mean_extract = sum(r["extraction_time"] for r in results) / n
    mean_query = sum(r["query_time"] for r in results) / n
    n_cache_hits = sum(1 for r in results if r["cache_hit"])
    err_counts = Counter(r["error_type"] for r in results)

    summary = {
        "n_total": n,
        "n_correct": exact,
        "accuracy": exact / n,
        "by_file_count": by_nfiles,
        "mean_input_tokens": mean_in,
        "mean_output_tokens": mean_out,
        "mean_extraction_time_s": mean_extract,
        "mean_query_time_s": mean_query,
        "cache_hits": n_cache_hits,
        "cache_hit_rate": n_cache_hits / n,
        "wall_seconds": wall_seconds,
        "error_taxonomy": dict(err_counts),
        "model": MODEL_NAME,
        "backend": "local_transformers",
        "model_path": model_path,
        "max_new_tokens": max_new_tokens,
        "max_workers": workers,
        "pixelmem_cache_stats": cache.stats.to_dict(),
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    add_model_arguments(ap, max_new_tokens=200)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    llm = LocalQwen3(args.model_path, max_new_tokens=args.max_new_tokens)

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"DependEval data not found at {DATA_PATH}; "
            "user said it should already be present."
        )
    with open(DATA_PATH) as f:
        data = json.load(f)
    picks = [d for d in data if 3 <= len(d["files"]) <= 5]

    print(f"EXP 58: CardMem (V5) on DependEval Task 2 ({MODEL_NAME})")
    print(f"  items: {len(picks)}  workers: {args.workers}  cache: {CACHE_DIR}")
    print("=" * 72)

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    cache = PixelMemCache(CACHE_DIR)
    # Clear so the run is reproducible (and so cache_hit measurements
    # reflect this run's behavior, not residue from prior runs).
    cache.clear()

    results: list[dict] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_q, qi, item, cache, llm): qi
            for qi, item in enumerate(picks)
        }
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                qi = futures[fut]
                print(f"  ERROR (qi={qi}): {e}")
                traceback.print_exc()
                results.append({
                    "qi": qi,
                    "exact": False,
                    "predicted_order": [],
                    "gt_order": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "n_files": 0,
                    "extraction_time": 0.0,
                    "query_time": 0.0,
                    "cache_hit": False,
                    "error_type": "resolver",
                    "error": f"thread: {repr(e)[:200]}",
                })
            if len(results) % 10 == 0:
                _write_partial(results)
                done = len(results)
                acc = sum(r["exact"] for r in results) / max(1, done)
                print(
                    f"  --- progress: {done}/{len(picks)} "
                    f"(running acc={acc:.0%}; "
                    f"elapsed={time.perf_counter() - t0:.1f}s) ---"
                )

    _write_partial(results)
    wall = time.perf_counter() - t0

    results.sort(key=lambda r: r["qi"])
    summary = _summarize(results, wall, cache, args.workers,
                         str(llm.path), args.max_new_tokens)

    print(f"\n{'=' * 72}")
    print(f"CARDMEM-ON-DEPENDEVAL RESULTS  (wall: {wall:.1f}s)")
    print(f"{'=' * 72}")
    print(
        f"  Aggregate accuracy: {summary['n_correct']}/{summary['n_total']} "
        f"({summary['accuracy']:.1%})"
    )
    print(
        f"  Mean input tokens : {summary['mean_input_tokens']:.0f}"
    )
    print(f"  Mean output tokens: {summary['mean_output_tokens']:.1f}")
    print(
        f"  Cache hit rate   : {summary['cache_hits']}/{summary['n_total']} "
        f"= {summary['cache_hit_rate']:.1%} "
        f"(expected ~0% for DependEval — each Q is a unique 3-5 file slice)"
    )
    print(f"  Mean extract time: {summary['mean_extraction_time_s']:.2f}s")
    print(f"  Mean query   time: {summary['mean_query_time_s']:.2f}s")
    print(f"\n  By file count:")
    for nf, stats in summary["by_file_count"].items():
        print(
            f"    {nf} files: {stats['correct']}/{stats['total']} "
            f"({stats['accuracy']:.1%})"
        )
    print(f"\n  Error taxonomy:")
    for err, cnt in sorted(
        summary["error_taxonomy"].items(), key=lambda kv: -kv[1]
    ):
        print(f"    {err:12s}: {cnt}")

    Path(SUMMARY_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Saved: {RESULTS_JSON}")
    print(f"  Saved: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
