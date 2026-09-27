"""DependEval Task 2 Python: full DepWeave evaluation on 3-5 file items."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.dependeval.data import DATA_PATH, parse_dependeval_content
from benchmarks.local_model import LocalQwen3, MODEL_NAME, RESULTS_DIR, add_model_arguments
from depweave import DepWeaveRunner

RESULTS_JSON = RESULTS_DIR / "depweave_v1_depeval_full.json"
CACHE_DIR = "/tmp/depweave_depeval_dependency_cache"


def load_items(path: str = DATA_PATH) -> list[dict]:
    with open(path) as file:
        data = json.load(file)
    return [item for item in data if 3 <= len(item["files"]) <= 5]


def legacy_full_record(record: dict) -> dict:
    """Keep exactly the twelve per-question fields written by exp58_old.py."""
    return {key: record[key] for key in (
        "qi", "exact", "predicted_order", "gt_order", "input_tokens",
        "output_tokens", "n_files", "extraction_time", "query_time",
        "cache_hit", "error_type", "error")}


def remove_old_sidecars(output: Path, summary_path: Path) -> None:
    output.with_name(output.stem + "_run_meta.json").unlink(missing_ok=True)
    summary_path.with_name(summary_path.stem.removesuffix("_summary") + "_details.json").unlink(missing_ok=True)


async def run_items(items: list[dict], llm, *, max_input_tokens: int = 4096,
                    output: Path | None = None, cache_dir: str | None = None,
                    clear_cache: bool = False,
                    cache_stats_out: dict | None = None) -> list[dict]:
    records = []
    async with DepWeaveRunner(llm, max_input_tokens=max_input_tokens,
                               dependency_cache_dir=cache_dir) as runner:
        if clear_cache and runner.dependency_cache is not None:
            runner.dependency_cache.clear()
        for qi, item in enumerate(items):
            started = time.perf_counter()
            files_raw, contents, gt_raw = parse_dependeval_content(item)
            files = [f.strip("'\" ") for f in files_raw]
            documents = {path: contents.get(path, "") for path in files}
            gold = [Path(f.strip("'\" ")).name for f in gt_raw]
            try:
                result = await runner.dependeval(f"depeval:{qi}", documents, files)
                gt_case = {name.lower(): name for name in gold}
                pred = [gt_case.get(Path(f).name.lower(), Path(f).name)
                        for f in result["predicted_order"]]
                exact = pred == gold
                if exact:
                    error_type = "correct"
                elif not pred:
                    error_type = "empty"
                elif len(pred) != len(gold) or set(pred) != set(gold):
                    error_type = "wrong_files"
                else:
                    error_type = "wrong_order"
                record = {"qi": qi, "n_files": len(files), "gt_order": gold,
                          **result, "predicted_order": pred, "exact": exact,
                          "error_type": error_type, "error": None}
            except Exception as exc:
                record = {"qi": qi, "n_files": len(files), "gt_order": gold,
                          "predicted_order": [], "exact": False,
                          "input_tokens": 0, "output_tokens": 0,
                          "extraction_time": 0.0, "query_time": 0.0,
                          "cache_hit": False,
                          "error_type": "resolver", "error": repr(exc)}
            record["wallclock_s"] = time.perf_counter() - started
            records.append(record)
            print(f"[{qi + 1}/{len(items)}] files={len(files)} "
                  f"exact={'OK' if record['exact'] else 'X'} "
                  f"tokens={record.get('input_tokens', 0)} "
                  f"error={record.get('error')}")
            if output and ((qi + 1) % 10 == 0 or qi + 1 == len(items)):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps([legacy_full_record(r) for r in records], indent=2),
                                  encoding="utf-8")
        if cache_stats_out is not None and runner.dependency_cache is not None:
            cache_stats_out.update(runner.dependency_cache.stats.to_dict())
    return records


def summarize(records: list[dict], max_input_tokens: int = 4096, *, model_path: str | None = None,
              max_new_tokens: int | None = None, run_meta: dict | None = None) -> dict:
    """Keep the pre-merge DependEval summary schema for direct result comparison."""
    run_meta = run_meta or {}
    n = len(records)
    n_correct = sum(bool(r["exact"]) for r in records)
    by_file_count = {}
    for k in sorted({r["n_files"] for r in records}):
        total = sum(r["n_files"] == k for r in records)
        correct = sum(r["n_files"] == k and r["exact"] for r in records)
        by_file_count[str(k)] = {"correct": correct, "total": total,
                                 "accuracy": correct / total}
    def mean_if_recorded(key: str) -> float | None:
        if not n or not all(key in r and r[key] is not None for r in records):
            return None
        return sum(r[key] for r in records) / n

    has_phase_times = bool(n and all(
        all(key in r for key in ("global_index_time_s", "local_extract_time_s", "wallclock_s"))
        for r in records))
    if n and all("cache_hit" in r for r in records):
        cache_hits = sum(bool(r["cache_hit"]) for r in records)
    elif has_phase_times:
        # This DepWeave entrypoint runs without PixelMemCache.
        cache_hits = 0
    else:
        cache_hits = None
    if n and all("extraction_time" in r and "query_time" in r for r in records):
        mean_extraction = mean_if_recorded("extraction_time")
        mean_query = mean_if_recorded("query_time")
    elif has_phase_times:
        # Equivalent system stages: prepare (global index + local extraction),
        # then all remaining per-case work, including evidence lookup and Qwen.
        mean_extraction = sum(r["global_index_time_s"] + r["local_extract_time_s"]
                              for r in records) / n
        mean_query = sum(r["wallclock_s"] - r["global_index_time_s"]
                         - r["local_extract_time_s"] for r in records) / n
    else:
        mean_extraction = None
        mean_query = None
    cache_stats = run_meta.get("pixelmem_cache_stats")
    cache_disabled = run_meta.get("cache_enabled") is False
    if cache_stats is None:
        cache_stats = {
            "hits": cache_hits,
            "misses": 0 if cache_disabled else n - cache_hits if cache_hits is not None else None,
            "hit_rate": cache_hits / n if cache_hits is not None else None,
            "total_wallclock_extract_s": 0.0 if cache_disabled else None,
            "total_wallclock_load_s": 0.0 if cache_disabled else None,
            "total_wallclock_save_s": 0.0 if cache_disabled else None,
            "bytes_written": 0 if cache_disabled else None,
            "primitives_cached": 0 if cache_disabled else None,
            "cache_entries": 0 if cache_disabled else None,
            "avg_entry_size_bytes": 0.0 if cache_disabled else None,
        }
    case_wall_total = (sum(r["wallclock_s"] for r in records)
                       if n and all("wallclock_s" in r for r in records) else None)

    return {
        "n_total": n, "n_correct": n_correct,
        "accuracy": n_correct / n if n else None,
        "by_file_count": by_file_count,
        "mean_input_tokens": mean_if_recorded("input_tokens"),
        "mean_output_tokens": mean_if_recorded("output_tokens"),
        "mean_extraction_time_s": mean_extraction,
        "mean_query_time_s": mean_query,
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hits / n if cache_hits is not None else None,
        "wall_seconds": run_meta.get("suite_wallclock_s", case_wall_total),
        "error_taxonomy": dict(Counter(r["error_type"] for r in records)),
        "model": MODEL_NAME, "backend": "local_transformers",
        "model_path": model_path, "max_new_tokens": max_new_tokens,
        "max_workers": 1,
        "pixelmem_cache_stats": cache_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, max_new_tokens=200)
    parser.add_argument("--limit", type=int, default=0, help="0 runs all 166 eligible items")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_JSON)
    parser.add_argument("--summary", type=Path, default=None,
                        help="Summary path (default: <output stem>_summary.json)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Recompute from --output, retaining run-only values in the existing summary")
    args = parser.parse_args()
    summary_path = args.summary or args.output.with_name(args.output.stem + "_summary.json")
    if args.summary_only:
        records = json.loads(args.output.read_text(encoding="utf-8"))
        if not summary_path.exists():
            parser.error("Exact suite wall time and PixelMem timings cannot be recovered "
                         "from the old full.json alone; an existing summary is required")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        run_info = {"suite_wallclock_s": previous["wall_seconds"],
                    "pixelmem_cache_stats": previous["pixelmem_cache_stats"]}
        model_path = previous["model_path"]
        max_new_tokens = previous["max_new_tokens"]
    else:
        if not args.cache_dir:
            parser.error("--cache-dir must name a directory for DependEval extraction cache")
        items = load_items()
        if args.limit > 0:
            items = items[:args.limit]
        llm = LocalQwen3(args.model_path, max_new_tokens=args.max_new_tokens)
        model_path = str(llm.path)
        max_new_tokens = args.max_new_tokens
        cache_stats = {}
        started = time.perf_counter()
        records = asyncio.run(run_items(items, llm, max_input_tokens=args.max_input_tokens,
                                        output=args.output, cache_dir=args.cache_dir,
                                        clear_cache=True, cache_stats_out=cache_stats))
        records = [legacy_full_record(r) for r in records]
        run_info = {"suite_wallclock_s": time.perf_counter() - started,
                    "pixelmem_cache_stats": cache_stats}
    summary = summarize(records, args.max_input_tokens,
                        model_path=model_path, max_new_tokens=max_new_tokens,
                        run_meta=run_info)
    if args.summary_only and (previous["n_total"] != summary["n_total"] or
                              previous["n_correct"] != summary["n_correct"] or
                              previous["cache_hits"] != summary["cache_hits"]):
        parser.error("Existing summary does not match this full.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    remove_old_sidecars(args.output, summary_path)
    print(json.dumps(summary, indent=2))
    failures = sum(bool(r.get("error")) for r in records)
    if failures:
        raise SystemExit(f"{failures} DependEval cases failed in the DepWeave pipeline; see {args.output}")


if __name__ == "__main__":
    main()
