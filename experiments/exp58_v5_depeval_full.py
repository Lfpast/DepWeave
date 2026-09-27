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

RESULTS_JSON = RESULTS_DIR / "exp58_v5_depeval_full.json"
SUMMARY_JSON = RESULTS_DIR / "exp58_v5_depeval_full_summary.json"


def load_items(path: str = DATA_PATH) -> list[dict]:
    with open(path) as file:
        data = json.load(file)
    return [item for item in data if 3 <= len(item["files"]) <= 5]


async def run_items(items: list[dict], llm, *, max_input_tokens: int = 4096,
                    output: Path | None = None) -> list[dict]:
    records = []
    async with DepWeaveRunner(llm, max_input_tokens=max_input_tokens) as runner:
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
                          "error_type": "pipeline", "error": repr(exc)}
            record["wallclock_s"] = time.perf_counter() - started
            records.append(record)
            print(f"[{qi + 1}/{len(items)}] files={len(files)} "
                  f"exact={'OK' if record['exact'] else 'X'} "
                  f"tokens={record.get('input_tokens', 0)} "
                  f"error={record.get('error')}")
            if output and ((qi + 1) % 10 == 0 or qi + 1 == len(items)):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def summarize(records: list[dict], max_input_tokens: int) -> dict:
    n = len(records)
    return {"system": "DepWeave", "model": MODEL_NAME, "n_total": n,
            "n_correct": sum(bool(r["exact"]) for r in records),
            "accuracy": sum(bool(r["exact"]) for r in records) / n if n else 0,
            "by_file_count": {str(k): {"total": sum(r["n_files"] == k for r in records),
                                       "correct": sum(r["n_files"] == k and r["exact"] for r in records)}
                              for k in sorted({r["n_files"] for r in records})},
            "error_taxonomy": dict(Counter(r["error_type"] for r in records)),
            "max_input_tokens": max_input_tokens,
            "mean_input_tokens": sum(r.get("input_tokens", 0) for r in records) / n if n else 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, max_new_tokens=200)
    parser.add_argument("--limit", type=int, default=0, help="0 runs all 166 eligible items")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=RESULTS_JSON)
    parser.add_argument("--summary", type=Path, default=SUMMARY_JSON)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        records = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        items = load_items()
        if args.limit:
            items = items[:args.limit]
        llm = LocalQwen3(args.model_path, max_new_tokens=args.max_new_tokens)
        records = asyncio.run(run_items(items, llm, max_input_tokens=args.max_input_tokens,
                                        output=args.output))
    summary = summarize(records, args.max_input_tokens)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    failures = sum(bool(r.get("error")) for r in records)
    if failures:
        raise SystemExit(f"{failures} DependEval cases failed in the DepWeave pipeline; see {args.output}")


if __name__ == "__main__":
    main()
