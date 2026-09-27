"""RepoQA Python: full DepWeave global navigation plus local evidence.

Official name accuracy is retained. Canonical entity accuracy is reported
separately against the dataset's source path and definition line.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.local_model import LocalQwen3, MODEL_NAME, RESULTS_DIR, add_model_arguments
from benchmarks.repoqa import REPOQA_JSON
from depweave import DepWeaveRunner

RESULTS_PATH = RESULTS_DIR / "exp56_repoqa_python_full.json"
SUMMARY_PATH = RESULTS_DIR / "exp56_repoqa_python_full_summary.json"


def load_cases(path: str = REPOQA_JSON) -> list[dict]:
    with open(path) as file:
        data = json.load(file)
    return [{"repo": repo["repo"], "content": repo["content"],
             "name": needle["name"], "description": needle["description"],
             "path": needle["path"], "line": needle["start_line"] + 1}
            for repo in data["python"] for needle in repo["needles"]]


def expected_id(snapshot_id: str, case: dict) -> str | None:
    path = case["path"]
    source = case["content"].get(path)
    if not isinstance(source, str):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    found = []

    def walk(body, owners):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = ".".join([*owners, node.name])
                if node.lineno == case["line"] and node.name == case["name"]:
                    found.append(qual)
                walk(node.body, [*owners, node.name])

    walk(tree.body, [])
    if len(found) != 1:
        return None
    return f"{snapshot_id}|{path}|{path}::{found[0]}|{case['line']}"


async def run_cases(cases: list[dict], llm, *, max_input_tokens: int = 4096,
                    cache_dir: str | None = None, output: Path | None = None) -> list[dict]:
    records = []
    async with DepWeaveRunner(llm, max_input_tokens=max_input_tokens,
                               cache_dir=cache_dir) as runner:
        for i, case in enumerate(cases):
            started = time.perf_counter()
            try:
                result = await runner.repoqa(case["repo"], case["content"], case["description"])
                gold_id = expected_id(result["snapshot_id"], case)
                record = {"qi": i, "repo": case["repo"], "expected_function_name": case["name"],
                          "needle_path": case["path"], "needle_line": case["line"],
                          "expected_id": gold_id, "exact": result["predicted"] == case["name"],
                          "entity_exact": result["predicted_id"] == gold_id if gold_id else None,
                          **result, "error": None}
            except Exception as exc:
                record = {"qi": i, "repo": case["repo"], "expected_function_name": case["name"],
                          "needle_path": case["path"], "needle_line": case["line"],
                          "exact": False, "entity_exact": None, "error": repr(exc)}
            record["wallclock_s"] = time.perf_counter() - started
            records.append(record)
            print(f"[{i + 1}/{len(cases)}] {case['repo']}::{case['name']} "
                  f"name={'OK' if record['exact'] else 'X'} "
                  f"entity={record.get('entity_exact')} "
                  f"tokens={record.get('input_tokens', 0)} "
                  f"error={record.get('error')}")
            if output and ((i + 1) % 5 == 0 or i + 1 == len(cases)):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def summarize(records: list[dict], max_input_tokens: int) -> dict:
    n = len(records)
    scored = [r for r in records if r.get("entity_exact") is not None]
    return {"system": "DepWeave", "model": MODEL_NAME, "n_total": n,
            "name_correct": sum(bool(r["exact"]) for r in records),
            "name_accuracy": sum(bool(r["exact"]) for r in records) / n if n else 0,
            "entity_scored": len(scored),
            "entity_correct": sum(bool(r["entity_exact"]) for r in scored),
            "entity_accuracy": sum(bool(r["entity_exact"]) for r in scored) / len(scored) if scored else None,
            "max_input_tokens": max_input_tokens,
            "mean_input_tokens": sum(r.get("input_tokens", 0) for r in records) / n if n else 0,
            "errors": sum(bool(r.get("error")) for r in records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, max_new_tokens=256)
    parser.add_argument("--limit", type=int, default=0, help="0 runs all Python needles")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--cache-dir", default="/tmp/depweave_repoqa_local_cache")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    cases = load_cases()
    if args.limit:
        cases = cases[:args.limit]
    llm = LocalQwen3(args.model_path, max_new_tokens=args.max_new_tokens)
    records = asyncio.run(run_cases(cases, llm, max_input_tokens=args.max_input_tokens,
                                    cache_dir=args.cache_dir, output=args.output))
    summary = summarize(records, args.max_input_tokens)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(f"{summary['errors']} RepoQA cases failed in the DepWeave pipeline; see {args.output}")


if __name__ == "__main__":
    main()
