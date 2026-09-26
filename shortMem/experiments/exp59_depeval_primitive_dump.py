"""Exp 59: Primitive-dump baseline on DependEval Task 2.

Isolates "what does the structured prompt + derivation add over just dumping
the extracted primitives?".

For each item:
  1. Run hand-coded PythonDependencyExtractor on each source file -> a flat list of
     (subject, relation, object, condition) quadruples (no derivation,
     no graph, no topo-sort precomputation).
  2. Concatenate ALL quadruples into the prompt as plain text lines.
  3. Ask local Qwen3-4B for the topological order.
  4. Score with the same parser/evaluator as exp57.

Compare this output with CardMem and full-text runs made with the same local
Qwen3-4B checkpoint. Historical GPT-4o-mini scores use a different protocol.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.dependeval.data import DATA_PATH, parse_dependeval_content
from benchmarks.dependeval.python_deps import PythonDependencyExtractor
from benchmarks.local_model import LocalQwen3, MODEL_NAME, RESULTS_DIR, add_model_arguments


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_JSON = str(RESULTS_DIR / "exp59_depeval_primitive_dump.json")
SUMMARY_JSON = str(RESULTS_DIR / "exp59_depeval_primitive_dump_summary.json")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def format_quad(p) -> str:
    """Format a Primitive as `<subject> <relation> <object> [<condition>]`."""
    if p.condition:
        return f"{p.subject} {p.relation} {p.object} [{p.condition}]"
    return f"{p.subject} {p.relation} {p.object}"


def build_prompt(quad_lines: list[str], basenames: list[str]) -> str:
    parts = [
        "Below are knowledge-graph quadruples extracted from 3-5 Python source",
        'files. Each line is "<subject> <relation> <object> [<condition>]".',
        "",
        "Output a JSON list giving the correct topological ordering of the files",
        "(most-fundamental first, most-dependent last) -- only the basenames,",
        "in order.",
        "",
        "QUADRUPLES (extracted, no derivation applied):",
    ]
    parts.extend(quad_lines)
    parts.append("")
    parts.append("FILES:")
    parts.extend(basenames)
    parts.append("")
    parts.append("Output ONLY the JSON list, no explanation.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------


def process_q(qi: int, item: dict, llm) -> dict:
    f_raw, fc, gt_raw = parse_dependeval_content(item)
    files = [f.strip("'\"") for f in f_raw]
    gt_bn = [f.strip("'\"").split("/")[-1] for f in gt_raw]
    basenames = [f.split("/")[-1] for f in files]

    # documents dict keyed by full file path; matches PythonDependencyExtractor's API
    documents = {f: fc.get(f, "") for f in files}

    # Run the V4 extractor on the FULL set of files together (so internal
    # alias resolution works the same way it does in PyDepCard); collect
    # the flat list of primitives.
    extractor = PythonDependencyExtractor(language="python")
    try:
        primitives = extractor.extract(documents)
    except Exception as e:
        print(f"  [{qi+1:3d}] extractor error: {e}")
        primitives = []

    quad_lines = [format_quad(p) for p in primitives]
    n_quadruples = len(quad_lines)

    prompt = build_prompt(quad_lines, basenames)
    answer, in_tok, out_tok = llm(prompt)

    # Parse prediction (mirror exp57)
    try:
        m = re.search(r"\[.*\]", answer, re.DOTALL)
        pred = (
            [f.strip("'\" ").split("/")[-1] for f in json.loads(m.group(0))]
            if m
            else []
        )
    except Exception:
        pred = []

    gt_map: dict[str, str] = {}
    for g in gt_bn:
        gt_map[g.lower()] = g
        gt_map[g.lower().removesuffix(".py")] = g
    pred_fixed: list[str] = []
    for p in pred:
        key = p.lower()
        if key in gt_map:
            pred_fixed.append(gt_map[key])
        elif key.removesuffix(".py") in gt_map:
            pred_fixed.append(gt_map[key.removesuffix(".py")])
        else:
            pred_fixed.append(p)

    exact = pred_fixed == gt_bn

    if exact:
        error_type = "correct"
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
        f"quads={n_quadruples:>4d} in_tok={in_tok:>5d} out_tok={out_tok:>3d}"
    )

    return {
        "qi": qi,
        "exact": exact,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "n_files": len(files),
        "n_quadruples": n_quadruples,
        "predicted_order": pred_fixed,
        "gt_order": gt_bn,
        "error_type": error_type,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_partial(results: list[dict]) -> None:
    Path(RESULTS_JSON).parent.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(results, key=lambda r: r["qi"])
    with open(RESULTS_JSON, "w") as f:
        json.dump(sorted_results, f, indent=2)


def summarize(results: list[dict], wall_seconds: float, workers: int,
              model_path: str, max_new_tokens: int) -> dict:
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
    mean_quads = sum(r["n_quadruples"] for r in results) / n
    err_counts = Counter(r["error_type"] for r in results)

    summary = {
        "n_total": n,
        "n_correct": exact,
        "accuracy": exact / n,
        "by_file_count": by_nfiles,
        "mean_input_tokens": mean_in,
        "mean_output_tokens": mean_out,
        "mean_quadruples_per_item": mean_quads,
        "wall_seconds": wall_seconds,
        "error_taxonomy": dict(err_counts),
        "model": MODEL_NAME,
        "backend": "local_transformers",
        "model_path": model_path,
        "max_new_tokens": max_new_tokens,
        "max_workers": workers,
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
        raise FileNotFoundError(f"DependEval data not found at {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)
    picks = [d for d in data if 3 <= len(d["files"]) <= 5]

    print(f"EXP 59: Primitive-dump baseline on DependEval Task 2 ({MODEL_NAME})")
    print(f"  items: {len(picks)}  workers: {args.workers}")
    print("=" * 72)

    results: list[dict] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_q, qi, item, llm): qi
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
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "n_files": 0,
                    "n_quadruples": 0,
                    "predicted_order": [],
                    "gt_order": [],
                    "error_type": "exception",
                })
            if len(results) % 10 == 0:
                write_partial(results)
                done = len(results)
                acc = sum(r["exact"] for r in results) / max(1, done)
                print(
                    f"  --- progress: {done}/{len(picks)} "
                    f"(running acc={acc:.0%}; "
                    f"elapsed={time.perf_counter() - t0:.1f}s) ---"
                )

    write_partial(results)
    wall = time.perf_counter() - t0

    results.sort(key=lambda r: r["qi"])
    summary = summarize(results, wall, args.workers,
                        str(llm.path), args.max_new_tokens)

    print(f"\n{'=' * 72}")
    print(f"PRIMITIVE-DUMP BASELINE RESULTS  (wall: {wall:.1f}s)")
    print(f"{'=' * 72}")
    print(
        f"  Aggregate accuracy: {summary['n_correct']}/{summary['n_total']} "
        f"({summary['accuracy']:.1%})"
    )
    print(f"  Mean input tokens : {summary['mean_input_tokens']:.0f}")
    print(f"  Mean output tokens: {summary['mean_output_tokens']:.1f}")
    print(f"  Mean quadruples/item: {summary['mean_quadruples_per_item']:.1f}")
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
        json.dump(summary, f, indent=2)

    print(f"\n  Saved: {RESULTS_JSON}")
    print(f"  Saved: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
