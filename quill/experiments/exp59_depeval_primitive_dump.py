"""Exp 59: Primitive-dump baseline on DependEval Task 2 for the Quill paper.

Isolates "what does the structured prompt + derivation add over just dumping
the extracted primitives?".

For each item:
  1. Run hand-coded PythonDependencyExtractor on each source file -> a flat list of
     (subject, relation, object, condition) quadruples (no derivation,
     no graph, no topo-sort precomputation).
  2. Concatenate ALL quadruples into the prompt as plain text lines.
  3. Ask GPT-4o-mini for the topological order.
  4. Score with the same parser/evaluator as exp57.

This sits between PyDepCard (V4 standalone, 81.3 % / 389 tok), CardMem
("Quill", 84.9 % / 322 tok), and full-text (exp57, 41.6 % / 2722 tok).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: pull OPENAI_API_KEY from ~/.openai_key.
# ---------------------------------------------------------------------------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    keyfile = Path.home() / ".openai_key"
    if keyfile.exists():
        for line in keyfile.read_text().splitlines():
            m = re.match(
                r'\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*[\'"]?([^\'"\s]+)',
                line,
            )
            if m:
                OPENAI_KEY = m.group(1)
                os.environ["OPENAI_API_KEY"] = OPENAI_KEY
                break
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not set (and ~/.openai_key not parseable)")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from benchmarks.dependeval.data import DATA_PATH, parse_dependeval_content
from benchmarks.dependeval.python_deps import PythonDependencyExtractor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = str(REPO_ROOT / "results" / "exp59_depeval_primitive_dump.json")
SUMMARY_JSON = str(REPO_ROOT / "results" / "exp59_depeval_primitive_dump_summary.json")
MAX_WORKERS = 15
MODEL = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_KEY)


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


def ask(prompt: str) -> tuple[str, int, int]:
    try:
        r = client.chat.completions.create(
            model=MODEL,
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return (
            r.choices[0].message.content.strip(),
            r.usage.prompt_tokens,
            r.usage.completion_tokens,
        )
    except Exception as e:
        print(f"  [ask] OpenAI error: {e}")
        return "[]", 0, 0


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------


def process_q(qi: int, item: dict) -> dict:
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
    answer, in_tok, out_tok = ask(prompt)

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
    (REPO_ROOT / "results").mkdir(exist_ok=True)
    sorted_results = sorted(results, key=lambda r: r["qi"])
    with open(RESULTS_JSON, "w") as f:
        json.dump(sorted_results, f, indent=2)


def summarize(results: list[dict], wall_seconds: float) -> dict:
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

    references = {
        "quill_cardmem": {"accuracy": 0.849, "mean_input_tokens": 322},
        "pydepcard_v4": {"accuracy": 0.813, "mean_input_tokens": 389},
        "full_text_exp57": {"accuracy": 0.416, "mean_input_tokens": 2722},
    }

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
        "model": MODEL,
        "max_workers": MAX_WORKERS,
        "references": references,
        "deltas": {
            "vs_quill_accuracy_pp": (exact / n - references["quill_cardmem"]["accuracy"]) * 100,
            "vs_pydepcard_accuracy_pp": (exact / n - references["pydepcard_v4"]["accuracy"]) * 100,
            "vs_full_text_accuracy_pp": (exact / n - references["full_text_exp57"]["accuracy"]) * 100,
            "vs_quill_input_tokens": mean_in - references["quill_cardmem"]["mean_input_tokens"],
            "vs_pydepcard_input_tokens": mean_in - references["pydepcard_v4"]["mean_input_tokens"],
            "vs_full_text_input_tokens": mean_in - references["full_text_exp57"]["mean_input_tokens"],
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"DependEval data not found at {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)
    picks = [d for d in data if 3 <= len(d["files"]) <= 5]

    print(f"EXP 59: Primitive-dump baseline on DependEval Task 2 ({MODEL})")
    print(f"  items: {len(picks)}  workers: {MAX_WORKERS}")
    print("=" * 72)

    results: list[dict] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_q, qi, item): qi
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
    summary = summarize(results, wall)

    print(f"\n{'=' * 72}")
    print(f"PRIMITIVE-DUMP BASELINE RESULTS  (wall: {wall:.1f}s)")
    print(f"{'=' * 72}")
    print(
        f"  Aggregate accuracy: {summary['n_correct']}/{summary['n_total']} "
        f"({summary['accuracy']:.1%})"
    )
    print(f"    vs Quill (CardMem)   84.9% : {summary['deltas']['vs_quill_accuracy_pp']:+.1f} pp")
    print(f"    vs PyDepCard (V4)    81.3% : {summary['deltas']['vs_pydepcard_accuracy_pp']:+.1f} pp")
    print(f"    vs Full-text (exp57) 41.6% : {summary['deltas']['vs_full_text_accuracy_pp']:+.1f} pp")
    print(f"  Mean input tokens : {summary['mean_input_tokens']:.0f}")
    print(f"    vs Quill 322    : {summary['deltas']['vs_quill_input_tokens']:+.0f}")
    print(f"    vs PyDepCard 389: {summary['deltas']['vs_pydepcard_input_tokens']:+.0f}")
    print(f"    vs Full-text 2722: {summary['deltas']['vs_full_text_input_tokens']:+.0f}")
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

    (REPO_ROOT / "results").mkdir(exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved: {RESULTS_JSON}")
    print(f"  Saved: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
