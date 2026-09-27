"""Exp 37 — RepoQA Search Needle Function with V5.

Task: given an entire code repository + a natural-language description of a
"needle" function (with function name obfuscated), identify the function in
the repo that matches the description. Output is the function NAME.

Why this fits V5:
- Input is document-shaped (repo = dict of files).
- Output is structured (a function name from a bounded candidate set).
- Graph-extractable: every function def is a primitive.
- Workload-reducing: repo has N functions; tool narrows to top-K; LLM picks.

V5 pipeline:
- Extractor: pulls every `def NAME(...)` / `class NAME` as primitives
  with their docstring / body-preview provenance.
- Ranker: scores each function by word overlap between the description
  and the function's docstring + name + body snippet.
- Prompt: shows top-K candidates + their snippets, asks LLM to pick.

Metric: exact function name match against the needle's `name` field.
(RepoQA's official metric uses tree-sitter similarity >= 0.8; exact name
match is strictly harder and a good proxy at smoke scale.)

Backend: local Qwen3-4B.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "localMem"))

from core import TaskCard
from core.pipeline import V5Pipeline
from core.plugins import Extractor, PluginSet, PromptTemplate
from core.types import EvidenceBundle, PipelineStats, Primitive, TaskSpec
from core.harness import default_input_adapter
from benchmarks.repoqa import REPOQA_JSON, RepoQAFunctionExtractor, RepoQASearchPrompt

from benchmarks.local_model import LocalQwen3, MODEL_NAME, RESULTS_DIR, add_model_arguments


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_repoqa_examples(seed: int, n_holdout: int):
    with open(REPOQA_JSON) as f:
        data = json.load(f)

    # Gather all (repo, needle) pairs across Python repos. V5's extractor
    # parses each file once and only top-K candidates go to the LLM, so
    # large repos are fine.
    all_cases = []
    for repo_entry in data.get("python", []):
        content = repo_entry.get("content", {})
        if not isinstance(content, dict):
            continue
        for needle in repo_entry.get("needles", []):
            all_cases.append({
                "repo": repo_entry["repo"],
                "content": content,
                "needle_name": needle["name"],
                "needle_description": needle.get("description", ""),
                "needle_path": needle.get("path", ""),
            })

    rng = random.Random(seed)
    rng.shuffle(all_cases)
    picks = all_cases[:n_holdout]
    n_files_avg = sum(len(p['content']) for p in picks) / max(len(picks), 1)
    print(f"[loader] sampled {len(picks)} cases from {len(all_cases)} total "
          f"(avg {n_files_avg:.0f} files/case)")

    examples = []
    for i, c in enumerate(picks):
        examples.append({
            "qid": f"rq_{i}_{c['repo'].split('/')[-1]}_{c['needle_name']}",
            "input": {
                "documents": c["content"],
                "query": {
                    "description": c["needle_description"],
                    "repo": c["repo"],
                },
            },
            "expected_output": c["needle_name"],
        })
    return examples


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run(card, plugins, llm, workers):
    examples = card.spec.holdout

    def run_one(ex):
        docs, qi = default_input_adapter(ex)
        pipe = V5Pipeline(plugins, card.spec, llm)
        try:
            pred, stats = pipe.run(qi, documents=docs)
            correct = str(pred).strip() == ex.expected_output
            return ex, pred, stats, correct, None
        except Exception as e:
            return ex, None, PipelineStats(), False, e

    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed({pool.submit(run_one, ex): ex for ex in examples}):
            results.append(f.result())

    per_case = []
    n_correct = 0
    tok = 0
    for ex, pred, stats, correct, err in sorted(results, key=lambda r: r[0].qid or ""):
        tok += stats.tokens_total
        if correct:
            n_correct += 1
        per_case.append({
            "qid": ex.qid, "correct": correct,
            "pred": repr(err)[:200] if err else pred,
            "expected": ex.expected_output,
            "tokens": stats.tokens_total,
            "n_primitives": stats.n_primitives,
        })
    n = len(examples) or 1
    return {
        "n_examples": n, "n_correct": n_correct,
        "accuracy": n_correct / n,
        "avg_tokens": tok / n,
        "wallclock_s": time.perf_counter() - t0,
        "per_case": per_case,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    add_model_arguments(ap, max_new_tokens=128)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-holdout", type=int, default=5)
    ap.add_argument("--output", default=str(RESULTS_DIR / "exp37_v5_repoqa.json"))
    args = ap.parse_args()
    llm = LocalQwen3(args.model_path, max_new_tokens=args.max_new_tokens)

    examples = _load_repoqa_examples(args.seed, args.n_holdout)
    card = TaskCard.from_dict({
        "domain": "repoqa_search_needle_function",
        "description": "Search Needle Function: find the function in a repo that matches an NL description.",
        "input_schema": {"kind": "doc_set"},
        "query": {"kind": "function_selection", "output": "string"},
        "eval": {"metric": "exact_match", "threshold": 0.5},
        "few_shot": [],
        "holdout": examples,
    })

    plugins = PluginSet(
        name="repoqa_v5",
        extractor=RepoQAFunctionExtractor(),
        prompt_template=RepoQASearchPrompt(),
        derivation_rules=[],
    )

    print(f"=== RepoQA Search Needle Function — V5 ({MODEL_NAME}) ===")
    print(f"holdout={args.n_holdout}")

    t0 = time.perf_counter()
    eval_ = _run(card, plugins, llm, args.workers)
    out = {
        "backend": "local_transformers", "model": MODEL_NAME,
        "model_path": str(llm.path),
        "holdout_qids": [e["qid"] for e in examples],
        "holdout": eval_,
        "wallclock_total_s": time.perf_counter() - t0,
    }

    print(f"\n  accuracy: {eval_['n_correct']}/{eval_['n_examples']} = {eval_['accuracy']:.0%}")
    print(f"  avg tokens: {eval_['avg_tokens']:.0f}")
    print(f"  wallclock: {eval_['wallclock_s']:.1f}s")
    print()
    for c in eval_["per_case"]:
        mark = "✓" if c["correct"] else "✗"
        print(f"  {mark} {c['qid'][:50]:50s}  pred={c['pred']!r}  "
              f"exp={c['expected']!r}  n_prim={c['n_primitives']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    if "--ablation-only" not in sys.argv:
        raise SystemExit("Local-only historical run; use experiments.exp56_repoqa_python_full for DepWeave, or pass --ablation-only")
    sys.argv.remove("--ablation-only")
    main()
