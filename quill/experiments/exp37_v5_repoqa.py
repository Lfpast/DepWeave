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

Backend: gpt-4o-mini, parallel 5 workers.
"""

from __future__ import annotations

import argparse
import ast as _pyast
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

from quill import TaskCard
from quill.core.pipeline import V5Pipeline
from quill.core.plugins import Extractor, PluginSet, PromptTemplate
from quill.core.types import EvidenceBundle, PipelineStats, Primitive, TaskSpec
from quill.harness import default_input_adapter

from experiments.exp22_v5_haiku_smoke import openai_4omini


REPOQA_JSON = "/tmp/repoqa-2024-06-23.json"


# ---------------------------------------------------------------------------
# RepoQA extractor: one primitive per function definition in the repo
# ---------------------------------------------------------------------------


class RepoQAFunctionExtractor(Extractor):
    """Parses every Python file in the repo and emits a Primitive per top-level
    function/method. Provenance stores docstring, name, and a body snippet.
    """

    def extract(self, documents: dict[str, str], **kwargs: Any) -> list[Primitive]:
        out: list[Primitive] = []
        for path, code in documents.items():
            if not isinstance(code, str) or not path.endswith(".py"):
                continue
            try:
                tree = _pyast.parse(code)
            except SyntaxError:
                continue
            for node in _pyast.walk(tree):
                if isinstance(node, (_pyast.FunctionDef, _pyast.AsyncFunctionDef)):
                    docstring = _pyast.get_docstring(node) or ""
                    body_src = self._body_snippet(code, node)
                    out.append(Primitive(
                        path,
                        "defines_function",
                        node.name,
                        f"lineno={node.lineno}",
                        {
                            "path": path,
                            "lineno": node.lineno,
                            "docstring": docstring,
                            "snippet": body_src,
                        },
                    ))
        return out

    def _body_snippet(self, code: str, node, max_chars: int = 600) -> str:
        # Pull the function's source by line numbers (approximate).
        lines = code.splitlines()
        start = node.lineno - 1
        end = min(start + 20, len(lines))
        return "\n".join(lines[start:end])[:max_chars]


# ---------------------------------------------------------------------------
# RepoQA prompt — candidate-aware, ranks by overlap with NL description
# ---------------------------------------------------------------------------


_STOP = frozenset({
    "the", "a", "an", "is", "and", "or", "of", "to", "in", "on", "for",
    "with", "by", "from", "at", "as", "that", "this", "it", "its",
    "be", "are", "was", "were", "will", "can", "should", "would",
    "if", "else", "when", "where", "what", "how", "which", "who",
    "not", "no", "but", "also", "these", "those", "any", "all",
    "function", "method", "returns", "return", "takes", "given",
    "value", "values", "object", "objects", "code",
})


def _tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_]+", (s or "").lower())
            if w not in _STOP and len(w) > 2]


def _score_function_against_desc(func_name: str, docstring: str,
                                 snippet: str, desc_tokens: set[str]) -> int:
    """Count overlapping tokens (description word count that appear in function context)."""
    cand_tokens = set(_tokens(func_name)) | set(_tokens(docstring)) | set(_tokens(snippet))
    return len(cand_tokens & desc_tokens)


class RepoQASearchPrompt(PromptTemplate):
    def build(self, task: TaskSpec, query_input: dict, evidence: EvidenceBundle) -> str:
        description = query_input.get("description", "") or ""
        desc_tokens = set(_tokens(description))

        # Score all function primitives and pick top-K.
        scored = []
        for p in evidence.raw_primitives:
            if p.relation != "defines_function":
                continue
            prov = p.provenance or {}
            s = _score_function_against_desc(
                p.object, prov.get("docstring", ""),
                prov.get("snippet", ""), desc_tokens,
            )
            scored.append((s, p))

        scored.sort(key=lambda x: -x[0])
        top_k = 12
        top = scored[:top_k]

        cand_lines = []
        for score, p in top:
            prov = p.provenance or {}
            doc1 = (prov.get("docstring") or "").splitlines()[0] if prov.get("docstring") else ""
            cand_lines.append(
                f"  {p.object}  (score={score})"
                + (f"\n     docstring: {doc1[:120]}" if doc1 else "")
                + f"\n     path: {prov.get('path')}:{prov.get('lineno')}"
            )
        cand_block = "\n".join(cand_lines) if cand_lines else "  (no candidates extracted)"

        return (
            "Task: pick the Python function from the candidate list below whose "
            "behavior best matches the natural-language description. The "
            "function name itself has been obfuscated in the description.\n\n"
            f"DESCRIPTION:\n{description[:2000]}\n\n"
            f"TOP-{len(top)} CANDIDATE FUNCTIONS (ranked by word overlap):\n"
            f"{cand_block}\n\n"
            "Return ONLY the exact function name (the identifier shown at the "
            "start of each candidate row). No labels like 'C0', no 'score=', "
            "no explanation. Just the identifier."
        )

    def parse(self, completion: str, task: TaskSpec) -> str:
        txt = completion.strip()
        if txt.startswith("```"):
            txt = "\n".join(l for l in txt.splitlines() if not l.startswith("```")).strip()
        # First identifier-like token
        m = re.search(r"[A-Za-z_][A-Za-z0-9_]*", txt)
        if not m:
            raise ValueError("no identifier in completion")
        return m.group(0)


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
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-holdout", type=int, default=5)
    ap.add_argument("--output", default="results/exp37_v5_repoqa.json")
    args = ap.parse_args()

    def llm(p):
        return openai_4omini(p, model=args.model)

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

    print(f"=== RepoQA Search Needle Function — V5 ({args.model}) ===")
    print(f"holdout={args.n_holdout}")

    t0 = time.perf_counter()
    eval_ = _run(card, plugins, llm, args.workers)
    out = {
        "backend": "openai", "model": args.model,
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
    main()
