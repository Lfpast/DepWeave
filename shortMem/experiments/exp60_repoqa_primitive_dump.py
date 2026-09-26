"""Exp 60 — RepoQA Python primitive-dump baseline.

Goal: isolate the contribution of the Quill candidate-ranker (top-12 by
word-overlap) by stripping it out and dumping ALL extracted function
definitions into the prompt instead. Same hand-coded
``RepoQAFunctionExtractor`` is applied to the repo, but rather than the
ranked top-12 list, we concatenate every (path, name, docstring) entry
up to a 30 K-char cap, and ask local Qwen3-4B to pick the matching name.

Compare this candidate-ranker ablation with Config A and C from exp 56
using the same local Qwen3-4B checkpoint.

Outputs:
- ``results/exp60_repoqa_primitive_dump.json``: per-question records
- ``results/exp60_repoqa_primitive_dump_summary.json``: aggregates
"""

from __future__ import annotations

import json
import argparse
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.repoqa import REPOQA_JSON, RepoQAFunctionExtractor
from benchmarks.local_model import LocalQwen3, MODEL_NAME, RESULTS_DIR, add_model_arguments

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_PATH = str(RESULTS_DIR / "exp60_repoqa_primitive_dump.json")
SUMMARY_PATH = str(RESULTS_DIR / "exp60_repoqa_primitive_dump_summary.json")

PROMPT_CHAR_CAP = 30000  # cap on the function-list block
MAX_FUNCTIONS_HARD = 500  # secondary cap by entry count
DOCSTRING_PREVIEW_CHARS = 120  # docstring excerpt per line
MAX_OUTPUT_TOKENS = 80


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_python_needles() -> list[dict]:
    with open(REPOQA_JSON) as f:
        data = json.load(f)
    cases = []
    for repo_entry in data.get("python", []):
        content = repo_entry.get("content", {})
        if not isinstance(content, dict):
            continue
        for needle in repo_entry.get("needles", []):
            cases.append({
                "repo": repo_entry["repo"],
                "content": content,
                "needle_name": needle["name"],
                "needle_description": needle.get("description", ""),
                "needle_path": needle.get("path", ""),
            })
    return cases


# ---------------------------------------------------------------------------
# Per-repo extraction cache (10 needles share one repo => 1 extraction each)
# ---------------------------------------------------------------------------


_extractor = RepoQAFunctionExtractor()
_repo_cache: dict[str, list] = {}
import threading
_repo_cache_lock = threading.Lock()


def _get_primitives_for_repo(repo: str, content: dict) -> list:
    with _repo_cache_lock:
        if repo in _repo_cache:
            return _repo_cache[repo]
    prims = list(_extractor.extract(content))
    with _repo_cache_lock:
        _repo_cache.setdefault(repo, prims)
        return _repo_cache[repo]


# ---------------------------------------------------------------------------
# Prompt building (NO ranking — every primitive in extraction order)
# ---------------------------------------------------------------------------


def _build_function_lines(primitives: list) -> list[str]:
    """Return one line per ``defines_function`` primitive:
       ``<path>::<name> [docstring excerpt]``."""
    lines = []
    for p in primitives:
        if p.relation != "defines_function":
            continue
        prov = p.provenance or {}
        path = prov.get("path") or p.subject
        name = p.object
        ds = (prov.get("docstring") or "").strip()
        if ds:
            ds_first = ds.splitlines()[0][:DOCSTRING_PREVIEW_CHARS]
            line = f"{path}::{name} [{ds_first}]"
        else:
            line = f"{path}::{name}"
        lines.append(line)
    return lines


def _truncate_lines(lines: list[str]) -> tuple[str, int, bool]:
    """Concatenate lines (newline-joined) up to PROMPT_CHAR_CAP and
    MAX_FUNCTIONS_HARD entries. Returns (block, n_in_prompt, truncated)."""
    truncated = False
    total = 0
    kept = []
    for ln in lines:
        if len(kept) >= MAX_FUNCTIONS_HARD:
            truncated = True
            break
        added_chars = len(ln) + 1  # newline
        if total + added_chars > PROMPT_CHAR_CAP:
            truncated = True
            break
        kept.append(ln)
        total += added_chars
    return "\n".join(kept), len(kept), truncated


def _build_prompt(description: str, primitives: list) -> tuple[str, int, int, bool]:
    """Returns (prompt, n_extracted, n_in_prompt, truncated)."""
    lines = _build_function_lines(primitives)
    n_extracted = len(lines)
    block, n_in_prompt, truncated = _truncate_lines(lines)

    header_funcs = (
        f"EXTRACTED FUNCTIONS"
        + (f" (truncated at {PROMPT_CHAR_CAP} chars; showing {n_in_prompt} of {n_extracted}):"
           if truncated
           else f" ({n_in_prompt} total):")
    )

    prompt = (
        "Below are ALL function definitions extracted from a Python repo. "
        "Each line is \"<path>::<function_name> [docstring excerpt]\".\n\n"
        f"DESCRIPTION (function name obfuscated):\n{(description or '')[:2000]}\n\n"
        f"{header_funcs}\n"
        f"{block}\n\n"
        "Output format: a single line with EXACTLY this format (no other "
        "text, no code fence, no explanation):\n"
        "ANSWER: <function_name>\n"
        "where <function_name> is the bare Python identifier of the matching "
        "function from the list above."
    )
    return prompt, n_extracted, n_in_prompt, truncated


_ANSWER_RE = re.compile(r"ANSWER\s*:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?")
_BACKTICK_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
_STOP_WORDS = {
    "the", "this", "that", "based", "answer", "function", "name", "is",
    "would", "looks", "given", "from", "context", "above", "below",
    "snippet", "code", "python", "repository", "matches", "match",
}


def _parse_answer(completion: str) -> str:
    txt = (completion or "").strip()
    if txt.startswith("```"):
        txt = "\n".join(l for l in txt.splitlines() if not l.startswith("```")).strip()
    m = _ANSWER_RE.search(txt)
    if m:
        return m.group(1)
    m = _BACKTICK_RE.search(txt)
    if m:
        return m.group(1)
    for tok in _IDENT_RE.findall(txt):
        if tok.lower() not in _STOP_WORDS and len(tok) > 3:
            return tok
    return ""


# ---------------------------------------------------------------------------
# Per-question worker
# ---------------------------------------------------------------------------


def _qid(case: dict) -> str:
    return f"{case['repo'].split('/')[-1]}::{case['needle_name']}"


def _process_case(qi: int, case: dict, llm) -> dict:
    qid = _qid(case)
    expected = case["needle_name"]
    t0 = time.perf_counter()

    record: dict = {
        "question_id": qid,
        "qi": qi,
        "repo": case["repo"],
        "needle_function": expected,
        "expected": expected,
        "needle_path": case["needle_path"],
        "needle_description_preview": (case["needle_description"] or "")[:160],
        "predicted": None,
        "exact": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "n_functions_extracted": 0,
        "n_functions_in_prompt": 0,
        "chars_in_prompt": 0,
        "truncated": False,
        "wallclock_s": 0.0,
    }

    try:
        primitives = _get_primitives_for_repo(case["repo"], case["content"])
    except Exception as e:
        record["error"] = f"extract: {repr(e)[:200]}"
        record["wallclock_s"] = time.perf_counter() - t0
        return record

    try:
        prompt, n_ext, n_in, truncated = _build_prompt(
            case["needle_description"], primitives,
        )
    except Exception as e:
        record["error"] = f"prompt: {repr(e)[:200]}"
        record["wallclock_s"] = time.perf_counter() - t0
        return record

    record["n_functions_extracted"] = n_ext
    record["n_functions_in_prompt"] = n_in
    record["chars_in_prompt"] = len(prompt)
    record["truncated"] = truncated

    try:
        completion, t_in, t_out = llm(prompt)
    except Exception as e:
        record["error"] = f"llm: {repr(e)[:200]}"
        record["wallclock_s"] = time.perf_counter() - t0
        return record

    pred = _parse_answer(completion)
    record["predicted"] = pred
    record["exact"] = (pred == expected)
    record["input_tokens"] = t_in
    record["output_tokens"] = t_out
    record["raw_completion"] = completion[:200]
    record["wallclock_s"] = time.perf_counter() - t0

    print(
        f"  [{qi+1:3d}/100] [{ 'OK' if record['exact'] else 'X '}] "
        f"{case['repo'].split('/')[-1]:<28s} "
        f"funcs={n_in}/{n_ext}{'T' if truncated else ' '} "
        f"in_tok={t_in:>5d} out_tok={t_out:>3d} "
        f"pred={pred!r:<32s} expected={expected!r}",
        flush=True,
    )
    return record


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _write_partial(records: list[dict]) -> None:
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.get("qi", 0))
    with open(RESULTS_PATH, "w") as f:
        json.dump(ordered, f, indent=2, default=str)


def _summarize(records: list[dict], wall: float, workers: int,
               model_path: str, max_new_tokens: int) -> dict:
    n = len(records)
    if n == 0:
        return {}
    n_exact = sum(1 for r in records if r.get("exact"))
    n_err = sum(1 for r in records if r.get("error"))
    n_trunc = sum(1 for r in records if r.get("truncated"))

    def _mean(field: str) -> float:
        vals = [int(r.get(field, 0) or 0) for r in records]
        return sum(vals) / max(n, 1)

    summary = {
        "model": MODEL_NAME,
        "backend": "local_transformers",
        "model_path": model_path,
        "max_new_tokens": max_new_tokens,
        "max_workers": workers,
        "n_total": n,
        "n_correct": n_exact,
        "accuracy": n_exact / n,
        "n_errors": n_err,
        "mean_input_tokens": _mean("input_tokens"),
        "mean_output_tokens": _mean("output_tokens"),
        "mean_n_functions_extracted": _mean("n_functions_extracted"),
        "mean_n_functions_in_prompt": _mean("n_functions_in_prompt"),
        "mean_chars_in_prompt": _mean("chars_in_prompt"),
        "n_truncated": n_trunc,
        "truncation_rate": n_trunc / n,
        "prompt_char_cap": PROMPT_CHAR_CAP,
        "max_functions_hard": MAX_FUNCTIONS_HARD,
        "wall_seconds": wall,
    }
    # Per-repo accuracy
    by_repo: dict[str, list[dict]] = {}
    for r in records:
        by_repo.setdefault(r["repo"], []).append(r)
    repo_stats = {}
    for repo, rs in by_repo.items():
        c = sum(1 for r in rs if r.get("exact"))
        repo_stats[repo] = {
            "correct": c,
            "total": len(rs),
            "accuracy": c / len(rs) if rs else 0.0,
        }
    summary["by_repo"] = repo_stats
    return summary


def main():
    ap = argparse.ArgumentParser()
    add_model_arguments(ap, max_new_tokens=MAX_OUTPUT_TOKENS)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    llm = LocalQwen3(args.model_path, max_new_tokens=args.max_new_tokens)
    cases = _load_python_needles()
    print(f"=== Exp 60: RepoQA Python primitive-dump baseline ({MODEL_NAME}, workers={args.workers}) ===")
    print(f"  n_needles={len(cases)} prompt_cap={PROMPT_CHAR_CAP} max_funcs={MAX_FUNCTIONS_HARD}")
    print(f"  output -> {RESULTS_PATH}")
    repos = sorted({c["repo"] for c in cases})
    print(f"  {len(repos)} repos:")
    for r in repos:
        n = sum(1 for c in cases if c["repo"] == r)
        print(f"    {r}: {n} needles")
    print("=" * 72)

    # Pre-warm extraction cache sequentially: avoid concurrent threads
    # extracting the same repo. Total extraction time is small (~seconds per
    # repo) and this guarantees we count one extraction per repo.
    print("\nPre-extracting all repos (one pass, sequential)...")
    t_extract0 = time.perf_counter()
    for r in repos:
        case = next(c for c in cases if c["repo"] == r)
        prims = _get_primitives_for_repo(r, case["content"])
        n_funcs = sum(1 for p in prims if p.relation == "defines_function")
        print(f"  {r}: {n_funcs} functions extracted")
    print(f"  total extraction wall: {time.perf_counter()-t_extract0:.1f}s")

    print(f"\nDispatching {len(cases)} LLM calls (workers={args.workers}) ...")
    t0 = time.perf_counter()
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_process_case, qi, case, llm): qi
            for qi, case in enumerate(cases)
        }
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                qi = futs[fut]
                rec = {
                    "qi": qi,
                    "question_id": _qid(cases[qi]),
                    "repo": cases[qi]["repo"],
                    "needle_function": cases[qi]["needle_name"],
                    "expected": cases[qi]["needle_name"],
                    "predicted": None,
                    "exact": False,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "n_functions_extracted": 0,
                    "n_functions_in_prompt": 0,
                    "chars_in_prompt": 0,
                    "truncated": False,
                    "wallclock_s": 0.0,
                    "error": f"thread: {repr(e)[:200]}",
                }
                traceback.print_exc()
            records.append(rec)
            if len(records) % 10 == 0:
                _write_partial(records)
                print(f"  ... saved partial ({len(records)}/{len(cases)})", flush=True)

    _write_partial(records)
    wall = time.perf_counter() - t0

    summary = _summarize(records, wall, args.workers,
                         str(llm.path), args.max_new_tokens)
    Path(SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("EXP 60 PRIMITIVE-DUMP BASELINE — RESULTS")
    print("=" * 72)
    print(f"  Aggregate accuracy : {summary['n_correct']}/{summary['n_total']} "
          f"({summary['accuracy']:.1%})")
    print(f"  Mean input tokens  : {summary['mean_input_tokens']:.0f}")
    print(f"  Mean output tokens : {summary['mean_output_tokens']:.1f}")
    print(f"  Mean #funcs extracted: {summary['mean_n_functions_extracted']:.1f}")
    print(f"  Mean #funcs in prompt: {summary['mean_n_functions_in_prompt']:.1f}")
    print(f"  Mean chars in prompt : {summary['mean_chars_in_prompt']:.0f}")
    print(f"  Truncation rate    : {summary['n_truncated']}/{summary['n_total']} "
          f"({summary['truncation_rate']:.1%})")
    print(f"  Errors             : {summary['n_errors']}")
    print(f"  Wallclock          : {wall:.1f}s")
    print(f"\n  By repo:")
    for repo, st in sorted(summary["by_repo"].items()):
        print(f"    {repo:<40s} {st['correct']}/{st['total']}  ({st['accuracy']:.0%})")
    print(f"\n  Saved: {RESULTS_PATH}")
    print(f"  Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
