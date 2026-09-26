"""Exp 56 — RepoQA Python full-scale benchmark for the CardMem paper.

Three system configurations across all Python needles in
RepoQA-2024-06-23:

  Config A — CardMem WITH PixelMemCache    (V5 cache-on, headline)
  Config B — CardMem WITHOUT cache          (V5 cache-off ablation)
  Config C — full-text baseline             (concat .py files, no card)

All three use ``gpt-4o-mini``. Aggressive parallelism (workers=15) is
safe at OpenAI Tier 4 (10K RPM / 10M TPM).

The cache layer serializes per-document-set, so concurrent Config-A
queries against the same repo will produce 1 miss + (n-1) hits. We
log per-query cache_hit by probing the cache entry_dir before the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quill import TaskCard
from quill.cache import PixelMemCache
from quill.cache.pixel_cache import CachedExtractor
from quill.core.pipeline import V5Pipeline
from quill.core.plugins import Extractor, PluginSet
from quill.core.types import Primitive
from quill.harness import default_input_adapter

from experiments.exp22_v5_haiku_smoke import openai_4omini
from experiments.exp37_v5_repoqa import (
    RepoQAFunctionExtractor, RepoQASearchPrompt, REPOQA_JSON,
)


CACHE_DIR = "/tmp/v5_repoqa_cache_full"
RESULTS_PATH = "results/exp56_repoqa_python_full.json"
SUMMARY_PATH = "results/exp56_repoqa_python_full_summary.json"
FULLTEXT_CHAR_CAP = 30000


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


def _hash_documents(documents: dict[str, str]) -> str:
    """Mirror PixelMemCache._hash_documents so we can probe entry_dir for hits."""
    h = hashlib.sha256()
    for key in sorted(documents):
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        value = documents[key]
        if not isinstance(value, str):
            value = str(value)
        h.update(value.encode("utf-8", errors="replace"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


class _ProbeCachedExtractor(Extractor):
    """Wraps PixelMemCache so each extract() call records hit-or-miss.

    Workaround for a v5 cache mismatch where the hit check looks for
    ``provenance.json`` but the save path writes ``provenance.json.gz``
    — meaning ``cache.get_or_extract`` always reports a miss. Here we
    duplicate the cache's hit-check logic against the actual filename
    that gets written. We still call into the underlying cache for
    save/load semantics; on a hit (sidecar already on disk) we read it
    directly to bypass the broken hit branch.
    """

    def __init__(self, inner: Extractor, cache: PixelMemCache) -> None:
        self._inner = inner
        self._cache = cache
        self.last_hit: bool = False

    def extract(self, documents: dict[str, str], **kwargs) -> list[Primitive]:
        key = _hash_documents(documents)
        entry_dir = Path(self._cache.root) / key
        # Acquire the cache's per-key lock so concurrent threads on the
        # same key serialize properly (mirrors what get_or_extract does).
        lock = self._cache._lock_for(key)
        with lock:
            sidecar_gz = entry_dir / "provenance.json.gz"
            index_json = entry_dir / "index.json"
            if index_json.exists() and sidecar_gz.exists():
                # Hit: load from sidecar directly (the v5 load path also
                # works because it accepts the .gz file).
                t0 = time.perf_counter()
                primitives = self._cache._load_from_pixels(entry_dir)
                with self._cache._stats_lock:
                    self._cache.stats.total_wallclock_load_s += time.perf_counter() - t0
                    self._cache.stats.hits += 1
                self.last_hit = True
                return primitives
            # Miss
            self.last_hit = False
            with self._cache._stats_lock:
                self._cache.stats.misses += 1
            t0 = time.perf_counter()
            primitives = list(self._inner.extract(documents, **kwargs))
            extract_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            self._cache._save_to_pixels(entry_dir, primitives)
            save_s = time.perf_counter() - t1
            from quill.cache.pixel_cache import _dir_size_bytes
            size = _dir_size_bytes(entry_dir)
            with self._cache._stats_lock:
                self._cache.stats.total_wallclock_extract_s += extract_s
                self._cache.stats.total_wallclock_save_s += save_s
                self._cache.stats.bytes_written += size
                self._cache.stats.primitives_cached += len(primitives)
                self._cache.stats.cache_entries += 1
                self._cache.stats.entry_sizes.append(size)
            return primitives


# ---------------------------------------------------------------------------
# CardMem (Config A and B) runner
# ---------------------------------------------------------------------------


def _build_card(qid: str, content: dict, description: str, repo: str,
                expected: str) -> TaskCard:
    """Single-question TaskCard so the V5 Pipeline can run one needle."""
    return TaskCard.from_dict({
        "domain": "repoqa_search_needle_function",
        "description": "RepoQA SNF: pick the function that matches the NL description.",
        "input_schema": {"kind": "doc_set"},
        "query": {"kind": "function_selection", "output": "string"},
        "eval": {"metric": "exact_match", "threshold": 0.5},
        "few_shot": [],
        "holdout": [{
            "qid": qid,
            "input": {
                "documents": content,
                "query": {"description": description, "repo": repo},
            },
            "expected_output": expected,
        }],
    })


def _run_cardmem(case: dict, llm, cache: PixelMemCache | None) -> dict:
    """Run one needle through the V5 RepoQA pipeline. If `cache` is given,
    wraps the extractor with it (Config A). Otherwise, plain extractor (B).
    """
    qid = f"{case['repo'].split('/')[-1]}::{case['needle_name']}"
    expected = case["needle_name"]
    content = case["content"]
    description = case["needle_description"]
    repo = case["repo"]

    inner_extractor = RepoQAFunctionExtractor()
    probe: _ProbeCachedExtractor | None = None
    if cache is not None:
        # _ProbeCachedExtractor records this thread's hit/miss as the
        # cache itself decides it (under the per-key lock).
        probe = _ProbeCachedExtractor(inner_extractor, cache)
        extractor = probe
    else:
        extractor = inner_extractor

    plugins = PluginSet(
        name=f"repoqa_{'cached' if cache else 'nocache'}",
        extractor=extractor,
        prompt_template=RepoQASearchPrompt(),
        derivation_rules=[],
    )
    card = _build_card(qid, content, description, repo, expected)
    ex = card.spec.holdout[0]

    # Time extraction vs query (the LLM call). Since V5Pipeline.index() is
    # called inside .run(), we instrument by calling them separately.
    pipe = V5Pipeline(plugins, card.spec, llm)
    docs, qi = default_input_adapter(ex)

    t_extract0 = time.perf_counter()
    try:
        pipe.index(docs)
    except Exception as e:
        return {
            "predicted": None, "exact": False, "n_candidates": 0,
            "extraction_time": time.perf_counter() - t_extract0,
            "query_time": 0.0,
            "cache_hit": (probe.last_hit if probe else False),
            "input_tokens": 0, "output_tokens": 0,
            "error": f"extract: {repr(e)[:200]}",
        }
    extract_time = time.perf_counter() - t_extract0
    cache_hit = probe.last_hit if probe else False

    t_query0 = time.perf_counter()
    try:
        pred, stats = pipe.run(qi, documents=docs, reuse_index=True)
    except Exception as e:
        return {
            "predicted": None, "exact": False,
            "n_candidates": len(pipe.primitives),
            "extraction_time": extract_time,
            "query_time": time.perf_counter() - t_query0,
            "cache_hit": cache_hit,
            "input_tokens": 0, "output_tokens": 0,
            "error": f"run: {repr(e)[:200]}",
        }
    query_time = time.perf_counter() - t_query0

    pred_str = (str(pred).strip() if pred is not None else "")
    return {
        "predicted": pred_str,
        "exact": pred_str == expected,
        "n_candidates": stats.n_primitives,
        "extraction_time": extract_time,
        "query_time": query_time,
        "cache_hit": cache_hit,
        "input_tokens": stats.tokens_in,
        "output_tokens": stats.tokens_out,
    }


# ---------------------------------------------------------------------------
# Full-text baseline (Config C)
# ---------------------------------------------------------------------------


def _build_fulltext_prompt(content: dict, description: str) -> tuple[str, int, bool]:
    parts = []
    total = 0
    truncated = False
    for path in sorted(content):
        if not path.endswith(".py"):
            continue
        body = content[path] if isinstance(content[path], str) else str(content[path])
        block = f"# === {path} ===\n{body}\n"
        if total + len(block) > FULLTEXT_CHAR_CAP:
            remaining = FULLTEXT_CHAR_CAP - total
            if remaining > 0:
                parts.append(block[:remaining])
                total += remaining
            truncated = True
            break
        parts.append(block)
        total += len(block)
    body_blob = "".join(parts)
    prompt = (
        "You are given a Python repository and a natural-language "
        "description of one function in it (with the function name "
        "obfuscated in the description). Pick the function whose "
        "behavior matches.\n\n"
        f"DESCRIPTION:\n{description[:2000]}\n\n"
        f"REPOSITORY (Python source files concatenated"
        f"{'; TRUNCATED at ' + str(FULLTEXT_CHAR_CAP) + ' chars' if truncated else ''}):\n"
        f"{body_blob}\n\n"
        "Output format: a single line with EXACTLY this format (no other "
        "text, no code fence, no explanation):\n"
        "ANSWER: <function_name>\n"
        "where <function_name> is the bare Python identifier."
    )
    return prompt, total, truncated


def _parse_fulltext_answer(completion: str) -> str:
    import re
    txt = (completion or "").strip()
    if txt.startswith("```"):
        txt = "\n".join(l for l in txt.splitlines() if not l.startswith("```")).strip()
    # Prefer the explicit ANSWER: line.
    m = re.search(r"ANSWER\s*:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?", txt)
    if m:
        return m.group(1)
    # Fallback: first backticked identifier.
    m = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", txt)
    if m:
        return m.group(1)
    # Last fallback: first identifier longer than 3 chars that isn't a stop word.
    stop = {"the", "this", "that", "based", "answer", "function", "name", "is",
            "would", "looks", "given", "from", "context", "above", "below",
            "snippet", "code", "python", "repository", "matches", "match"}
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", txt):
        if tok.lower() not in stop and len(tok) > 3:
            return tok
    return ""


def _run_fulltext(case: dict, llm) -> dict:
    expected = case["needle_name"]
    prompt, chars_in, truncated = _build_fulltext_prompt(
        case["content"], case["needle_description"],
    )
    try:
        completion, t_in, t_out = llm(prompt)
    except Exception as e:
        return {
            "predicted": None, "exact": False,
            "input_tokens": 0, "output_tokens": 0,
            "chars_in_prompt": chars_in, "truncated": truncated,
            "wallclock_s": 0.0,
            "error": f"llm: {repr(e)[:200]}",
        }
    pred = _parse_fulltext_answer(completion)
    return {
        "predicted": pred,
        "exact": pred == expected,
        "input_tokens": t_in,
        "output_tokens": t_out,
        "chars_in_prompt": chars_in,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Orchestration: per-needle (A, B, C) with shared thread pool
# ---------------------------------------------------------------------------


def _qid(case: dict) -> str:
    return f"{case['repo'].split('/')[-1]}::{case['needle_name']}"


def _run_one_config(label: str, case: dict, llm, cache):
    """Run one (config, needle) pair. Returns (qid, label, payload, wall_s)."""
    t0 = time.perf_counter()
    if label == "A":
        payload = _run_cardmem(case, llm, cache)
    elif label == "B":
        payload = _run_cardmem(case, llm, None)
    else:  # C
        payload = _run_fulltext(case, llm)
    payload["wallclock_s"] = time.perf_counter() - t0
    return _qid(case), label, payload


def _write_partial(records: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Stable order: repo then needle name
    ordered = sorted(records.values(), key=lambda r: r["question_id"])
    with open(path, "w") as f:
        json.dump(ordered, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0,
                    help="limit number of needles (0 = all)")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--output", default=RESULTS_PATH)
    ap.add_argument("--summary", default=SUMMARY_PATH)
    args = ap.parse_args()

    def llm(p):
        return openai_4omini(p, model=args.model)

    needles = _load_python_needles()
    if args.limit > 0:
        needles = needles[: args.limit]

    repo_dist: dict[str, int] = {}
    for n in needles:
        repo_dist[n["repo"]] = repo_dist.get(n["repo"], 0) + 1
    print(f"=== Exp 56: RepoQA Python full ({args.model}, workers={args.workers}) ===")
    print(f"  n_needles={len(needles)} across {len(repo_dist)} repos:")
    for r, c in sorted(repo_dist.items()):
        print(f"    {r}: {c} needles")

    # Initialize cache for Config A — clean slate so the run is reproducible.
    cache = PixelMemCache(args.cache_dir)
    cache.clear()

    # Build per-needle record skeletons.
    records: dict[str, dict] = {}
    for case in needles:
        qid = _qid(case)
        records[qid] = {
            "question_id": qid,
            "repo": case["repo"],
            "needle_function": case["needle_name"],
            "expected_function_name": case["needle_name"],
            "needle_description_preview": (case["needle_description"] or "")[:160],
            "needle_path": case["needle_path"],
            "config_A_with_cache": None,
            "config_B_without_cache": None,
            "config_C_full_text": None,
        }

    # Submit all (case, label) pairs into a single thread pool. Order:
    # interleave so within Config A, repo-grouped jobs spread out and
    # the per-key cache lock serializes them naturally (first miss, rest hit).
    tasks: list[tuple[str, dict]] = []
    for label in ("A", "B", "C"):
        for case in needles:
            tasks.append((label, case))

    label_field = {
        "A": "config_A_with_cache",
        "B": "config_B_without_cache",
        "C": "config_C_full_text",
    }

    print(f"\nRunning {len(tasks)} task units (workers={args.workers}) ...")
    t0_suite = time.perf_counter()
    completed = 0
    last_save = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_one_config, label, case, llm, cache): (label, case)
            for (label, case) in tasks
        }
        for f in as_completed(futures):
            try:
                qid, label, payload = f.result()
            except Exception as e:
                label, case = futures[f]
                qid = _qid(case)
                payload = {
                    "predicted": None, "exact": False, "wallclock_s": 0.0,
                    "error": f"thread: {repr(e)[:200]}",
                }
            records[qid][label_field[label]] = payload
            completed += 1
            if completed % 5 == 0 or completed == len(tasks):
                _write_partial(records, args.output)
                print(f"  [{completed}/{len(tasks)}] saved partial "
                      f"({time.perf_counter()-t0_suite:.1f}s elapsed)")
                last_save = completed

    if last_save != len(tasks):
        _write_partial(records, args.output)

    total_wall = time.perf_counter() - t0_suite

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n = len(records)
    summary: dict[str, Any] = {
        "model": args.model,
        "workers": args.workers,
        "n_needles": n,
        "n_repos": len(repo_dist),
        "repo_distribution": repo_dist,
        "total_wallclock_s": total_wall,
        "fulltext_char_cap": FULLTEXT_CHAR_CAP,
        "configs": {},
    }

    for cfg_key, label in [("A", "config_A_with_cache"),
                            ("B", "config_B_without_cache"),
                            ("C", "config_C_full_text")]:
        n_correct = 0
        sum_in = 0
        sum_out = 0
        sum_wall = 0.0
        n_with_data = 0
        n_errors = 0
        for r in records.values():
            p = r[label]
            if p is None:
                continue
            n_with_data += 1
            if p.get("error"):
                n_errors += 1
            if p.get("exact"):
                n_correct += 1
            sum_in += int(p.get("input_tokens", 0) or 0)
            sum_out += int(p.get("output_tokens", 0) or 0)
            sum_wall += float(p.get("wallclock_s", 0.0) or 0.0)
        cfg_summary = {
            "n_correct": n_correct,
            "n_total": n_with_data,
            "accuracy": n_correct / max(n_with_data, 1),
            "mean_input_tokens": sum_in / max(n_with_data, 1),
            "mean_output_tokens": sum_out / max(n_with_data, 1),
            "mean_wallclock_s": sum_wall / max(n_with_data, 1),
            "sum_wallclock_s": sum_wall,
            "n_errors": n_errors,
        }
        if cfg_key == "A":
            n_hits = sum(
                1 for r in records.values()
                if r[label] and r[label].get("cache_hit")
            )
            hit_walls = [
                float(r[label].get("wallclock_s", 0.0) or 0.0)
                for r in records.values()
                if r[label] and r[label].get("cache_hit")
            ]
            miss_walls = [
                float(r[label].get("wallclock_s", 0.0) or 0.0)
                for r in records.values()
                if r[label] and not r[label].get("cache_hit")
            ]
            cfg_summary["cache_hits"] = n_hits
            cfg_summary["cache_hit_rate"] = n_hits / max(n_with_data, 1)
            cfg_summary["mean_wallclock_hit_s"] = (
                sum(hit_walls) / len(hit_walls) if hit_walls else 0.0
            )
            cfg_summary["mean_wallclock_miss_s"] = (
                sum(miss_walls) / len(miss_walls) if miss_walls else 0.0
            )
        if cfg_key == "C":
            n_truncated = sum(
                1 for r in records.values()
                if r[label] and r[label].get("truncated")
            )
            cfg_summary["truncated_count"] = n_truncated
            cfg_summary["truncation_rate"] = n_truncated / max(n_with_data, 1)
        summary["configs"][cfg_key] = cfg_summary

    # Speedup: cache vs no-cache (mean wallclock A vs B).
    a_mean = summary["configs"]["A"]["mean_wallclock_s"]
    b_mean = summary["configs"]["B"]["mean_wallclock_s"]
    a_hit_mean = summary["configs"]["A"]["mean_wallclock_hit_s"]
    summary["speedup_cache_vs_nocache"] = b_mean / max(a_mean, 1e-6)
    summary["speedup_cache_hit_vs_nocache"] = b_mean / max(a_hit_mean, 1e-6)

    summary["pixelmem_cache_stats"] = cache.stats.to_dict()

    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Pretty print
    print("\n=== SUMMARY ===")
    for k in ("A", "B", "C"):
        c = summary["configs"][k]
        print(f"  Config {k}: accuracy={c['n_correct']}/{c['n_total']}={c['accuracy']:.1%}  "
              f"in_tok={c['mean_input_tokens']:.0f}  "
              f"out_tok={c['mean_output_tokens']:.0f}  "
              f"mean_wall={c['mean_wallclock_s']:.2f}s  "
              f"errs={c['n_errors']}")
    a = summary["configs"]["A"]
    print(f"  Cache hit rate (A): {a['cache_hits']}/{a['n_total']} = {a['cache_hit_rate']:.1%}")
    cT = summary["configs"]["C"]
    print(f"  Truncation rate (C): {cT['truncated_count']}/{cT['n_total']} = "
          f"{cT['truncation_rate']:.1%}")
    print(f"  Speedup A-vs-B (mean wall, cache vs no-cache): "
          f"{summary['speedup_cache_vs_nocache']:.2f}x")
    print(f"  Speedup A-hit-vs-B (cache-hit-only wall vs no-cache): "
          f"{summary['speedup_cache_hit_vs_nocache']:.2f}x  "
          f"(A_hit={a_hit_mean:.2f}s vs B={b_mean:.2f}s)")
    print(f"  Total wallclock: {total_wall:.1f}s")
    print(f"  Cache: {cache.stats.to_dict()}")
    print(f"\nWrote per-question records to {args.output}")
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
