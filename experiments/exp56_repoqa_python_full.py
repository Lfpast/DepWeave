"""RepoQA Python: full DepWeave global navigation plus local evidence.

Official name accuracy is retained. Canonical entity accuracy is reported
separately against the dataset's source path and definition line.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.local_model import LocalQwen3, MODEL_NAME, RESULTS_DIR, add_model_arguments
from benchmarks.repoqa import REPOQA_JSON
from depweave import DepWeaveRunner

RESULTS_PATH = RESULTS_DIR / "depweave_v1_repoqa_python_full.json"
FULLTEXT_CHAR_CAP = 30000
CONFIG_FIELDS = {"A": "config_A_with_cache", "B": "config_B_without_cache",
                 "C": "config_C_full_text"}


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

    def walk(node, owners):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                next_owners = [*owners, child.name]
                if child.lineno == case["line"] and child.name == case["name"]:
                    found.append(".".join(next_owners))
                walk(child, next_owners)
            else:
                walk(child, owners)

    walk(tree, [])
    if len(found) != 1:
        return None
    return f"{snapshot_id}|{path}|{path}::{found[0]}|{case['line']}"


async def run_cases(cases: list[dict], llm, *, max_input_tokens: int = 4096,
                    cache_dir: str | None = None, cache_stats_out: dict | None = None,
                    reuse_local_functions: bool = True, clear_cache: bool = False,
                    on_record=None, label: str | None = None) -> list[dict]:
    records = []
    async with DepWeaveRunner(llm, max_input_tokens=max_input_tokens,
                               cache_dir=cache_dir,
                               reuse_local_functions=reuse_local_functions) as runner:
        if clear_cache and runner.cache is not None:
            runner.cache.clear()
        for i, case in enumerate(cases):
            # RepoQA needles are independent questions; only source/index caches carry over.
            runner.sessions.pop(case["repo"], None)
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
                          "predicted": None, "predicted_id": None,
                          "input_tokens": 0, "output_tokens": 0,
                          "extraction_time": 0.0, "query_time": 0.0,
                          "cache_hit": False, "exact": False,
                          "entity_exact": None, "error": repr(exc)}
            record["wallclock_s"] = time.perf_counter() - started
            records.append(record)
            if on_record is not None:
                on_record(i, record)
            print(f"[{label + ' ' if label else ''}{i + 1}/{len(cases)}] "
                  f"{case['repo']}::{case['name']} "
                  f"name={'OK' if record['exact'] else 'X'} "
                  f"entity={record.get('entity_exact')} "
                  f"tokens={record.get('input_tokens', 0)} "
                  f"error={record.get('error')}")
        if cache_stats_out is not None and runner.cache is not None:
            cache_stats_out.update(runner.cache.stats.to_dict())
    return records


def build_fulltext_prompt(content: dict, description: str) -> tuple[str, int, bool]:
    """The historical Config C prompt and 30,000-character source cap."""
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


def parse_fulltext_answer(completion: str) -> str:
    txt = (completion or "").strip()
    if txt.startswith("```"):
        txt = "\n".join(line for line in txt.splitlines() if not line.startswith("```"))
    match = re.search(r"ANSWER\s*:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?", txt)
    if match:
        return match.group(1)
    match = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", txt)
    if match:
        return match.group(1)
    stop = {"the", "this", "that", "based", "answer", "function", "name", "is",
            "would", "looks", "given", "from", "context", "above", "below",
            "snippet", "code", "python", "repository", "matches", "match"}
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", txt):
        if token.lower() not in stop and len(token) > 3:
            return token
    return ""


def run_fulltext(case: dict, llm) -> dict:
    started = time.perf_counter()
    prompt, chars_in, truncated = build_fulltext_prompt(case["content"], case["description"])
    try:
        completion, tokens_in, tokens_out = llm(prompt)
        prediction = parse_fulltext_answer(completion)
        result = {"predicted": prediction, "exact": prediction == case["name"],
                  "input_tokens": tokens_in, "output_tokens": tokens_out,
                  "chars_in_prompt": chars_in, "truncated": truncated}
    except Exception as exc:
        result = {"predicted": None, "exact": False, "input_tokens": 0,
                  "output_tokens": 0, "chars_in_prompt": chars_in,
                  "truncated": truncated, "error": f"llm: {repr(exc)[:200]}"}
    result["wallclock_s"] = time.perf_counter() - started
    return result


def legacy_config_payload(label: str, result: dict) -> dict:
    """Project a DepWeave result onto the old exp56 per-config JSON fields."""
    if label in ("A", "B"):
        payload = {"predicted": result.get("predicted"),
                   "exact": bool(result.get("exact")),
                   "n_candidates": result.get("n_candidates", result.get("n_local_functions", 0)),
                   "extraction_time": result.get("extraction_time", 0.0),
                   "query_time": result.get("query_time", 0.0),
                   "cache_hit": bool(result.get("cache_hit")),
                   "input_tokens": result.get("input_tokens", 0),
                   "output_tokens": result.get("output_tokens", 0),
                   "wallclock_s": result.get("wallclock_s", 0.0)}
    else:
        payload = {"predicted": result.get("predicted"),
                   "exact": bool(result.get("exact")),
                   "input_tokens": result.get("input_tokens", 0),
                   "output_tokens": result.get("output_tokens", 0),
                   "chars_in_prompt": result.get("chars_in_prompt", 0),
                   "truncated": bool(result.get("truncated")),
                   "wallclock_s": result.get("wallclock_s", 0.0)}
    if result.get("error"):
        payload["error"] = result["error"]
    return payload


def legacy_full_record(record: dict) -> dict:
    """Keep exactly the top-level fields written by exp56_old.py."""
    row = {key: record[key] for key in (
        "question_id", "repo", "needle_function", "expected_function_name",
        "needle_description_preview", "needle_path")}
    for label, field in CONFIG_FIELDS.items():
        result = record.get(field)
        row[field] = legacy_config_payload(label, result) if isinstance(result, dict) else None
    return row


def remove_old_sidecars(output: Path, summary_path: Path) -> None:
    output.with_name(output.stem + "_run_meta.json").unlink(missing_ok=True)
    summary_path.with_name(summary_path.stem.removesuffix("_summary") + "_details.json").unlink(missing_ok=True)


async def run_all_configs(cases: list[dict], llm, *, max_input_tokens: int,
                          cache_dir: str, output: Path, cache_stats_out: dict) -> list[dict]:
    """Execute every old configuration, with DepWeave replacing CardMem in A/B."""
    records = [{"question_id": f"{case['repo'].split('/')[-1]}::{case['name']}",
                "repo": case["repo"], "needle_function": case["name"],
                "expected_function_name": case["name"],
                "needle_description_preview": (case["description"] or "")[:160],
                "needle_path": case["path"],
                **{field: None for field in CONFIG_FIELDS.values()}}
               for i, case in enumerate(cases)]
    completed = 0

    def write_records() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(records, key=lambda row: row["question_id"])
        output.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    def record_result(i: int, label: str, result: dict) -> None:
        nonlocal completed
        records[i][CONFIG_FIELDS[label]] = legacy_config_payload(label, result)
        completed += 1
        if completed % 5 == 0 or completed == 3 * len(cases):
            write_records()

    await run_cases(cases, llm, max_input_tokens=max_input_tokens, cache_dir=cache_dir,
                    cache_stats_out=cache_stats_out, reuse_local_functions=False,
                    clear_cache=True, on_record=lambda i, r: record_result(i, "A", r),
                    label="A")
    write_records()
    await run_cases(cases, llm, max_input_tokens=max_input_tokens,
                    reuse_local_functions=False,
                    on_record=lambda i, r: record_result(i, "B", r), label="B")
    write_records()
    for i, case in enumerate(cases):
        result = run_fulltext(case, llm)
        record_result(i, "C", result)
        print(f"[C {i + 1}/{len(cases)}] {case['repo']}::{case['name']} "
              f"name={'OK' if result['exact'] else 'X'} error={result.get('error')}")
    if not cases:
        write_records()
    return records


def summarize(records: list[dict], max_input_tokens: int = 4096,
              *, model_path: str | None = None,
              max_new_tokens: int | None = None, run_meta: dict | None = None) -> dict:
    """Compute the historical A/B/C schema from three measured configurations."""
    run_meta = run_meta or {}
    n = len(records)
    if any(not isinstance(r.get(field), dict) for r in records
           for field in CONFIG_FIELDS.values()):
        raise ValueError("RepoQA raw results lack a measured A, B or C configuration; "
                         "rerun the fixed script to obtain a comparable summary")
    repo_distribution = dict(Counter(r["repo"] for r in records))
    if "repo_distribution" in run_meta:
        if dict(run_meta["repo_distribution"]) != repo_distribution:
            raise ValueError("Summary repo distribution does not match full.json")
        repo_distribution = run_meta["repo_distribution"]
    configs = {}
    for label, field in CONFIG_FIELDS.items():
        rows = [r[field] for r in records]
        wall = [float(r.get("wallclock_s", 0) or 0) for r in rows]
        measured = {
            "n_correct": sum(bool(r.get("exact")) for r in rows),
            "n_total": n,
            "accuracy": sum(bool(r.get("exact")) for r in rows) / n if n else 0.0,
            "mean_input_tokens": sum(int(r.get("input_tokens", 0) or 0) for r in rows) / n if n else 0.0,
            "mean_output_tokens": sum(int(r.get("output_tokens", 0) or 0) for r in rows) / n if n else 0.0,
            "mean_wallclock_s": sum(wall) / n if n else 0.0,
            "sum_wallclock_s": sum(wall),
            "n_errors": sum(bool(r.get("error")) for r in rows),
        }
        if label == "A":
            hit_walls = [r.get("wallclock_s", 0) for r in rows if r.get("cache_hit")]
            miss_walls = [r.get("wallclock_s", 0) for r in rows if not r.get("cache_hit")]
            measured.update({"cache_hits": len(hit_walls),
                             "cache_hit_rate": len(hit_walls) / n if n else 0.0,
                             "mean_wallclock_hit_s": sum(hit_walls) / len(hit_walls) if hit_walls else 0.0,
                             "mean_wallclock_miss_s": sum(miss_walls) / len(miss_walls) if miss_walls else 0.0})
        if label == "C":
            truncated = sum(bool(r.get("truncated")) for r in rows)
            measured.update({"truncated_count": truncated,
                             "truncation_rate": truncated / n if n else 0.0})
        configs[label] = measured
    a_mean = configs["A"]["mean_wallclock_s"]
    b_mean = configs["B"]["mean_wallclock_s"]
    hit_mean = configs["A"]["mean_wallclock_hit_s"]
    return {
        "model": MODEL_NAME, "backend": "local_transformers",
        "model_path": model_path, "max_new_tokens": max_new_tokens,
        "workers": 1, "n_needles": n,
        "n_repos": len(repo_distribution),
        "repo_distribution": repo_distribution,
        "total_wallclock_s": run_meta.get("suite_wallclock_s", sum(
            configs[label]["sum_wallclock_s"] for label in CONFIG_FIELDS)),
        "fulltext_char_cap": FULLTEXT_CHAR_CAP,
        "configs": configs,
        "speedup_cache_vs_nocache": b_mean / max(a_mean, 1e-6),
        "speedup_cache_hit_vs_nocache": b_mean / max(hit_mean, 1e-6),
        "pixelmem_cache_stats": run_meta.get("pixelmem_cache_stats"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, max_new_tokens=512)
    parser.add_argument("--limit", type=int, default=0, help="0 runs all Python needles")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--cache-dir", default="/tmp/depweave_repoqa_local_cache")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
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
        run_info = {"suite_wallclock_s": previous["total_wallclock_s"],
                    "pixelmem_cache_stats": previous["pixelmem_cache_stats"],
                    "repo_distribution": previous["repo_distribution"]}
        model_path = previous["model_path"]
        max_new_tokens = previous["max_new_tokens"]
    else:
        if not args.cache_dir:
            parser.error("--cache-dir must name a directory for Config A")
        cases = load_cases()
        if args.limit > 0:
            cases = cases[:args.limit]
        llm = LocalQwen3(args.model_path, max_new_tokens=args.max_new_tokens)
        model_path = str(llm.path)
        max_new_tokens = args.max_new_tokens
        cache_stats = {}
        started = time.perf_counter()
        records = asyncio.run(run_all_configs(cases, llm,
                                              max_input_tokens=args.max_input_tokens,
                                              cache_dir=args.cache_dir, output=args.output,
                                              cache_stats_out=cache_stats))
        run_info = {"suite_wallclock_s": time.perf_counter() - started,
                    "pixelmem_cache_stats": cache_stats}
    if not isinstance(records, list):
        parser.error("--output must contain a list of RepoQA records")
    if any(not isinstance(r, dict) or "repo" not in r or
           any(not isinstance(r.get(field), dict) for field in CONFIG_FIELDS.values())
           for r in records):
        parser.error("--output must contain completed RepoQA A, B and C records")
    summary = summarize(records, args.max_input_tokens,
                        model_path=model_path, max_new_tokens=max_new_tokens,
                        run_meta=run_info)
    if args.summary_only:
        for label in CONFIG_FIELDS:
            old = previous["configs"][label]
            new = summary["configs"][label]
            if (old["n_total"] != new["n_total"] or old["n_correct"] != new["n_correct"]
                    or abs(old["sum_wallclock_s"] - new["sum_wallclock_s"]) > 1e-6):
                parser.error("Existing summary does not match this full.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    remove_old_sidecars(args.output, summary_path)
    print(json.dumps(summary, indent=2))
    failures = sum(summary["configs"][label]["n_errors"] for label in CONFIG_FIELDS)
    if failures:
        raise SystemExit(f"{failures} RepoQA configuration runs failed; see {args.output}")


if __name__ == "__main__":
    main()
