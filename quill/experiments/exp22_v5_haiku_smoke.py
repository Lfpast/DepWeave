"""Exp 22 — V5 LLM smoke test.

Purpose: prove that PixelMem V5 actually runs end-to-end on three different
benchmark *shapes*, using a parallel LLM backend (OpenAI 4o-mini by
default, or the haiku CLI with ``--backend haiku``).

This is NOT a full benchmark — 5 questions per task card, synthetic where
real data isn't cached locally. The goal is to verify:

  1. The V4-wrapping plugin set answers a python-deps task correctly.
  2. The synthesis path produces a runnable pipeline for a non-code domain
     (here: a simple conversational-fact retrieval task).
  3. Mixed-relation derivation (here: a mini "who-mentored-whom" KG QA).

Parallelism: ThreadPoolExecutor per task card — matches the feedback
memory's "always parallelize benchmark LLM calls" rule. The OpenAI backend
handles 5+ concurrent calls without contention; the haiku CLI backend is
capped at ~2 concurrent calls (local auth-file lock contention).

Runtime: ~15 LLM calls total; completes in well under a minute on OpenAI,
around a minute on haiku CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quill import TaskCard
from quill.core.plugins import PluginSet
from quill.plugins.python_deps import build_python_deps_plugins


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


def openai_4omini(
    prompt: str,
    model: str = "gpt-4o-mini",
    timeout: int = 60,
    max_tokens: int = 1500,
) -> tuple[str, int, int]:
    """Call OpenAI 4o-mini via the openai SDK. Returns (completion, tokens_in, tokens_out).

    ``max_tokens=1500`` is enough for V5 synthesis responses (schema, rule
    lists, prompt templates) as well as short-answer outputs. The old cap of
    200 truncated schema-design JSON mid-object.

    Reads the API key from ``OPENAI_API_KEY``. If the env var isn't set,
    raises. (The old fallback to a hardcoded project key has been removed.)
    """
    from openai import OpenAI
    import os
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Export it before running V5 experiments."
        )
    client = OpenAI(api_key=key, timeout=timeout)
    r = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt[:12000]}],
    )
    out = (r.choices[0].message.content or "").strip()
    return out, r.usage.prompt_tokens, r.usage.completion_tokens


def haiku_cli(prompt: str, model: str = "haiku", timeout: int = 180) -> tuple[str, int, int]:
    """Call the claude CLI. Returns (completion, tokens_in, tokens_out).

    Flags:

    - ``--max-turns 1``: non-negotiable. Without it the CLI enters agent
      mode and tries Bash/Read/Edit on filenames mentioned in the prompt.
    - ``--bare``: skip hooks, plugin sync, keychain reads, auto-memory, and
      CLAUDE.md auto-discovery. These are the init steps that serialize
      parallel invocations (auth file contention) and balloon wall time.

    Token counts are word-count approximations since ``-p`` does not report
    usage in its non-interactive output. Fine for a smoke test.
    """
    try:
        r = subprocess.run(
            ["claude", "-p", prompt, "--model", model, "--max-turns", "1"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (r.stdout or "").strip()
        if not out:
            out = "__EMPTY__"
        return out, len(prompt.split()), len(out.split())
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__", len(prompt.split()), 0
    except FileNotFoundError:
        raise RuntimeError(
            "claude CLI not found in PATH — install it or pass a different llm backend"
        )


# ---------------------------------------------------------------------------
# Task card 1: python-deps (V4 wrapper path)
# ---------------------------------------------------------------------------


def _py_deps_card() -> TaskCard:
    def case(files: dict, order: list[str], qid: str) -> dict:
        return {
            "qid": qid,
            "input": {"files": list(files.keys()), "file_contents": files},
            "expected_output": order,
        }

    return TaskCard.from_dict({
        "domain": "python_deps_smoke",
        "description": "Order 3 python files by dependency.",
        "input_schema": {"kind": "file_set"},
        "query": {"kind": "ordering"},
        "eval": {"metric": "exact_match", "threshold": 0.6},
        "few_shot": [],
        "holdout": [
            case(
                {
                    "base.py": "class Base:\n    pass\n",
                    "model.py": "from base import Base\nclass Model(Base):\n    pass\n",
                    "main.py": "from model import Model\ndef run(): return Model()\n",
                },
                ["base.py", "model.py", "main.py"],
                "pd1",
            ),
            case(
                {
                    "config.py": "DEBUG = True\n",
                    "logger.py": "from config import DEBUG\nclass L:\n    pass\n",
                    "app.py": "from logger import L\ndef go(): return L()\n",
                },
                ["config.py", "logger.py", "app.py"],
                "pd2",
            ),
            case(
                {
                    "util.py": "def add(a,b): return a+b\n",
                    "service.py": "from util import add\nclass S:\n    def do(self,x,y): return add(x,y)\n",
                    "cli.py": "from service import S\ndef main(): print(S().do(1,2))\n",
                },
                ["util.py", "service.py", "cli.py"],
                "pd3",
            ),
            case(
                {
                    "token.py": "class T: pass\n",
                    "parser.py": "from token import T\nclass P:\n    def parse(self): return T()\n",
                    "engine.py": "from parser import P\ndef run(): return P().parse()\n",
                },
                ["token.py", "parser.py", "engine.py"],
                "pd4",
            ),
            case(
                {
                    "db.py": "class DB: pass\n",
                    "repo.py": "from db import DB\nclass R:\n    def __init__(self): self.db = DB()\n",
                    "api.py": "from repo import R\ndef get_r(): return R()\n",
                },
                ["db.py", "repo.py", "api.py"],
                "pd5",
            ),
        ],
    })


# ---------------------------------------------------------------------------
# Task card 2: conversational-fact retrieval (synthesis path)
# ---------------------------------------------------------------------------


def _convo_card() -> TaskCard:
    """A tiny LongMemEval-shaped task: sessions + a question, single-fact answer.

    V5 synthesis would design a schema like relation=mentions/lives_in/works_at.
    For the smoke test we hand-assemble the plugin set (V5 template extractor)
    so we don't need a live schema-design LLM call.
    """
    def case(sessions: list[str], q: str, a: str, qid: str) -> dict:
        docs = {f"s{i}": text for i, text in enumerate(sessions)}
        return {
            "qid": qid,
            "input": {"documents": docs, "query": {"question": q}},
            "expected_output": a,
        }

    return TaskCard.from_dict({
        "domain": "convo_fact_smoke",
        "description": "Answer a single-fact question from a small set of prior chat sessions.",
        "input_schema": {"kind": "doc_set"},
        "query": {"kind": "qa", "output": "string"},
        "eval": {"metric": "fuzzy", "threshold": 0.6},
        "few_shot": [],
        "holdout": [
            case(
                [
                    "User: I moved to Denver last month. Assistant: cool, how's the altitude?",
                    "User: finally found a good coffee shop near work. Assistant: nice.",
                ],
                "Where does the user live?",
                "Denver",
                "cv1",
            ),
            case(
                [
                    "User: I started my job at Acme Corp this week. Assistant: exciting.",
                    "User: the commute is painful though. Assistant: traffic?",
                ],
                "What company does the user work at?",
                "Acme Corp",
                "cv2",
            ),
            case(
                [
                    "User: my cat Whiskers keeps knocking things off the shelf.",
                    "User: Whiskers is being especially dramatic tonight.",
                ],
                "What is the user's cat named?",
                "Whiskers",
                "cv3",
            ),
            case(
                [
                    "User: I'm training for the Chicago marathon in October.",
                    "User: did a 15-mile run today, legs are dead.",
                ],
                "Which marathon is the user training for?",
                "Chicago",
                "cv4",
            ),
            case(
                [
                    "User: learning Rust has been rough but rewarding.",
                    "User: the borrow checker finally clicked yesterday.",
                ],
                "Which programming language is the user learning?",
                "Rust",
                "cv5",
            ),
        ],
    })


# A hand-assembled plugin set for the convo task — same shape the synthesis
# loop would produce, but deterministic so the smoke test does not depend
# on a live schema-design LLM call.

def _convo_plugins() -> PluginSet:
    from quill.synth.extractor_synth import (
        ExtractionPattern, TemplateExtractor,
    )
    from quill.synth.prompt_synth import SynthesizedPrompt

    patterns = [
        ExtractionPattern(
            name="mentions_entity",
            regex=r"(?P<s>User|Assistant):\s*(?P<o>[^\n]+)",
            relation="mentions",
            condition="turn",
            field_mapping={"subject": "s", "object": "o"},
        ),
    ]
    extractor = TemplateExtractor(patterns)
    prompt = SynthesizedPrompt(
        header=(
            "You are answering a single-fact question from prior chat "
            "turns. Use only the facts in Raw primitives."
        ),
        instruction=(
            'Return the answer as a JSON array with one element, e.g. ["Denver"].'
        ),
        output_format="json_array",
        token_budget=400,
    )
    return PluginSet(
        name="convo_fact_smoke",
        extractor=extractor,
        prompt_template=prompt,
        derivation_rules=[],  # no chains for this simple task
    )


# ---------------------------------------------------------------------------
# Task card 3: small mentorship KG (multi-hop; tests derivation chains)
# ---------------------------------------------------------------------------


def _kg_card() -> TaskCard:
    """A toy KG: each "document" is one fact line; questions are 2-hop lookups
    like "who mentored X's mentor?" — designed to exercise derivation chains.
    """
    def case(facts: list[str], q: str, a: str, qid: str) -> dict:
        docs = {f"f{i}": f for i, f in enumerate(facts)}
        return {
            "qid": qid,
            "input": {"documents": docs, "query": {"question": q}},
            "expected_output": a,
        }

    return TaskCard.from_dict({
        "domain": "mentor_kg_smoke",
        "description": "Answer 2-hop mentorship queries over a small fact set.",
        "input_schema": {"kind": "fact_list"},
        "query": {"kind": "qa", "output": "string"},
        "eval": {"metric": "fuzzy", "threshold": 0.6},
        "few_shot": [],
        "holdout": [
            case(
                ["Alice mentored Bob.", "Bob mentored Carol.",
                 "Dan mentored Eve.", "Frank mentored Gina."],
                "Who is Carol's mentor's mentor?", "Alice", "kg1",
            ),
            case(
                ["Harry mentored Ingrid.", "Ingrid mentored Jun.",
                 "Kevin mentored Lila.", "Maya mentored Noah."],
                "Who did Jun's mentor learn under?", "Harry", "kg2",
            ),
            case(
                ["Olive mentored Pablo.", "Pablo mentored Quinn.",
                 "Rita mentored Sam.", "Tara mentored Uri."],
                "Who taught Quinn's teacher?", "Olive", "kg3",
            ),
            case(
                ["Vera mentored Will.", "Will mentored Xena.",
                 "Yale mentored Zeb.", "Abe mentored Bea."],
                "Who mentored Xena's mentor?", "Vera", "kg4",
            ),
            case(
                ["Cass mentored Dee.", "Dee mentored Ed.",
                 "Flo mentored Gus.", "Hal mentored Ivy."],
                "Who mentored Ed's mentor?", "Cass", "kg5",
            ),
        ],
    })


def _kg_plugins() -> PluginSet:
    from quill.core.plugins import DerivationRule
    from quill.synth.extractor_synth import ExtractionPattern, TemplateExtractor
    from quill.synth.prompt_synth import SynthesizedPrompt

    patterns = [
        ExtractionPattern(
            name="mentored",
            regex=r"(?P<s>[A-Z][a-z]+)\s+mentored\s+(?P<o>[A-Z][a-z]+)",
            relation="mentored",
            condition="fact",
            field_mapping={"subject": "s", "object": "o"},
        ),
    ]
    rule_two_hop = DerivationRule(
        name="two_hop_mentor",
        pattern=[
            ("?A", "mentored", "?B", "?c1"),
            ("?B", "mentored", "?C", "?c2"),
        ],
        derived=("?A", "mentored_mentor_of", "?C", "derived"),
        confidence=0.9,
    )
    prompt = SynthesizedPrompt(
        header=(
            "You answer a 2-hop mentorship question. Use the Confirmed facts "
            "(derived chains) first; only look at Raw primitives if no "
            "derived fact answers the question."
        ),
        instruction=(
            'Return the answer as a JSON array with one element, e.g. ["Alice"].'
        ),
        output_format="json_array",
        token_budget=400,
    )
    return PluginSet(
        name="mentor_kg_smoke",
        extractor=TemplateExtractor(patterns),
        prompt_template=prompt,
        derivation_rules=[rule_two_hop],
    )


# ---------------------------------------------------------------------------
# Harness with parallel execution
# ---------------------------------------------------------------------------


def _parallel_evaluate(card: TaskCard, plugins: PluginSet, llm, max_workers: int = 5):
    """Run TaskCard holdout via TestHarness, parallelizing per-example."""
    from quill.core.pipeline import V5Pipeline
    from quill.core.types import PipelineStats
    from dataclasses import asdict
    from quill.harness import (
        FailureCase, EvalReport, _METRICS, _exact_match, _preview,
        classify_failure, default_input_adapter,
    )
    from collections import Counter

    examples = card.spec.holdout
    adapter = default_input_adapter
    metric = _METRICS.get(card.spec.eval_metric, _exact_match)

    def run_one(ex):
        documents, query_input = adapter(ex)
        card.spec.options["_last_doc_keys"] = list(documents.keys())
        pipe = V5Pipeline(plugins, card.spec, llm)
        try:
            pred, stats = pipe.run(query_input, documents=documents)
            # For convo/kg tasks the output is a single-element JSON array;
            # unwrap so the metric can compare cleanly.
            if isinstance(pred, list) and len(pred) == 1 and isinstance(
                ex.expected_output, str,
            ):
                pred = pred[0]
            correct = bool(metric(pred, ex.expected_output))
            return ex, pred, stats, correct, None
        except Exception as e:
            return ex, None, PipelineStats(), False, e

    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, ex): ex for ex in examples}
        for f in as_completed(futures):
            results.append(f.result())

    n_correct = sum(1 for _, _, _, c, _ in results if c)
    total_tokens = sum(s.tokens_total for _, _, s, _, _ in results)
    total_prim = sum(s.n_primitives for _, _, s, _, _ in results)
    failures = []
    failure_counts: Counter = Counter()
    per_case = []
    for ex, pred, stats, correct, err in sorted(results, key=lambda r: r[0].qid or ""):
        if correct:
            per_case.append({"qid": ex.qid, "correct": True, "pred": pred,
                             "tokens": stats.tokens_total, "n_primitives": stats.n_primitives})
            continue
        if err is not None:
            category = "plugin_error"
            pred_repr = repr(err)
        else:
            category = classify_failure(
                [], stats, pred, ex.expected_output, card.spec,
            )
            pred_repr = pred
        failure_counts[category] += 1
        failures.append(FailureCase(
            qid=ex.qid, category=category, pred=pred_repr,
            expected=ex.expected_output, stats=asdict(stats),
            input_preview=_preview(ex.input),
        ))
        per_case.append({"qid": ex.qid, "correct": False, "category": category,
                         "pred": pred_repr, "expected": ex.expected_output,
                         "tokens": stats.tokens_total})

    n = len(examples) or 1
    report = EvalReport(
        task_domain=card.domain,
        n_examples=len(examples),
        n_correct=n_correct,
        accuracy=n_correct / n,
        avg_tokens=total_tokens / n,
        avg_primitives=total_prim / n,
        failure_counts=dict(failure_counts),
        failures=failures,
        per_case=per_case,
        wallclock_s=time.perf_counter() - start,
    )
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "haiku"], default="openai",
                    help="LLM backend: 'openai' = gpt-4o-mini via SDK (default), "
                         "'haiku' = claude CLI --max-turns 1")
    ap.add_argument("--model", default=None,
                    help="Model name override; defaults to 'gpt-4o-mini' for openai, "
                         "'haiku' for haiku")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument(
        "--output",
        default="results/exp22_v5_smoke.json",
    )
    args = ap.parse_args()

    if args.backend == "openai":
        model = args.model or "gpt-4o-mini"
        def llm(prompt: str) -> tuple[str, int, int]:
            return openai_4omini(prompt, model=model)
    else:
        model = args.model or "haiku"
        def llm(prompt: str) -> tuple[str, int, int]:
            return haiku_cli(prompt, model=model)
    args.model = model  # for the output header

    suites = [
        ("python_deps", _py_deps_card(), build_python_deps_plugins()),
        ("convo_fact",  _convo_card(),   _convo_plugins()),
        ("mentor_kg",   _kg_card(),      _kg_plugins()),
    ]

    overall = {"backend": args.backend, "model": args.model,
               "workers": args.workers, "suites": {}}
    print(f"=== V5 smoke test ({args.backend}:{args.model}, parallel={args.workers}) ===")
    t0 = time.perf_counter()
    for name, card, plugins in suites:
        print(f"\n-- {name} ({len(card.spec.holdout)} questions) --")
        rep = _parallel_evaluate(card, plugins, llm, max_workers=args.workers)
        print(f"  accuracy: {rep.n_correct}/{rep.n_examples} = {rep.accuracy:.0%}")
        print(f"  avg tokens: {rep.avg_tokens:.0f}  avg primitives: {rep.avg_primitives:.1f}")
        print(f"  wallclock: {rep.wallclock_s:.1f}s")
        if rep.failure_counts:
            print(f"  failures: {rep.failure_counts}")
        overall["suites"][name] = {
            "accuracy": rep.accuracy,
            "n_correct": rep.n_correct,
            "n_examples": rep.n_examples,
            "avg_tokens": rep.avg_tokens,
            "avg_primitives": rep.avg_primitives,
            "wallclock_s": rep.wallclock_s,
            "failure_counts": rep.failure_counts,
            "per_case": rep.per_case,
        }

    overall["wallclock_s_total"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(overall, f, indent=2, default=str)
    print(f"\nWrote {args.output} (total wallclock: {overall['wallclock_s_total']:.1f}s)")


if __name__ == "__main__":
    main()
