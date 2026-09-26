"""Test harness — runs a V5Pipeline over TaskCard examples.

Separates V5's eval loop from the hand-written benchmark scripts in
``experiments/``. Benchmarks can now be added by writing a TaskCard plus a
small input adapter, rather than a new exp*.py per benchmark.

The harness also classifies failures into a small taxonomy, which is what
the synthesis refiner reads to decide which stage to patch.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from shortmem.pipeline import V5Pipeline
from shortmem.plugins import LLMCaller, PluginSet
from shortmem.types import Example, PipelineStats, Primitive, TaskSpec
from shortmem.task_card import TaskCard


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _exact_match(pred: Any, expected: Any) -> bool:
    return pred == expected


def _fuzzy_match(pred: Any, expected: Any) -> bool:
    """Lowercase, whitespace-normalized equality for strings; list/set equal for lists."""
    if isinstance(pred, str) and isinstance(expected, str):
        return " ".join(pred.lower().split()) == " ".join(expected.lower().split())
    if isinstance(pred, list) and isinstance(expected, list):
        return [str(p).lower().strip() for p in pred] == [str(e).lower().strip() for e in expected]
    return pred == expected


def _f1_list(pred: Any, expected: Any) -> float:
    if not isinstance(pred, list) or not isinstance(expected, list):
        return 0.0
    ps, es = set(pred), set(expected)
    if not ps and not es:
        return 1.0
    if not ps or not es:
        return 0.0
    prec = len(ps & es) / len(ps)
    rec = len(ps & es) / len(es)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


_METRICS: dict[str, Callable[[Any, Any], bool | float]] = {
    "exact_match": _exact_match,
    "fuzzy": _fuzzy_match,
    "f1": _f1_list,
}


# ---------------------------------------------------------------------------
# Failure classifier
# ---------------------------------------------------------------------------


def classify_failure(
    primitives: list[Primitive],
    stats: PipelineStats,
    pred: Any,
    expected: Any,
    task: TaskSpec,
) -> str:
    """Return one of:

    - ``missing_primitive``: extractor produced too few primitives
      (task.options["min_primitives_per_doc"] violated, or zero for a non-empty doc set).
    - ``zero_strong_edges``: derivation produced no strong evidence — graph is
      too sparse to constrain the LLM (similar symptom to V4's "ambiguous" bucket).
    - ``parse_error``: the prompt template couldn't parse the LLM output.
    - ``wrong_order`` / ``wrong_answer``: stats look healthy but output is wrong.
    - ``true_ambiguity``: parse clean, strong edges present, output still wrong —
      likely a gap the current plugin set can't fill without more info.

    The classifier is deliberately conservative; ambiguous cases fall back to
    ``wrong_answer`` so the refiner doesn't thrash on false signals.
    """
    if stats.parse_error:
        return "parse_error"

    min_primitives = int(task.options.get("min_primitives_per_doc", 1))
    n_docs = len(task.options.get("_last_doc_keys", []) or [])
    if stats.n_primitives == 0:
        return "missing_primitive"
    if n_docs > 0 and stats.n_primitives < min_primitives * n_docs:
        return "missing_primitive"

    if stats.n_strong_edges == 0 and stats.n_ambiguous_edges == 0:
        return "zero_strong_edges"

    if task.query.get("kind") == "ordering":
        if isinstance(pred, list) and isinstance(expected, list):
            if set(pred) == set(expected) and pred != expected:
                return "wrong_order"
        return "wrong_answer" if pred != expected else "correct"

    return "wrong_answer" if pred != expected else "correct"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class FailureCase:
    qid: Optional[str]
    category: str
    pred: Any
    expected: Any
    stats: dict
    input_preview: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalReport:
    task_domain: str
    n_examples: int
    n_correct: int
    accuracy: float
    avg_tokens: float
    avg_primitives: float
    failure_counts: dict = field(default_factory=dict)
    failures: list[FailureCase] = field(default_factory=list)
    per_case: list[dict] = field(default_factory=list)
    wallclock_s: float = 0.0

    @property
    def passed_threshold(self) -> bool:
        return False  # set by harness, see TestHarness.evaluate

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    def to_markdown(self) -> str:
        lines = [
            f"# Eval report — {self.task_domain}",
            "",
            f"- examples: **{self.n_examples}**",
            f"- correct: **{self.n_correct}** ({self.accuracy:.1%})",
            f"- avg tokens/query: **{self.avg_tokens:.0f}**",
            f"- avg primitives: **{self.avg_primitives:.1f}**",
            f"- wallclock: **{self.wallclock_s:.1f}s**",
            "",
            "## Failure breakdown",
            "",
        ]
        if not self.failure_counts:
            lines.append("_(no failures)_")
        for cat, n in sorted(self.failure_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cat}**: {n}")
        lines.append("")
        if self.failures:
            lines.append("## First 5 failures")
            for fc in self.failures[:5]:
                lines.append(
                    f"- `{fc.qid or '?'}` ({fc.category})\n"
                    f"    - expected: `{fc.expected}`\n"
                    f"    - got:      `{fc.pred}`"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adapters — how to turn an Example into (documents, query_input)
# ---------------------------------------------------------------------------


InputAdapter = Callable[[Example], tuple[dict, dict]]
"""Example -> (documents dict, query_input dict)."""


def default_input_adapter(ex: Example) -> tuple[dict, dict]:
    """Pull ``documents`` / ``query`` off the example input dict.

    Accepts either::

        {"documents": {...}, "query": {...}}

    or the python-deps shape::

        {"files": [...], "file_contents": {...}}
    """
    src = ex.input or {}
    if "documents" in src:
        return dict(src["documents"]), dict(src.get("query", {}))
    if "file_contents" in src:
        docs = dict(src["file_contents"])
        query = {"files": list(src.get("files", list(docs.keys()))),
                 "file_contents": docs}
        return docs, query
    # Fallback: treat entire input as both.
    return dict(src), dict(src)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class TestHarness:
    """Runs a V5Pipeline over a TaskCard's examples.

    Usage::

        harness = TestHarness(card, plugins, llm)
        report = harness.evaluate(split="holdout")
        print(report.to_markdown())
    """

    def __init__(
        self,
        card: TaskCard,
        plugins: PluginSet,
        llm: LLMCaller,
        input_adapter: Optional[InputAdapter] = None,
        debug: bool = False,
    ) -> None:
        self._card = card
        self._plugins = plugins
        self._llm = llm
        self._adapter = input_adapter or default_input_adapter
        self._debug = debug

    def evaluate(
        self,
        split: str = "holdout",
        limit: Optional[int] = None,
    ) -> EvalReport:
        examples = self._examples_for(split)
        if limit is not None:
            examples = examples[:limit]

        metric = _METRICS.get(self._card.spec.eval_metric, _exact_match)

        n_correct = 0
        total_tokens = 0
        total_primitives = 0
        failure_cases: list[FailureCase] = []
        per_case: list[dict] = []
        failure_counter: Counter = Counter()

        start = time.perf_counter()

        for ex in examples:
            documents, query_input = self._adapter(ex)

            self._card.spec.options["_last_doc_keys"] = list(documents.keys())

            pipeline = V5Pipeline(
                plugins=self._plugins,
                task=self._card.spec,
                llm=self._llm,
                debug=self._debug,
            )

            try:
                pred, stats = pipeline.run(query_input, documents=documents)
            except Exception as err:  # plugin raised — report, don't crash
                stats = PipelineStats()
                pred = None
                category = "plugin_error"
                failure_counter[category] += 1
                failure_cases.append(FailureCase(
                    qid=ex.qid,
                    category=category,
                    pred=repr(err),
                    expected=ex.expected_output,
                    stats=asdict(stats),
                    input_preview=_preview(ex.input),
                ))
                per_case.append({
                    "qid": ex.qid,
                    "correct": False,
                    "category": category,
                    "error": repr(err),
                })
                continue

            correct = bool(metric(pred, ex.expected_output))
            total_tokens += stats.tokens_total
            total_primitives += stats.n_primitives

            if correct:
                n_correct += 1
                per_case.append({
                    "qid": ex.qid,
                    "correct": True,
                    "category": "correct",
                    "tokens": stats.tokens_total,
                    "n_primitives": stats.n_primitives,
                })
            else:
                category = classify_failure(
                    pipeline.primitives, stats, pred, ex.expected_output,
                    self._card.spec,
                )
                failure_counter[category] += 1
                failure_cases.append(FailureCase(
                    qid=ex.qid,
                    category=category,
                    pred=pred,
                    expected=ex.expected_output,
                    stats=asdict(stats),
                    input_preview=_preview(ex.input),
                ))
                per_case.append({
                    "qid": ex.qid,
                    "correct": False,
                    "category": category,
                    "tokens": stats.tokens_total,
                    "n_primitives": stats.n_primitives,
                })

        n = len(examples) or 1
        report = EvalReport(
            task_domain=self._card.domain,
            n_examples=len(examples),
            n_correct=n_correct,
            accuracy=n_correct / n,
            avg_tokens=total_tokens / n,
            avg_primitives=total_primitives / n,
            failure_counts=dict(failure_counter),
            failures=failure_cases,
            per_case=per_case,
            wallclock_s=time.perf_counter() - start,
        )
        return report

    def _examples_for(self, split: str) -> list[Example]:
        if split == "holdout":
            return list(self._card.spec.holdout)
        if split == "few_shot":
            return list(self._card.spec.few_shot)
        if split == "all":
            return list(self._card.spec.few_shot) + list(self._card.spec.holdout)
        raise ValueError(f"unknown split: {split}")


def _preview(d: dict, max_chars: int = 300) -> dict:
    preview: dict = {}
    for k, v in (d or {}).items():
        if isinstance(v, str):
            preview[k] = v[:max_chars] + ("..." if len(v) > max_chars else "")
        elif isinstance(v, dict):
            preview[k] = {kk: (vv[:max_chars] if isinstance(vv, str) else vv)
                          for kk, vv in list(v.items())[:4]}
        elif isinstance(v, list):
            preview[k] = v[:10]
        else:
            preview[k] = v
    return preview


__all__ = [
    "TestHarness",
    "EvalReport",
    "FailureCase",
    "classify_failure",
    "default_input_adapter",
]
