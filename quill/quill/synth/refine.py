"""Stage 7 — Refinement loop.

Reads the failure-counts off an ``EvalReport`` and decides which synthesis
stage to re-run. Keeps the change minimal per iteration so the test harness
can localize any regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from quill.harness import EvalReport


# Map from failure category to the synthesis stage responsible.
FAILURE_TO_STAGE = {
    "missing_primitive": "extractor",
    "zero_strong_edges": "derivation",
    "parse_error": "prompt",
    "wrong_order": "prompt",      # prompt likely needs clearer ordering hint
    "wrong_answer": "prompt",
    "plugin_error": "extractor",  # extractor is the first place to look
    "true_ambiguity": None,       # unfixable from this input alone
}


@dataclass
class RefinementOutcome:
    action: str       # "rerun_extractor" | "rerun_derivation" | "rerun_prompt" | "stop"
    reason: str
    dominant_failure: Optional[str] = None


class Refiner:
    """Decides the next synthesis action from an EvalReport.

    The rule is deliberately conservative:

    - if accuracy >= threshold, stop
    - else pick the single most-common failure category and rerun its stage
    - if the dominant category is unfixable (true_ambiguity), stop with best-so-far
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        self._threshold = threshold

    def decide(self, report: EvalReport, task_threshold: float) -> RefinementOutcome:
        bar = self._threshold if self._threshold is not None else task_threshold
        if report.accuracy >= bar:
            return RefinementOutcome(
                action="stop",
                reason=f"accuracy {report.accuracy:.2%} meets threshold {bar:.2%}",
            )

        if not report.failure_counts:
            return RefinementOutcome(
                action="stop",
                reason="no failures logged but accuracy below threshold — "
                       "likely empty dataset or metric misconfig",
            )

        dominant, count = max(report.failure_counts.items(), key=lambda x: x[1])
        stage = FAILURE_TO_STAGE.get(dominant)
        if stage is None:
            return RefinementOutcome(
                action="stop",
                reason=f"dominant failure '{dominant}' is unfixable from the "
                       f"current input set",
                dominant_failure=dominant,
            )
        return RefinementOutcome(
            action=f"rerun_{stage}",
            reason=f"{count}/{report.n_examples} failures classified as "
                   f"'{dominant}' -> retry {stage}",
            dominant_failure=dominant,
        )


__all__ = ["Refiner", "RefinementOutcome", "FAILURE_TO_STAGE"]
