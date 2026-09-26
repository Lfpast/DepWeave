"""End-to-end synthesis orchestrator.

Runs the full synthesis loop for a TaskCard:

    1. SchemaDesigner    -> SchemaProposal
    2. ExtractorSynth    -> TemplateExtractor
    3. DerivationSynth   -> list[DerivationRule]
    4. PromptSynth       -> SynthesizedPrompt
    5. PluginSet assembled and fed to TestHarness
    6. Refiner decides to stop or patch one stage; repeat up to N times.

The orchestrator is the top-level entry point used by benchmark scripts and
by anyone who wants "give PixelMem a task, get a pipeline back."

Note: for domains where V4's extractor is still best (python code), a caller
can bypass synthesis entirely by passing a pre-built PluginSet to TestHarness
directly. Synthesis is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quill.plugins import LLMCaller, PluginSet
from quill.harness import EvalReport, TestHarness
from quill.synth.derivation_synth import DerivationSynth
from quill.synth.extractor_synth import (
    ExtractionPattern, ExtractorSynth, TemplateExtractor,
)
from quill.synth.prompt_synth import PromptSynth, SynthesizedPrompt
from quill.synth.refine import Refiner, RefinementOutcome
from quill.synth.schema_designer import SchemaDesigner, SchemaProposal
from quill.task_card import TaskCard


# Fallback plugins used when synthesis raises. Not pretty, but keeps the
# loop moving so one bad LLM response can't bring down the whole run.

def _fallback_extractor(schema: SchemaProposal) -> TemplateExtractor:
    """Catch-all: one broad pattern that always emits at least one primitive."""
    rel = schema.relations[0] if schema.relations else "mentions"
    cond = schema.conditions[0] if schema.conditions else ""
    patterns = [
        ExtractionPattern(
            name="any_line",
            regex=r"^(?P<o>[^\n]{3,})$",
            relation=rel,
            condition=cond,
            field_mapping={"object": "o"},
        ),
    ]
    return TemplateExtractor(patterns)


def _fallback_prompt(schema: SchemaProposal) -> SynthesizedPrompt:
    fmt = "json_array"
    qos = schema.query_output_schema or {}
    if qos.get("type") == "string":
        fmt = "json_string"
    return SynthesizedPrompt(
        header=(
            "Answer the user's question from the raw primitives below. "
            "Do not invent facts."
        ),
        instruction='Return the answer as a JSON array, e.g. ["..."].',
        output_format=fmt,
    )


def _try(label: str, fn, debug: bool):
    """Run a synthesis step, log failures, return (result, ok)."""
    try:
        return fn(), True
    except Exception as e:
        if debug:
            print(f"[V5 synth] {label} failed: {repr(e)[:200]}")
        return None, False


@dataclass
class SynthesisResult:
    plugins: PluginSet
    schema: SchemaProposal
    iterations: int
    final_report: EvalReport
    trace: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.final_report.accuracy >= self._threshold

    _threshold: float = 0.0


def synthesize_pipeline(
    card: TaskCard,
    llm: LLMCaller,
    max_iterations: int = 3,
    debug: bool = False,
) -> SynthesisResult:
    """Synthesize a PluginSet for a TaskCard and return the best version.

    Currently the refiner resynthesizes only the stage it identifies as
    dominant (extractor / derivation / prompt). This bounds LLM-call cost
    per iteration to one or two calls.
    """
    task = card.spec
    threshold = task.eval_threshold

    designer = SchemaDesigner(llm)
    ext_synth = ExtractorSynth(llm)
    der_synth = DerivationSynth(llm)
    pr_synth = PromptSynth(llm)
    refiner = Refiner(threshold=threshold)

    schema, _ = _try("SchemaDesigner", lambda: designer.design(task), debug)
    if schema is None:
        # Last-resort minimal schema so the rest of the pipeline can run.
        schema = SchemaProposal(
            relations=["mentions"],
            conditions=["raw"],
            query_output_schema={"type": "list"},
            rationale="fallback (SchemaDesigner failed)",
        )

    sample = ""
    for ex in task.few_shot[:1]:
        for v in (ex.input or {}).values():
            if isinstance(v, str):
                sample = v
                break
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, str):
                        sample = vv
                        break
                break

    # Extractor: tight develop → test → revise inner loop.
    ext_inner_iters = int(task.options.get("extractor_inner_iterations", 3))
    if task.few_shot and ext_inner_iters > 1:
        result = _try(
            "ExtractorSynth (feedback loop)",
            lambda: ext_synth.synthesize_with_feedback(
                schema, task.domain, task.description,
                task, task.few_shot,
                max_iterations=ext_inner_iters,
                debug=debug,
                few_shot_sample=sample,
            ),
            debug,
        )
        # `_try` returns ((extractor, final_test, trace), ok) on success
        payload, ok = result
        if ok and payload and payload[0] is not None:
            extractor = payload[0]
            # Expose the test trace for debugging / reporting.
            task.options["_extractor_test_trace"] = [
                {
                    "passed": t.passed,
                    "n_primitives": t.n_primitives,
                    "cross_doc_edges": t.cross_doc_edges,
                    "relation_counts": t.relation_counts,
                    "issues": t.issues,
                }
                for t in payload[2]
            ]
        else:
            extractor = _fallback_extractor(schema)
    else:
        extractor, ext_ok = _try(
            "ExtractorSynth",
            lambda: ext_synth.synthesize(schema, task.domain, task.description, sample),
            debug,
        )
        if not ext_ok:
            extractor = _fallback_extractor(schema)

    rules, _rules_ok = _try(
        "DerivationSynth",
        lambda: der_synth.synthesize(schema, task.domain, task.description),
        debug,
    )
    if rules is None:
        rules = []

    prompt_template, pr_ok = _try(
        "PromptSynth",
        lambda: pr_synth.synthesize(schema, task.domain, task.description),
        debug,
    )
    if not pr_ok:
        prompt_template = _fallback_prompt(schema)

    plugins = PluginSet(
        name=f"synthesized_{task.domain}",
        extractor=extractor,
        derivation_rules=rules,
        prompt_template=prompt_template,
    )
    plugins.validate()

    trace: list[dict] = []
    report = None

    for it in range(max_iterations):
        harness = TestHarness(card, plugins, llm, debug=debug)
        split = "few_shot" if task.few_shot else "holdout"
        report = harness.evaluate(split=split)
        trace.append({
            "iteration": it,
            "accuracy": report.accuracy,
            "failure_counts": report.failure_counts,
        })

        outcome: RefinementOutcome = refiner.decide(report, threshold)
        if debug:
            print(f"[V5 synth] iter {it}: acc={report.accuracy:.2%} "
                  f"action={outcome.action} reason={outcome.reason}")

        if outcome.action == "stop":
            break
        if outcome.action == "rerun_extractor":
            new_ext, ok = _try(
                "rerun ExtractorSynth",
                lambda: ext_synth.synthesize(
                    schema, task.domain, task.description, sample,
                ),
                debug,
            )
            if ok:
                extractor = new_ext
            plugins = PluginSet(
                name=plugins.name,
                extractor=extractor,
                derivation_rules=rules,
                prompt_template=prompt_template,
            )
        elif outcome.action == "rerun_derivation":
            new_rules, ok = _try(
                "rerun DerivationSynth",
                lambda: der_synth.synthesize(schema, task.domain, task.description),
                debug,
            )
            if ok:
                rules = new_rules
            plugins = PluginSet(
                name=plugins.name,
                extractor=extractor,
                derivation_rules=rules,
                prompt_template=prompt_template,
            )
        elif outcome.action == "rerun_prompt":
            new_prompt, ok = _try(
                "rerun PromptSynth",
                lambda: pr_synth.synthesize(schema, task.domain, task.description),
                debug,
            )
            if ok:
                prompt_template = new_prompt
            plugins = PluginSet(
                name=plugins.name,
                extractor=extractor,
                derivation_rules=rules,
                prompt_template=prompt_template,
            )
        plugins.validate()

    assert report is not None
    result = SynthesisResult(
        plugins=plugins,
        schema=schema,
        iterations=len(trace),
        final_report=report,
        trace=trace,
    )
    result._threshold = threshold
    return result


__all__ = ["synthesize_pipeline", "SynthesisResult"]
