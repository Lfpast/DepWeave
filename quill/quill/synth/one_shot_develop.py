"""One-shot develop-tools workflow.

Given a single **labeled** example (input + expected_output), the LLM
designs an Extractor and PromptTemplate together, runs the full pipeline
on that one example, compares to the ground truth, and revises until the
pipeline reproduces the expected answer (or a partial-credit threshold is
met). The frozen PluginSet can then be applied to held-out items.

Motivation: the regular synthesis loop grades the extractor with generic
quality signals (coverage, relation mix). Those signals correlate
weakly with task accuracy. **End-to-end output matching on a labeled
example is a much sharper signal** — it tells the LLM exactly what
information its tools must surface for the real task to succeed.

Usage::

    from quill.synth.one_shot_develop import develop_from_one_shot
    plugins, trace = develop_from_one_shot(card, labeled_example, llm)
    # Then evaluate with V5Pipeline + plugins on holdout.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from quill.pipeline import V5Pipeline
from quill.plugins import LLMCaller, PluginSet, PromptTemplate
from quill.types import EvidenceBundle, Example, TaskSpec
from quill.harness import _METRICS, _exact_match, default_input_adapter
from quill.synth._json_tolerant import parse_json_object
from quill.synth.extractor_synth import (
    ExtractionPattern, TemplateExtractor, _patterns_to_json,
)


_SYSTEM_PROMPT = """You design PixelMem extraction + prompt tools from a SINGLE labeled example.

You see one (input, expected_output) pair and must design:
  1) regex extraction patterns that surface the symbols/signals the LLM
     will need to produce the expected output, and
  2) a prompt template that, when given those extracted primitives +
     the task input, will make a downstream LLM produce the expected output.

CRITICAL: return ONLY a valid JSON object. No code fences. No raw-string
literals like r"...". Plain JSON strings only — escape backslashes as \\\\.

Use this exact shape:

{
  "extractor": {
    "patterns": [
      {
        "name": "short_name",
        "regex": "Python regex using (?P<name>...) groups; escape \\\\ as needed",
        "relation": "snake_case_name",
        "condition": "optional_channel_or_empty",
        "field_mapping": {"subject": "group_name", "object": "group_name"}
      }
    ]
  },
  "prompt": {
    "header": "one-to-two-sentence task description",
    "instruction": "exact instruction telling the LLM what to return, one line",
    "output_format": "line"
  },
  "rationale": "one sentence"
}

Allowed output_format values: "json_array", "json_string", "single_token", "line".

Design guidance:
- Aim for GENERAL patterns, not patterns specific to the labeled example's
  symbol names. E.g. "extract all `def X():` lines" generalizes; "match
  the literal string `converse`" does not.
- The labeled example shows you what KIND of information the tools must
  surface. Abstract from the specific symbols used in that one example.
- The extractor should emit at least 3-10 primitives per document.
"""


_REVISION_PROMPT = """Your previous tools did NOT produce the expected output
on the labeled example. Revise MINIMALLY to close the gap.

Make the SMALLEST change that would flip the pipeline's prediction toward
the expected output. Don't rewrite what already works. Specifically:
- If primitives already cover the answer's key symbols, fix the prompt.
- If the prompt is right but primitives are missing key symbols, tighten
  or widen individual regexes.
- Never pivot the task (the task is STILL: produce exactly the expected
  output shape). Don't switch to "describe the code".

Return ONLY the revised JSON in the same shape as before.

=== Labeled example ===
INPUT DOCUMENTS (abbreviated):
{docs_summary}

QUERY INPUT:
{query_input}

EXPECTED OUTPUT:
{expected}

=== What happened on the last attempt ===
Your previous extractor patterns:
{prev_patterns}

They produced these primitives (first 20):
{prev_primitives}

Your previous prompt header:
{prev_header}
Your previous prompt instruction:
{prev_instruction}
Your previous output_format: {prev_format}

The full prompt that was sent to the downstream LLM was:
--- BEGIN PROMPT ---
{prev_prompt_text}
--- END PROMPT ---

The downstream LLM replied:
--- BEGIN RESPONSE ---
{prev_completion}
--- END RESPONSE ---

Which parsed to:
{prev_pred}

=== Issue ===
{issue_summary}

Redesign the tools. Common fixes:
- If the extractor missed a key symbol the expected output uses,
  widen the regex so it emits a primitive mentioning that symbol.
- If the LLM chose the wrong candidate, rewrite the prompt to
  disambiguate (show relevant defs more prominently, instruct "use the
  exact symbol name from the primitives", etc.).
- If the LLM's output format was wrong, tighten the instruction.
"""


# ---------------------------------------------------------------------------
# One-shot-specific PromptTemplate (built from the synthesized spec)
# ---------------------------------------------------------------------------


class OneShotSynthesizedPrompt(PromptTemplate):
    """PromptTemplate where the LLM freely designed header/instruction/format."""

    def __init__(
        self,
        header: str,
        instruction: str,
        output_format: str = "line",
    ) -> None:
        self._header = (header or "Answer the task.").rstrip() + "\n"
        self._instruction = (instruction or "Return only the answer.").strip()
        self._format = output_format or "line"

    def build(self, task: TaskSpec, query_input: dict, evidence: EvidenceBundle) -> str:
        raw_lines = [
            f"  ({p.subject}, {p.relation}, {p.object[:120]})"
            for p in evidence.raw_primitives[:50]
        ]
        prims_block = "\n".join(raw_lines) if raw_lines else "  (no primitives)"

        # Compact representation of the query input.
        qi = dict(query_input)
        # If the task provides a focal_prompt, keep only its tail (completion-style).
        if "focal_prompt" in qi and isinstance(qi["focal_prompt"], str):
            tail = "\n".join(qi["focal_prompt"].splitlines()[-60:])
            qi["focal_prompt"] = tail
        # If the task passes `documents`, drop full texts (primitives cover them).
        if "documents" in qi and isinstance(qi["documents"], dict):
            xf = qi["documents"].get("__crossfile_context__")
            qi = {k: v for k, v in qi.items() if k != "documents"}
            if xf:
                qi["__crossfile_context_head__"] = str(xf)[:1200]

        parts = [self._header,
                 "Extracted primitives:\n" + prims_block,
                 "Query input:\n" + json.dumps(qi, indent=2)[:2500],
                 self._instruction]
        return "\n\n".join(parts)

    def parse(self, completion: str, task: TaskSpec) -> Any:
        txt = completion.strip()
        if txt.startswith("```"):
            txt = "\n".join(l for l in txt.splitlines() if not l.startswith("```")).strip()

        if self._format == "json_array":
            m = re.search(r"\[.*\]", txt, re.DOTALL)
            if not m:
                raise ValueError("no JSON array")
            parsed = json.loads(m.group(0))
            if not isinstance(parsed, list):
                raise ValueError("not a list")
            return parsed
        if self._format == "json_string":
            m = re.search(r'"([^"]*)"', txt)
            if not m:
                raise ValueError("no quoted string")
            return m.group(1)
        if self._format == "single_token":
            tok = txt.split()[0] if txt else ""
            if not tok:
                raise ValueError("empty")
            return tok
        # "line" default
        for line in txt.splitlines():
            if line.strip():
                return line.rstrip("\n")
        raise ValueError("no non-empty line")


# ---------------------------------------------------------------------------
# Tool-spec parser
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    patterns: list[ExtractionPattern]
    header: str
    instruction: str
    output_format: str
    rationale: str = ""

    def to_plugin_set(self, name: str = "one_shot") -> PluginSet:
        return PluginSet(
            name=name,
            extractor=TemplateExtractor(self.patterns),
            prompt_template=OneShotSynthesizedPrompt(
                self.header, self.instruction, self.output_format,
            ),
            derivation_rules=[],
        )


def _parse_tool_spec(raw: str) -> ToolSpec:
    data = parse_json_object(raw)
    # Accept common shape variants the LLM drifts into.
    extractor = data.get("extractor") or {}
    if "patterns" not in extractor:
        # Sometimes emitted as a top-level "extractor_patterns" list
        # or nested under "extractor.patterns" or "patterns" alone.
        alt = data.get("extractor_patterns") or data.get("patterns")
        if isinstance(alt, list):
            extractor = {"patterns": alt}
        elif isinstance(alt, dict) and "patterns" in alt:
            extractor = alt
    prompt = (data.get("prompt") or data.get("prompt_template") or {})

    # Build a minimal schema-free container so the existing _parse_patterns
    # validator (which requires relations-in-schema) is bypassed; here we
    # accept free-form relation names.
    patterns: list[ExtractionPattern] = []
    rejected: list[str] = []
    for p in extractor.get("patterns", []):
        # Defend against the LLM emitting strings or malformed entries.
        if not isinstance(p, dict):
            rejected.append(f"non-dict pattern entry: {type(p).__name__}")
            continue
        regex = p.get("regex") or p.get("pattern")
        if not isinstance(regex, str) or not regex:
            rejected.append(f"pattern '{p.get('name')}': missing/invalid regex")
            continue
        try:
            re.compile(regex, re.MULTILINE)
        except re.error as e:
            rejected.append(f"pattern '{p.get('name') or p.get('label')}': {e}")
            continue
        patterns.append(ExtractionPattern(
            name=str(p.get("name") or p.get("label") or f"p{len(patterns)}"),
            regex=str(regex),
            relation=str(p.get("relation", p.get("label", "mention"))),
            condition=str(p.get("condition", "")),
            field_mapping=dict(p.get("field_mapping", {}) or {}),
            default_subject=p.get("default_subject"),
        ))

    if not patterns:
        reason = "; ".join(rejected) if rejected else "no patterns in tool spec"
        raise ValueError(f"no usable patterns: {reason}")

    return ToolSpec(
        patterns=patterns,
        header=str(prompt.get("header", "")),
        instruction=str(prompt.get("instruction", "")),
        output_format=str(prompt.get("output_format", "line")),
        rationale=str(data.get("rationale", "")),
    )


# ---------------------------------------------------------------------------
# Develop loop
# ---------------------------------------------------------------------------


@dataclass
class DevelopIteration:
    iteration: int
    parse_ok: bool
    pipeline_ok: bool
    correct: bool
    score: float
    pred: Any
    n_primitives: int
    issues: list[str] = field(default_factory=list)


def develop_from_one_shot(
    task: TaskSpec,
    example: Example,
    llm: LLMCaller,
    max_iterations: int = 4,
    score_fn: Optional[Callable[[Any, Any], float]] = None,
    threshold: float = 1.0,
    debug: bool = False,
) -> tuple[Optional[PluginSet], list[DevelopIteration], Any]:
    """Run the develop-loop until the pipeline reproduces ``example.expected_output``.

    Args:
        task: TaskSpec (eval_metric drives the default scoring).
        example: the single labeled example.
        llm: LLM caller.
        max_iterations: hard cap on LLM calls.
        score_fn: ``(pred, expected) -> float in [0, 1]``. If None, uses
            the TaskSpec metric (``exact_match`` → 1.0/0.0).
        threshold: score ≥ threshold counts as "done".

    Returns:
        (plugins, trace, final_pred). ``plugins`` is None if no valid tools
        were ever produced.
    """
    if score_fn is None:
        metric = _METRICS.get(task.eval_metric, _exact_match)
        def _score(p, e):
            r = metric(p, e)
            return 1.0 if bool(r) is True or r == 1.0 else (float(r) if isinstance(r, float) else 0.0)
        score_fn = _score

    docs, query_input = default_input_adapter(example)
    docs_summary = _abbreviate_docs(docs)
    expected = example.expected_output

    trace: list[DevelopIteration] = []
    tools: Optional[ToolSpec] = None
    last_pred = None
    last_prompt_text = ""
    last_completion = ""
    last_primitives: list = []
    # Keep the best-scoring tools so a bad revision can't erase progress.
    best_tools: Optional[ToolSpec] = None
    best_score: float = -1.0
    best_pred: Any = None

    for it in range(max_iterations):
        # ---------- 1. Ask the LLM for (possibly revised) tools ----------
        if it == 0:
            prompt = (
                _SYSTEM_PROMPT
                + "\n---\n"
                + f"Task domain: {task.domain}\n"
                + f"Task description: {task.description}\n"
                + "\nLABELED EXAMPLE:\n"
                + f"INPUT DOCUMENTS:\n{docs_summary}\n\n"
                + f"QUERY INPUT:\n{json.dumps(query_input)[:1500]}\n\n"
                + f"EXPECTED OUTPUT:\n{json.dumps(expected)[:500]}\n"
            )
        else:
            prompt = _REVISION_PROMPT.format(
                docs_summary=docs_summary,
                query_input=json.dumps(query_input)[:800],
                expected=json.dumps(expected)[:500],
                prev_patterns=_patterns_to_json(tools.patterns) if tools else "(no patterns)",
                prev_primitives=_prim_summary(last_primitives),
                prev_header=tools.header if tools else "",
                prev_instruction=tools.instruction if tools else "",
                prev_format=tools.output_format if tools else "",
                prev_prompt_text=last_prompt_text[:1800],
                prev_completion=last_completion[:600],
                prev_pred=json.dumps(last_pred)[:300] if last_pred is not None else "(parse failed)",
                issue_summary=_issue_summary(last_pred, expected),
            )

        raw, _, _ = llm(prompt)
        try:
            tools = _parse_tool_spec(raw)
        except Exception as e:
            if debug:
                print(f"[one-shot] iter {it} parse failed: {e}")
            trace.append(DevelopIteration(
                iteration=it, parse_ok=False, pipeline_ok=False, correct=False,
                score=0.0, pred=None, n_primitives=0,
                issues=[f"tool-spec parse: {type(e).__name__}: {e}"],
            ))
            continue

        # ---------- 2. Run the pipeline on the one-shot ----------
        plugins = tools.to_plugin_set()

        # Instrument the prompt template so we capture the exact text sent.
        captured = {}
        orig_build = plugins.prompt_template.build
        def capture(task_, q_, ev_):
            t = orig_build(task_, q_, ev_)
            captured["prompt"] = t
            return t
        plugins.prompt_template.build = capture  # type: ignore[method-assign]

        captured_llm_out = {}
        def llm_capture(p):
            c, tin, tout = llm(p)
            captured_llm_out["text"] = c
            return c, tin, tout

        pipe = V5Pipeline(plugins, task, llm_capture)
        pred = None
        n_prim = 0
        try:
            pred, stats = pipe.run(query_input, documents=docs)
            n_prim = stats.n_primitives
            last_primitives = pipe.primitives
        except Exception as e:
            if debug:
                print(f"[one-shot] iter {it} pipeline failed: {e}")
            trace.append(DevelopIteration(
                iteration=it, parse_ok=True, pipeline_ok=False, correct=False,
                score=0.0, pred=None, n_primitives=0,
                issues=[f"pipeline: {type(e).__name__}: {e}"],
            ))
            last_prompt_text = captured.get("prompt", "")
            last_completion = captured_llm_out.get("text", "")
            continue

        last_prompt_text = captured.get("prompt", "")
        last_completion = captured_llm_out.get("text", "")
        last_pred = pred

        # ---------- 3. Score against ground truth ----------
        score = float(score_fn(pred, expected))
        correct = score >= threshold

        if debug:
            print(f"[one-shot] iter {it}: n_prim={n_prim}  "
                  f"pred={str(pred)[:80]!r}  score={score:.2f}")

        trace.append(DevelopIteration(
            iteration=it, parse_ok=True, pipeline_ok=True, correct=correct,
            score=score, pred=pred, n_primitives=n_prim,
            issues=[] if correct else [_issue_summary(pred, expected)],
        ))

        if score > best_score:
            best_score = score
            best_tools = tools
            best_pred = pred

        if correct:
            break

    # Return the best-scoring tools, not the last ones. This is crucial:
    # the LLM can drift into worse revisions, and we don't want to punish
    # a good iter 0 with a bad iter 3.
    final_plugins = best_tools.to_plugin_set() if best_tools else (
        tools.to_plugin_set() if tools else None
    )
    return final_plugins, trace, best_pred if best_tools else last_pred


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abbreviate_docs(docs: dict[str, str], max_chars_per_doc: int = 1200) -> str:
    blocks = []
    for doc_id, text in list(docs.items())[:4]:
        blocks.append(f"# ===== {doc_id} =====\n{str(text)[:max_chars_per_doc]}")
    return "\n\n".join(blocks)


def _prim_summary(prims: list, limit: int = 20) -> str:
    if not prims:
        return "(none)"
    lines = []
    for p in prims[:limit]:
        lines.append(f"  ({p.subject}, {p.relation}, {str(p.object)[:100]})")
    return "\n".join(lines)


def _issue_summary(pred, expected) -> str:
    if pred is None:
        return "Pipeline produced no parseable prediction."
    if isinstance(pred, str) and isinstance(expected, str):
        if expected in pred:
            return (f"Prediction was a superstring of expected but not equal: "
                    f"pred={pred!r}, expected={expected!r}. "
                    "Tighten the instruction.")
        if pred in expected:
            return (f"Prediction is a substring of expected (missed tail). "
                    f"pred={pred!r}, expected={expected!r}. Extractor "
                    "probably missed a later symbol.")
    return (f"Prediction != expected. pred={str(pred)[:120]!r}, "
            f"expected={str(expected)[:120]!r}")


__all__ = ["develop_from_one_shot", "ToolSpec", "DevelopIteration",
           "OneShotSynthesizedPrompt"]
