"""Trace-informed two-stage tool development.

Motivation: surfacing candidates alone doesn't teach the LLM HOW to pick
the right one. Humans solve these tasks with a mental procedure:
  1. Reduce the search space (extract candidates, filter obvious noise)
  2. Finish the task (pick + compose + verify)

This module captures that split. A :class:`SolutionTrace` records HOW one
labeled example was solved — not just the final answer. The LLM studies
a small corpus of traces and develops TWO tools:

- :class:`ReducerTool`: extracts candidates + ranks them with
  disambiguation hints inferred from traces (e.g. "prefer shorter name
  when a longer compound name also matches"; "prefer function whose
  argument pattern matches what's visible in the focal").
- :class:`FinisherTool`: given the reducer's output + the task input,
  produces the final answer using a prompt template informed by traces.

Usage::

    traces = [SolutionTrace(...), SolutionTrace(...), ...]
    plugins, trace = develop_two_stage_from_traces(card, traces, llm)
    # plugins is a PluginSet with extractor = ReducerTool.as_extractor()
    # and prompt_template = FinisherTool.as_prompt().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from quill.plugins import LLMCaller, PluginSet, PromptTemplate
from quill.types import EvidenceBundle, Primitive, TaskSpec
from quill.synth._json_tolerant import parse_json_object
from quill.synth.candidate_aware import (
    SymbolExtractor, _analyze_completion_site, _rank_candidates,
)


# ---------------------------------------------------------------------------
# SolutionTrace — what an LLM studies
# ---------------------------------------------------------------------------


@dataclass
class SolutionTrace:
    """Captures HOW one labeled example was solved.

    This is what distinguishes trace-learning from plain few-shot: we
    don't just show the LLM (input, output), we show it the intermediate
    steps an effective solver would have taken.
    """

    input: dict                                  # Same shape as Example.input
    expected_output: Any
    qid: Optional[str] = None

    # --- the trace itself ---
    observation: str = ""                        # What the solver noticed
    candidate_extraction: list[str] = field(default_factory=list)
    # ex: ["extracted def lines from crossfile", "extracted bare identifiers"]
    disambiguation: list[str] = field(default_factory=list)
    # ex: ["both FOO and FOO_from_X present; prefer FOO because focal arg is .headers"]
    relevant_candidates: list[str] = field(default_factory=list)
    # ex: ["get_header_value", "get_header_value_from_response"]
    selected_candidate: Optional[str] = None
    # ex: "get_header_value"
    final_composition: str = ""
    # ex: "get_header_value(response.headers, self.rule.HEADER_NAME), \"0\")"

    def as_fewshot_block(self) -> str:
        """Compact string form the LLM can study."""
        lines = []
        lines.append(f"qid: {self.qid or '<unlabeled>'}")
        lines.append(f"expected_output: {json.dumps(self.expected_output)[:400]}")
        if self.observation:
            lines.append(f"observation: {self.observation}")
        if self.candidate_extraction:
            lines.append("candidate_extraction:")
            for s in self.candidate_extraction:
                lines.append(f"  - {s}")
        if self.disambiguation:
            lines.append("disambiguation:")
            for s in self.disambiguation:
                lines.append(f"  - {s}")
        if self.relevant_candidates:
            lines.append(f"relevant_candidates: {self.relevant_candidates}")
        if self.selected_candidate:
            lines.append(f"selected_candidate: {self.selected_candidate!r}")
        if self.final_composition:
            lines.append(f"final_composition: {self.final_composition!r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ReducerTool — extracts + ranks + filters candidates
# ---------------------------------------------------------------------------


@dataclass
class ReducerSpec:
    """LLM-proposed reducer configuration: extractor settings + ranker hints."""

    # Filters learned from traces. If the LLM observed "prefer shorter name
    # when compound variant exists", it may emit ``suppress_compound=True``.
    suppress_compound_siblings: bool = True
    # Penalty to subtract from a candidate's score if another candidate is a
    # prefix of it (i.e. `FOO` exists AND `FOO_from_X` exists → FOO_from_X loses).
    compound_penalty: int = 25
    # Names to always include in the candidate list (from trace observations).
    always_surface: list[str] = field(default_factory=list)
    # Names to never surface (noise learned from traces).
    suppress: list[str] = field(default_factory=list)
    top_k: int = 15

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class ReducerTool:
    """Two-phase reducer: SymbolExtractor + learned ranker.

    Used as the V5 Extractor: ``reduce`` runs extraction and attaches
    ranking metadata to each primitive via its ``provenance`` field.
    """

    def __init__(self, spec: ReducerSpec) -> None:
        self._spec = spec
        self._inner = SymbolExtractor()

    # V5 Extractor protocol
    def extract(self, documents: dict[str, str], **kwargs: Any) -> list[Primitive]:
        return self._inner.extract(documents)

    def rank(
        self,
        primitives: list[Primitive],
        completion_site,
        focal_tail: str,
    ) -> list[tuple[str, str, int]]:
        """Return ranked candidates with the spec's filters applied.

        Ranker: reuses the base ranker from candidate_aware, then applies
        compound-sibling suppression and the trace-learned white/black lists.
        """
        ranked = _rank_candidates(
            primitives, completion_site, focal_tail,
            top_k=self._spec.top_k * 2,
        )
        # Apply compound-sibling suppression
        if self._spec.suppress_compound_siblings:
            names = {name for name, _, _ in ranked}
            def is_compound_of_another(name: str) -> bool:
                for shorter in names:
                    if shorter == name:
                        continue
                    # Compound if name starts with shorter followed by "_" or
                    # shorter followed by some disambiguator.
                    if (len(shorter) >= 4 and len(name) > len(shorter) + 2
                            and name.startswith(shorter + "_")):
                        return True
                return False
            ranked = [
                (name, kind,
                 score - self._spec.compound_penalty if is_compound_of_another(name) else score)
                for name, kind, score in ranked
            ]
        # Apply whitelist boost
        for wsym in self._spec.always_surface:
            ranked = [
                (name, kind, score + 40 if name == wsym else score)
                for name, kind, score in ranked
            ]
        # Apply blacklist removal
        if self._spec.suppress:
            ranked = [(n, k, s) for n, k, s in ranked if n not in self._spec.suppress]
        # Re-sort and truncate
        ranked = sorted(ranked, key=lambda x: -x[2])[: self._spec.top_k]
        return ranked


# ---------------------------------------------------------------------------
# FinisherTool — produces final answer from candidates + input
# ---------------------------------------------------------------------------


@dataclass
class FinisherSpec:
    """LLM-proposed finisher configuration."""

    header: str = "Complete the next line."
    reasoning_steps: list[str] = field(default_factory=list)
    # ex: ["check which candidate's args match focal context",
    #      "prefer shorter name over compound sibling"]
    output_format: str = "line"    # "line" | "json_array" | "json_string"

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class FinisherTool(PromptTemplate):
    """Prompt template informed by trace reasoning steps.

    Injects the reducer's ranked candidates + explicit reasoning steps the
    LLM should follow (derived from successful solution traces).
    """

    def __init__(self, spec: FinisherSpec, reducer: ReducerTool) -> None:
        self._spec = spec
        self._reducer = reducer

    def build(self, task: TaskSpec, query_input: dict, evidence: EvidenceBundle) -> str:
        focal = query_input.get("focal_prompt", "") or ""
        focal_tail = "\n".join(focal.splitlines()[-40:])
        site = _analyze_completion_site(focal_tail)
        ranked = self._reducer.rank(evidence.raw_primitives, site, focal_tail)

        cand_lines = [f"  {name} ({kind}, score={score})"
                      for name, kind, score in ranked]
        cand_block = "\n".join(cand_lines) if cand_lines else "  (no candidates)"

        # Hard pick: the tool declares a top-1 choice when the top candidate
        # clearly outranks the runner-up. This turns the finisher into a
        # "compose, don't rechoose" step — useful when the LLM tends to
        # override the ranking with its priors.
        top_pick = None
        if len(ranked) >= 1:
            top_score = ranked[0][2]
            runner = ranked[1][2] if len(ranked) >= 2 else -999
            if top_score - runner >= 8:  # sufficiently confident
                top_pick = ranked[0][0]

        site_str = (f"kind={site.kind}"
                    + (f" receiver={site.receiver}" if site.receiver else "")
                    + (f" partial={site.partial}" if site.partial else ""))

        reasoning_block = ""
        if self._spec.reasoning_steps:
            reasoning_block = "\nFOLLOW THESE STEPS IN ORDER:\n" + "\n".join(
                f"  {i+1}. {s}" for i, s in enumerate(self._spec.reasoning_steps)
            )

        xfile = ""
        for doc_id, text in (query_input.get("documents") or {}).items():
            if doc_id == "__crossfile_context__":
                xfile = str(text)[:1500]
                break

        pick_block = ""
        if top_pick:
            pick_block = (
                f"\n\nTOOL'S CONFIDENT PICK: `{top_pick}`\n"
                "The ranking tool is confident this is the symbol to use. "
                "Compose the next line AROUND this exact name. Do NOT "
                "substitute a different similar-looking name from your "
                "prior. If the site is a method call on a receiver, the "
                "line usually starts with this exact symbol name."
            )

        return (
            f"{self._spec.header}\n\n"
            f"Completion site: {site_str}\n\n"
            f"Ranked candidates (reducer output):\n{cand_block}"
            f"{pick_block}\n\n"
            f"Cross-file context (trimmed):\n{xfile}\n\n"
            f"Focal file tail:\n{focal_tail}"
            f"{reasoning_block}\n\n"
            "Write ONE syntactically complete source line. No explanation, no fences."
        )

    def parse(self, completion: str, task: TaskSpec) -> str:
        txt = completion.strip()
        if txt.startswith("```"):
            txt = "\n".join(l for l in txt.splitlines()
                            if not l.startswith("```")).strip()
        for line in txt.splitlines():
            if line.strip():
                return line.rstrip("\n")
        raise ValueError("no non-empty line")


# ---------------------------------------------------------------------------
# Development loop — LLM studies traces, develops reducer + finisher
# ---------------------------------------------------------------------------


_DEVELOP_PROMPT = """You have studied the following SOLUTION TRACES for a task.
Each trace shows how a labeled example was solved — the observations made,
the candidate extraction and disambiguation steps, and how the answer
was composed.

Your job: develop TWO tools that generalize the traces' strategy.

=== Traces ===
{traces_block}

=== Your task ===
Design:

1. A REDUCER: how to narrow the candidate search space from the inputs.
2. A FINISHER: how to pick + compose the final answer from the reducer's output.

Return ONLY this JSON (no fences, no prose):

{{
  "reducer": {{
    "suppress_compound_siblings": true,
    "compound_penalty": 25,
    "always_surface": ["list of symbol names you learned ALWAYS matter"],
    "suppress": ["noise symbols to always drop"],
    "top_k": 15
  }},
  "finisher": {{
    "header": "one-sentence task description",
    "reasoning_steps": [
      "Step the LLM should take, grounded in what the traces show",
      "Second step",
      "..."
    ],
    "output_format": "line"
  }},
  "rationale": "one-sentence summary of the strategy you extracted from the traces"
}}
"""


def develop_two_stage_from_traces(
    task: TaskSpec,
    traces: list[SolutionTrace],
    llm: LLMCaller,
    debug: bool = False,
) -> tuple[Optional[PluginSet], dict]:
    """Ask the LLM to synthesize a ReducerTool + FinisherTool from traces.

    Returns ``(plugins, development_metadata)``. On parse failure,
    ``plugins`` is ``None``.
    """
    if not traces:
        raise ValueError("develop_two_stage_from_traces requires >= 1 trace")

    traces_block = "\n\n---\n\n".join(t.as_fewshot_block() for t in traces)
    prompt = _DEVELOP_PROMPT.format(traces_block=traces_block)

    raw, _in_tok, _out_tok = llm(prompt)
    if debug:
        print(f"[trace-learn] raw response (first 400 chars):\n{raw[:400]}")

    try:
        data = parse_json_object(raw)
    except Exception as e:
        if debug:
            print(f"[trace-learn] parse failed: {e}")
        return None, {"raw": raw, "error": str(e)}

    reducer_spec = ReducerSpec(
        suppress_compound_siblings=bool(
            data.get("reducer", {}).get("suppress_compound_siblings", True)
        ),
        compound_penalty=int(data.get("reducer", {}).get("compound_penalty", 25)),
        always_surface=list(data.get("reducer", {}).get("always_surface", [])),
        suppress=list(data.get("reducer", {}).get("suppress", [])),
        top_k=int(data.get("reducer", {}).get("top_k", 15)),
    )
    finisher_spec = FinisherSpec(
        header=str(data.get("finisher", {}).get("header", "Complete the next line.")),
        reasoning_steps=list(data.get("finisher", {}).get("reasoning_steps", [])),
        output_format=str(data.get("finisher", {}).get("output_format", "line")),
    )

    reducer = ReducerTool(reducer_spec)
    finisher = FinisherTool(finisher_spec, reducer)

    plugins = PluginSet(
        name="two_stage_trace_learned",
        extractor=reducer,
        prompt_template=finisher,
        derivation_rules=[],
    )
    metadata = {
        "reducer_spec": asdict(reducer_spec),
        "finisher_spec": asdict(finisher_spec),
        "rationale": data.get("rationale", ""),
        "raw_response": raw[:1500],
    }
    return plugins, metadata


__all__ = [
    "SolutionTrace", "ReducerSpec", "ReducerTool",
    "FinisherSpec", "FinisherTool",
    "develop_two_stage_from_traces",
]
