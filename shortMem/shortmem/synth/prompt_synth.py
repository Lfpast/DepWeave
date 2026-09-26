"""Stage 5 — Prompt template synthesis.

Again template-driven: the LLM fills slots rather than writing code.

The synthesized prompt has fixed structure:
  <header from LLM>
  <dynamic: raw primitives list>
  <dynamic: confirmed (strong) edges>
  <dynamic: ordering_hint, if applicable>
  <instruction + output format from LLM>

Parsing is JSON-array by default (works for ordering / list outputs); a
regex parser is used when the schema says output type is a single string.
"""

from __future__ import annotations

import json
import re
from typing import Any

from shortmem.plugins import LLMCaller, PromptTemplate
from shortmem.types import EvidenceBundle, TaskSpec
from shortmem.synth._json_tolerant import parse_json_object
from shortmem.synth.schema_designer import SchemaProposal


_SYSTEM_PROMPT = """You produce a PROMPT TEMPLATE for a PixelMem memory system.

Return ONLY JSON:
{
  "header": "one or two sentence role description ending with a newline",
  "instruction": "closing line telling the LLM exactly what to return",
  "output_format": "json_array" | "json_string" | "single_token",
  "token_budget": 500
}

Rules:
- The prompt will be assembled as:
    <header>
    Raw primitives:\\n<list>\\n
    Confirmed facts:\\n<list>\\n
    <optional: Computed hint:\\n<list>>
    <instruction>
- Keep header under 40 words, instruction under 20 words.
- output_format must match the task's declared query output shape.
"""


class SynthesizedPrompt(PromptTemplate):
    """Runtime for the slot-filled template."""

    def __init__(
        self,
        header: str,
        instruction: str,
        output_format: str = "json_array",
        token_budget: int = 500,
    ) -> None:
        self._header = header.rstrip() + "\n"
        self._instruction = instruction.strip()
        self._output_format = output_format
        self._token_budget = token_budget

    def build(
        self,
        task: TaskSpec,
        query_input: dict,
        evidence: EvidenceBundle,
    ) -> str:
        raw_lines = [
            f"  ({p.subject}, {p.relation}, {p.object}, {p.condition})"
            for p in evidence.raw_primitives[:60]
        ]
        strong_lines = [
            f"  ({p.subject}, {p.relation}, {p.object})"
            for p in evidence.strong[:40]
        ]
        hint = evidence.ordering_hint
        parts = [self._header]
        if raw_lines:
            parts.append("Raw primitives:\n" + "\n".join(raw_lines))
        if strong_lines:
            parts.append("Confirmed facts:\n" + "\n".join(strong_lines))
        if hint:
            parts.append("Computed hint: " + json.dumps(hint))
        if query_input:
            parts.append("Query: " + json.dumps(query_input)[:500])
        parts.append(self._instruction)
        return "\n\n".join(parts)

    def parse(self, completion: str, task: TaskSpec) -> Any:
        fmt = self._output_format
        if fmt == "json_array":
            m = re.search(r"\[.*\]", completion, re.DOTALL)
            if not m:
                raise ValueError("no JSON array in completion")
            parsed = json.loads(m.group(0))
            if not isinstance(parsed, list):
                raise ValueError("parsed JSON is not a list")
            return parsed
        if fmt == "json_string":
            m = re.search(r'"([^"]*)"', completion)
            if not m:
                raise ValueError("no quoted string in completion")
            return m.group(1)
        if fmt == "single_token":
            token = completion.strip().split()[0] if completion.strip() else ""
            if not token:
                raise ValueError("empty completion")
            return token
        raise ValueError(f"unknown output_format {fmt}")


class PromptSynth:
    def __init__(self, llm: LLMCaller) -> None:
        self._llm = llm

    def synthesize(
        self,
        schema: SchemaProposal,
        domain: str,
        description: str,
    ) -> SynthesizedPrompt:
        prompt = (
            _SYSTEM_PROMPT
            + "\n---\n"
            + f"Domain: {domain}\n"
            + f"Description: {description}\n"
            + f"Query output schema: {json.dumps(schema.query_output_schema)}\n"
        )
        raw, _tin, _tout = self._llm(prompt)
        return _parse_prompt_spec(raw)


def _parse_prompt_spec(raw: str) -> SynthesizedPrompt:
    data = parse_json_object(raw)
    return SynthesizedPrompt(
        header=str(data.get("header", "")),
        instruction=str(data.get("instruction", "Answer.")),
        output_format=str(data.get("output_format", "json_array")),
        token_budget=int(data.get("token_budget", 500)),
    )


__all__ = ["PromptSynth", "SynthesizedPrompt"]
