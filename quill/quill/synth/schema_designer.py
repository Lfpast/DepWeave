"""Stage 1 — Schema Designer.

Given a TaskCard (domain, description, few-shot), an LLMCaller produces a
proposed schema: relation vocabulary, condition channels, entity kinds,
and the expected query-output type.

This is the safest synthesis stage: the output is structured data (JSON),
not code. We validate shape before handing it downstream.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from quill.plugins import LLMCaller
from quill.types import TaskSpec
from quill.synth._json_tolerant import parse_json_object


@dataclass
class SchemaProposal:
    relations: list[str]
    conditions: list[str]
    entity_kinds: dict[str, str] = field(default_factory=dict)
    query_output_schema: dict = field(default_factory=dict)
    rationale: str = ""
    raw_response: str = ""

    def validate(self) -> None:
        if not self.relations:
            raise ValueError("schema must have at least one relation")
        if len(self.relations) > 20:
            # Pixel palette cap — docs/v5_plan.md §4.1
            raise ValueError(
                f"schema has {len(self.relations)} relations; V5 pixel "
                "palette supports up to 20 per shard"
            )
        if not self.conditions:
            raise ValueError("schema must have at least one condition channel")
        for r in self.relations:
            if not re.match(r"^[a-z][a-z0-9_]*$", r):
                raise ValueError(
                    f"relation '{r}' must be snake_case (a-z0-9_)"
                )


_SYSTEM_PROMPT = """You design schemas for a graph-based memory system called PixelMem.

Rules you MUST follow:
- Produce at most 20 relations (pixel palette cap).
- Separate orthogonal axes via the `conditions` channel, do NOT explode relations.
- Every relation must be snake_case (lowercase with underscores).
- Every derivable fact must be reachable by chaining declared relations.
- Entity kinds are abstract roles (e.g. "file", "symbol", "clause", "speaker").

Respond with ONLY a single JSON object, no commentary, matching:

{
  "relations": ["..."],
  "conditions": ["..."],
  "entity_kinds": {"<role>": "<format hint>"},
  "query_output_schema": {"type": "..."},
  "rationale": "one sentence explaining the design"
}
"""


def _build_user_prompt(task: TaskSpec) -> str:
    shots = task.few_shot[:3]
    shot_blocks = []
    for i, ex in enumerate(shots):
        shot_blocks.append(
            f"Example {i+1}:\n  input: {json.dumps(ex.input)[:400]}\n"
            f"  expected_output: {json.dumps(ex.expected_output)[:200]}"
        )
    shots_text = "\n\n".join(shot_blocks) if shot_blocks else "(none provided)"

    return (
        f"Domain: {task.domain}\n"
        f"Description: {task.description}\n"
        f"Input schema: {json.dumps(task.input_schema)}\n"
        f"Query: {json.dumps(task.query)}\n\n"
        f"Few-shot examples (truncated):\n{shots_text}\n\n"
        "Design a PixelMem schema for this task."
    )


class SchemaDesigner:
    """Calls the LLM to propose a schema, parses + validates."""

    def __init__(self, llm: LLMCaller) -> None:
        self._llm = llm

    def design(self, task: TaskSpec) -> SchemaProposal:
        prompt = _SYSTEM_PROMPT + "\n---\n" + _build_user_prompt(task)
        raw, _tin, _tout = self._llm(prompt)
        proposal = _parse_proposal(raw)
        proposal.validate()
        return proposal


def _parse_proposal(raw: str) -> SchemaProposal:
    data = parse_json_object(raw)
    return SchemaProposal(
        relations=list(data.get("relations", [])),
        conditions=list(data.get("conditions", [])),
        entity_kinds=dict(data.get("entity_kinds", {})),
        query_output_schema=dict(data.get("query_output_schema", {})),
        rationale=str(data.get("rationale", "")),
        raw_response=raw,
    )


__all__ = ["SchemaDesigner", "SchemaProposal"]
