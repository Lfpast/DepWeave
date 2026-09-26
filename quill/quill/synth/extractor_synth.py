"""Stage 2 — Extractor synthesis (template-driven).

We deliberately DO NOT let the LLM write free-form Python here. Instead the
LLM fills slots in a pre-written template: regex patterns + a relation map.
This covers a broad class of text-shaped domains (logs, dialogue turns,
legal clauses, medical notes) without the blast radius of code exec.

Free-form code synthesis is P4 in docs/v5_plan.md; it will require the
subprocess sandbox layer, which is not part of the initial V5 commit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from quill.core.plugins import Extractor, LLMCaller
from quill.core.types import Primitive
from quill.synth._json_tolerant import parse_json_object
from quill.synth.schema_designer import SchemaProposal


@dataclass
class ExtractionPattern:
    """One pattern: apply ``regex`` to each doc, emit a Primitive per match.

    Named groups in the regex map to the primitive's slots via
    ``field_mapping``. E.g. ``{"subject": "s", "object": "o"}`` pulls the
    ``s`` and ``o`` groups out of each match.
    """

    name: str
    regex: str
    relation: str
    condition: str = ""
    field_mapping: dict[str, str] = field(default_factory=dict)
    default_subject: Optional[str] = None


_SYSTEM_PROMPT = """You translate a PixelMem schema into a set of regex extraction patterns.

You MUST return ONLY a JSON object with this shape:

{
  "patterns": [
    {
      "name": "short_name",
      "regex": "Python regex with named groups",
      "relation": "relation_name_from_schema",
      "condition": "condition_channel_from_schema",
      "field_mapping": {"subject": "group_name", "object": "group_name"},
      "default_subject": null
    }
  ],
  "notes": "one sentence"
}

Rules:
- Every relation you emit MUST appear in the schema.
- Every condition you emit MUST appear in the schema.
- Regexes must use Python re syntax (?P<name>...) for named groups.
- Prefer MANY small precise patterns over one complex one.
- Do NOT try to be clever with lookaheads or recursion.
- If the pattern should emit a cross-document edge, set field_mapping so
  that the `object` group is the name of the OTHER document (a filename,
  a basename, a class name that will be matched against another doc, etc.)
- Do not emit a generic catch-all regex like ".+" — narrow patterns only.
"""


_REVISION_PROMPT = """You previously proposed extraction patterns that
did not pass the quality check. Review the diagnostic feedback below and
return a REVISED JSON object (same shape as before) that fixes the issues.

Only change what the feedback asks for. Keep the schema's relations and
conditions. Do not explain — return ONLY the revised JSON.

--- Diagnostic feedback ---
{feedback}

--- Your previous extractor patterns ---
{previous_patterns}

--- Sample documents (to calibrate regexes against) ---
{sample_docs}
"""


class TemplateExtractor(Extractor):
    """Extractor defined by a list of ExtractionPatterns.

    One scan per pattern per document; each match becomes one Primitive.
    """

    def __init__(
        self,
        patterns: list[ExtractionPattern],
        doc_subject_formatter: str = "{doc_id}",
    ) -> None:
        self._patterns = patterns
        self._compiled = [(p, re.compile(p.regex, re.MULTILINE)) for p in patterns]
        self._doc_subject_formatter = doc_subject_formatter

    def extract(self, documents: dict[str, str], **kwargs: Any) -> list[Primitive]:
        out: list[Primitive] = []
        for doc_id, text in documents.items():
            doc_subject = self._doc_subject_formatter.format(doc_id=doc_id)
            for pattern, compiled in self._compiled:
                for m in compiled.finditer(text or ""):
                    groups = m.groupdict()

                    def resolve(slot: str, fallback: str = "") -> str:
                        key = pattern.field_mapping.get(slot)
                        if key is None:
                            return fallback
                        val = groups.get(key)
                        return val if val is not None else fallback

                    subject = resolve("subject", pattern.default_subject or doc_subject)
                    obj = resolve("object", "")
                    if not obj:
                        continue

                    out.append(Primitive(
                        subject=subject,
                        relation=pattern.relation,
                        object=obj,
                        condition=pattern.condition or resolve("condition", ""),
                        provenance={
                            "pattern": pattern.name,
                            "doc_id": doc_id,
                        },
                    ))
        return out


class ExtractorSynth:
    """Calls the LLM to produce ExtractionPatterns from a SchemaProposal."""

    def __init__(self, llm: LLMCaller) -> None:
        self._llm = llm

    def synthesize(
        self,
        schema: SchemaProposal,
        domain: str,
        description: str,
        few_shot_sample: Optional[str] = None,
    ) -> TemplateExtractor:
        prompt = self._initial_prompt(schema, domain, description, few_shot_sample)
        raw, _tin, _tout = self._llm(prompt)
        patterns = _parse_patterns(raw, schema)
        return TemplateExtractor(patterns)

    # ------------------------------------------------------------------
    # Develop → test → revise inner loop
    # ------------------------------------------------------------------

    def synthesize_with_feedback(
        self,
        schema: SchemaProposal,
        domain: str,
        description: str,
        task,                         # TaskSpec (avoid circular import)
        examples,                     # list[Example] (few-shot docs)
        max_iterations: int = 3,
        debug: bool = False,
        few_shot_sample: Optional[str] = None,
    ):
        """Synthesize an Extractor, test it, and revise up to N times.

        Returns ``(extractor, test_result, trace)`` where ``trace`` is a
        list of per-iteration ``ExtractorTestResult`` objects.
        """
        # Deferred import to avoid a cycle (extractor_tester imports Extractor).
        from quill.synth.extractor_tester import test_extractor

        sample = few_shot_sample or _first_doc_sample(examples)
        prev_patterns_json = ""
        extractor: Optional[TemplateExtractor] = None
        trace = []

        for it in range(max_iterations):
            if it == 0:
                prompt = self._initial_prompt(schema, domain, description, sample)
            else:
                last = trace[-1]
                prompt = _REVISION_PROMPT.format(
                    feedback=last.feedback,
                    previous_patterns=prev_patterns_json[:3000],
                    sample_docs=sample[:2000],
                )

            try:
                raw, _tin, _tout = self._llm(prompt)
                patterns = _parse_patterns(raw, schema)
                extractor = TemplateExtractor(patterns)
                prev_patterns_json = _patterns_to_json(patterns)
                if debug:
                    pat_names = [p.name for p in patterns]
                    print(f"[extractor-synth] iter {it}: patterns={pat_names}")
            except Exception as e:
                if debug:
                    print(f"[extractor-synth] iter {it} synth failed: {e}")
                # Can't test what didn't parse; construct a synthetic
                # failure feedback and retry.
                from quill.synth.extractor_tester import ExtractorTestResult
                trace.append(ExtractorTestResult(
                    passed=False,
                    n_primitives=0, n_docs=0,
                    relation_counts={}, per_doc_primitives={},
                    cross_doc_edges=0, samples=[],
                    issues=[f"extractor synth raised: {type(e).__name__}: {e}"],
                    feedback=(
                        f"Your previous response could not be parsed: "
                        f"{type(e).__name__}: {e}\n"
                        "Return ONLY a JSON object with a 'patterns' list, "
                        "each pattern having name / regex / relation / "
                        "condition / field_mapping."
                    ),
                ))
                continue

            result = test_extractor(extractor, task, examples)
            trace.append(result)
            if debug:
                print(f"[extractor-synth] iter {it}: "
                      f"passed={result.passed}  "
                      f"n={result.n_primitives}  cross={result.cross_doc_edges}")
                for issue in result.issues:
                    print(f"  issue: {issue}")
            if result.passed:
                break

        # Fall through: return the best (last) extractor + trace.
        return extractor, trace[-1] if trace else None, trace

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _initial_prompt(
        self,
        schema: SchemaProposal,
        domain: str,
        description: str,
        few_shot_sample: Optional[str],
    ) -> str:
        prompt = (
            _SYSTEM_PROMPT
            + "\n---\n"
            + f"Domain: {domain}\n"
            + f"Description: {description}\n"
            + "Schema: " + json.dumps({
                "relations": schema.relations,
                "conditions": schema.conditions,
            }) + "\n"
        )
        if few_shot_sample:
            prompt += f"\nSample input:\n{few_shot_sample[:2000]}\n"
        return prompt


def _first_doc_sample(examples) -> str:
    """Concatenate the first example's docs for regex calibration."""
    if not examples:
        return ""
    ex = examples[0]
    src = ex.input or {}
    docs = src.get("documents") or src.get("file_contents") or {}
    if not docs:
        # Maybe the input itself is a doc-dict.
        docs = {k: v for k, v in src.items() if isinstance(v, str)}
    blocks = []
    for doc_id, text in list(docs.items())[:3]:
        blocks.append(f"# ====== {doc_id} ======\n{str(text)[:1500]}")
    return "\n\n".join(blocks)


def _patterns_to_json(patterns: list["ExtractionPattern"]) -> str:
    return json.dumps({
        "patterns": [
            {
                "name": p.name,
                "regex": p.regex,
                "relation": p.relation,
                "condition": p.condition,
                "field_mapping": p.field_mapping,
                "default_subject": p.default_subject,
            }
            for p in patterns
        ],
    }, indent=2)


def _parse_patterns(
    raw: str,
    schema: SchemaProposal,
) -> list[ExtractionPattern]:
    data = parse_json_object(raw)
    patterns = []
    allowed_rel = set(schema.relations)
    allowed_cond = set(schema.conditions) | {""}
    for p in data.get("patterns", []):
        rel = p.get("relation")
        cond = p.get("condition", "")
        if rel not in allowed_rel:
            raise ValueError(
                f"pattern '{p.get('name')}' uses relation '{rel}' not in schema"
            )
        if cond not in allowed_cond:
            raise ValueError(
                f"pattern '{p.get('name')}' uses condition '{cond}' not in schema"
            )
        try:
            re.compile(p["regex"], re.MULTILINE)
        except re.error as err:
            raise ValueError(
                f"pattern '{p.get('name')}' regex does not compile: {err}"
            ) from err
        patterns.append(ExtractionPattern(
            name=str(p.get("name", f"p{len(patterns)}")),
            regex=str(p["regex"]),
            relation=str(rel),
            condition=str(cond),
            field_mapping=dict(p.get("field_mapping", {})),
            default_subject=p.get("default_subject"),
        ))
    if not patterns:
        raise ValueError("no patterns returned")
    return patterns


__all__ = ["ExtractorSynth", "TemplateExtractor", "ExtractionPattern"]
