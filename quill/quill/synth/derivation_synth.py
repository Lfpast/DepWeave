"""Stage 4 — Derivation rule synthesis.

LLM emits a JSON list of chain rules; we convert to ``DerivationRule`` and
reject anything that references a relation not in the schema.
"""

from __future__ import annotations

from quill.plugins import DerivationRule, LLMCaller
from quill.synth._json_tolerant import parse_json_object
from quill.synth.schema_designer import SchemaProposal


_SYSTEM_PROMPT = """You produce derivation rules for a PixelMem schema.

A DerivationRule chains primitive facts into a higher-level fact. Example
for a code-dependency schema:

  imports_symbol(X, Y) + defined_in(Y, Z) -> depends_on(X, Z)

Encoding rules you MUST follow:
- Each rule is: {"name": ..., "pattern": [...], "derived": [...], "confidence": 0.0..1.0}
- Pattern terms and derived term are [subject, relation, object, condition].
- Variables start with '?'. Same variable used twice binds to the same value.
- Every relation mentioned must exist in the schema.
- Confidence >= 0.70 means "strong" edge, shown to the LLM at query time.
- Produce between 1 and 10 rules. More = noisier, not better.

Return ONLY JSON:
{
  "rules": [
    {
      "name": "...",
      "pattern": [["?X","rel_a","?Y","?c"], ...],
      "derived": ["?X","derived_rel","?Z","channel"],
      "confidence": 0.9,
      "notes": "brief why"
    }
  ]
}
"""


class DerivationSynth:
    def __init__(self, llm: LLMCaller) -> None:
        self._llm = llm

    def synthesize(
        self,
        schema: SchemaProposal,
        domain: str,
        description: str,
    ) -> list[DerivationRule]:
        prompt = (
            _SYSTEM_PROMPT
            + "\n---\n"
            + f"Domain: {domain}\n"
            + f"Description: {description}\n"
            + f"Schema relations: {schema.relations}\n"
            + f"Schema conditions: {schema.conditions}\n"
        )
        raw, _tin, _tout = self._llm(prompt)
        return _parse_rules(raw, schema)


def _pad_tuple4(t) -> tuple[str, str, str, str]:
    """Coerce a 3- or 4-element fact pattern into 4-tuple form.

    3-tuples are treated as ``(subject, relation, object)`` with a wildcard
    condition — this matches what LLMs usually emit when they skip the
    channel slot.
    """
    if len(t) == 3:
        return (str(t[0]), str(t[1]), str(t[2]), "?_")
    if len(t) == 4:
        return (str(t[0]), str(t[1]), str(t[2]), str(t[3]))
    raise ValueError(f"pattern term must be 3- or 4-tuple, got {t}")


def _parse_rules(raw: str, schema: SchemaProposal) -> list[DerivationRule]:
    data = parse_json_object(raw)
    allowed_rel = set(schema.relations)
    rules: list[DerivationRule] = []
    for r in data.get("rules", []):
        pattern = [_pad_tuple4(p) for p in r.get("pattern", [])]
        derived_list = _pad_tuple4(list(r.get("derived", [])))
        # Relations on lhs: must be in schema if they're literals (not ?var).
        for p in pattern:
            rel = p[1]
            if not rel.startswith("?") and rel not in allowed_rel:
                raise ValueError(
                    f"rule '{r.get('name')}' references unknown relation '{rel}'"
                )
        rules.append(DerivationRule(
            name=str(r.get("name", f"r{len(rules)}")),
            pattern=pattern,  # type: ignore[arg-type]
            derived=derived_list,
            confidence=float(r.get("confidence", 0.8)),
            notes=str(r.get("notes", "")),
        ))
    if not rules:
        raise ValueError("no rules returned")
    if len(rules) > 10:
        raise ValueError(f"too many rules returned: {len(rules)} (cap is 10)")
    return rules


__all__ = ["DerivationSynth"]
