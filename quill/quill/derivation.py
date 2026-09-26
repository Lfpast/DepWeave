"""Default rule-matching derivation engine.

The rule format is intentionally small: each DerivationRule is a conjunction
of fact-shape patterns that share variables. A match over the primitive set
produces one derived fact per consistent variable binding.

This is not a full Datalog engine — we deliberately disallow recursion and
cap fan-out, because the V5 trust model assumes derivation is fast and
bounded so it can run per-query without extra LLM calls.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from quill.plugins import DerivationEngine, DerivationRule
from quill.types import EvidenceBundle, Primitive, TaskSpec


_MAX_DERIVED_PER_RULE = 5000  # guard against pattern-match blow-up


def _is_var(token: str) -> bool:
    return isinstance(token, str) and token.startswith("?")


def _match_term(
    pattern_term: str,
    actual: str,
    bindings: dict[str, str],
) -> Optional[dict[str, str]]:
    """Return updated bindings if term unifies, else None."""
    if _is_var(pattern_term):
        existing = bindings.get(pattern_term)
        if existing is None:
            new = dict(bindings)
            new[pattern_term] = actual
            return new
        if existing == actual:
            return bindings
        return None
    # Literal — must match exactly.
    return bindings if pattern_term == actual else None


def _match_pattern(
    pattern: tuple[str, str, str, str],
    prim: Primitive,
    bindings: dict[str, str],
) -> Optional[dict[str, str]]:
    b = bindings
    for pt, av in zip(pattern, prim.as_tuple()):
        b = _match_term(pt, av, b)  # type: ignore[assignment]
        if b is None:
            return None
    return b


def _index_primitives(
    primitives: list[Primitive],
) -> dict[str, list[Primitive]]:
    """Index by relation for O(rule) match instead of O(rule * |primitives|)."""
    idx: dict[str, list[Primitive]] = defaultdict(list)
    for p in primitives:
        idx[p.relation].append(p)
    return idx


def _instantiate(
    template: tuple[str, str, str, str],
    bindings: dict[str, str],
) -> Optional[tuple[str, str, str, str]]:
    out = []
    for t in template:
        if _is_var(t):
            v = bindings.get(t)
            if v is None:
                return None
            out.append(v)
        else:
            out.append(t)
    return tuple(out)  # type: ignore[return-value]


class DefaultDerivationEngine(DerivationEngine):
    """Non-recursive rule matcher. Splits derivations into strong/ambiguous by
    the rule's confidence vs the threshold in ``task.options``.
    """

    def __init__(
        self,
        strong_threshold: float = 0.70,
        max_per_rule: int = _MAX_DERIVED_PER_RULE,
    ) -> None:
        self._strong_threshold = strong_threshold
        self._max_per_rule = max_per_rule

    def derive(
        self,
        primitives: list[Primitive],
        rules: list[DerivationRule],
        task: TaskSpec,
    ) -> EvidenceBundle:
        by_rel = _index_primitives(primitives)
        strong: list[Primitive] = []
        ambiguous: list[Primitive] = []
        seen: set[tuple[str, str, str, str]] = set()

        for rule in rules:
            matches = self._match_rule(rule, by_rel)
            for bindings in matches[: self._max_per_rule]:
                derived_tuple = _instantiate(rule.derived, bindings)
                if derived_tuple is None:
                    continue
                if derived_tuple in seen:
                    continue
                seen.add(derived_tuple)
                fact = Primitive(
                    subject=derived_tuple[0],
                    relation=derived_tuple[1],
                    object=derived_tuple[2],
                    condition=derived_tuple[3],
                    provenance={"rule": rule.name, "confidence": rule.confidence},
                )
                if rule.confidence >= self._strong_threshold:
                    strong.append(fact)
                else:
                    ambiguous.append(fact)

        return EvidenceBundle(
            strong=strong,
            ambiguous=ambiguous,
            raw_primitives=[],
            ordering_hint=[],
            metadata={
                "n_strong": len(strong),
                "n_ambiguous": len(ambiguous),
                "n_rules_fired": sum(1 for r in rules if r.name in {f.provenance["rule"] for f in strong + ambiguous if f.provenance}),
            },
        )

    # ----- internals -----

    def _match_rule(
        self,
        rule: DerivationRule,
        by_rel: dict[str, list[Primitive]],
    ) -> list[dict[str, str]]:
        if not rule.pattern:
            return []
        return list(self._match_conjunction(rule.pattern, 0, {}, by_rel))

    def _match_conjunction(
        self,
        pattern: list[tuple[str, str, str, str]],
        idx: int,
        bindings: dict[str, str],
        by_rel: dict[str, list[Primitive]],
    ) -> Iterable[dict[str, str]]:
        if idx >= len(pattern):
            yield bindings
            return

        term = pattern[idx]
        rel_slot = term[1]
        candidates: list[Primitive]
        if _is_var(rel_slot):
            candidates = []
            for rel_list in by_rel.values():
                candidates.extend(rel_list)
        else:
            candidates = by_rel.get(rel_slot, [])

        for prim in candidates:
            new_b = _match_pattern(term, prim, bindings)
            if new_b is None:
                continue
            yield from self._match_conjunction(pattern, idx + 1, new_b, by_rel)


__all__ = ["DefaultDerivationEngine"]
