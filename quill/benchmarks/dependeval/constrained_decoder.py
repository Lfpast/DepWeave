"""Constrained decoder — produces a total order that GUARANTEES all
strong constraints are respected.

This is the core improvement over bayesian_order.py.  The old approach
blended everything into soft scores and sorted, then tried to patch
violations after the fact.  That works most of the time, but the
"patching" step can fail when violations are non-adjacent or when
fixing one violation creates another.

The constrained decoder works differently:

1. Get topological layers from the PartialOrder (strong edges only).
2. Cross-layer order is FIXED — layer 0 before layer 1 before layer 2 …
3. Within each layer, files have no strong ordering between them, so we
   are free to arrange them using pairwise scores or heuristics.
4. This construction guarantees zero strong-constraint violations by
   design, not by patching.

The ``enforce_hard_constraints`` function is provided as a safety net
for orderings produced by other methods (e.g. LLM-predicted orders)
that may violate strong edges.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional, Protocol

from benchmarks.dependeval.partial_order import PartialOrder

if TYPE_CHECKING:
    # FileIDMapper may not exist yet — keep it optional / type-only.
    from benchmarks.dependeval.file_id_mapper import FileIDMapper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decode_order(
    partial: PartialOrder,
    pairwise_scores: Optional[dict[tuple[str, str], float]] = None,
    mapper: Optional["FileIDMapper"] = None,
) -> list[str]:
    """Produce a total order respecting ALL strong constraints.

    Algorithm:
        1. Compute topological layers from the strong-edge DAG.
        2. Layers define a fixed inter-layer ordering.
        3. Within each layer, sort by:
           a. ``pairwise_scores`` (if provided) — wins-above-losses.
           b. Heuristic tie-break when no pairwise data exists.
        4. Concatenate layers → total order.

    Args:
        partial: PartialOrder with constraints already populated.
        pairwise_scores: Optional mapping ``(fid_a, fid_b) → score``
            where a positive score means *a before b* is preferred.
            Scores should be symmetric-ish: ``scores[(a,b)] ≈ -scores[(b,a)]``.
        mapper: Optional FileIDMapper for resolving IDs back to
            filenames (used in heuristic tie-breaking).

    Returns:
        List of file IDs in dependency order (base files first).
    """
    if partial.has_cycle():
        # Fall back to best-effort decode — layers will have a catch-all
        # bucket for cycle members, which we sort by heuristic.
        pass  # topological_layers handles this gracefully

    layers = partial.topological_layers()
    result: list[str] = []

    for layer in layers:
        if len(layer) <= 1:
            result.extend(layer)
            continue

        sorted_layer = _sort_within_layer(
            layer, partial, pairwise_scores, mapper,
        )
        result.extend(sorted_layer)

    return result


def enforce_hard_constraints(
    order: list[str],
    partial: PartialOrder,
) -> list[str]:
    """Post-process an arbitrary order to fix all strong-constraint violations.

    Strategy: repeatedly scan for violations and move the offending
    "before" node to just before the "after" node.  Since the strong
    constraints form a DAG (checked by ``partial.has_cycle()``), this
    process terminates.

    If a cycle exists, we do at most ``N^2`` passes and return the
    best result we can.

    Args:
        order: An ordering that may violate strong constraints.
        partial: The PartialOrder defining constraints.

    Returns:
        A new list with (ideally) zero strong-constraint violations.
    """
    result = list(order)
    strong = partial.get_strong_constraints()
    if not strong:
        return result

    max_passes = len(result) ** 2  # safety bound
    for _ in range(max_passes):
        pos = {fid: i for i, fid in enumerate(result)}
        violation_found = False

        for before, after in strong:
            if before not in pos or after not in pos:
                continue
            if pos[before] > pos[after]:
                # before is too late — pull it to just before after
                result.remove(before)
                new_pos = result.index(after)
                result.insert(new_pos, before)
                violation_found = True
                break  # restart scan after mutation

        if not violation_found:
            break

    return result


def validate_output(
    order: list[str],
    partial: PartialOrder,
) -> tuple[bool, list[str]]:
    """Check ALL constraints (not just strong) against the given order.

    Returns:
        ``(is_valid, descriptions)`` where *is_valid* is True only if
        zero strong constraints are violated.  The descriptions list
        includes info about both strong and weak violations.
    """
    violations = partial.check_violations(order)
    if not violations:
        return True, []

    descriptions: list[str] = []
    has_strong_violation = False

    for before, after, conf in violations:
        severity = _severity_label(conf)
        desc = (
            f"{severity}: {before!r} should precede {after!r} "
            f"(conf={conf:.2f})"
        )
        descriptions.append(desc)
        if conf > 0.7:
            has_strong_violation = True

    is_valid = not has_strong_violation
    return is_valid, descriptions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sort_within_layer(
    layer: list[str],
    partial: PartialOrder,
    pairwise_scores: Optional[dict[tuple[str, str], float]],
    mapper: Optional["FileIDMapper"],
) -> list[str]:
    """Order files within a single topological layer.

    Within a layer, no strong ordering exists, so we use:
    1. Pairwise scores (if available) — net wins.
    2. Heuristic tie-break based on file naming patterns.
    """
    if len(layer) <= 1:
        return list(layer)

    # Compute a score for each file: positive = should come earlier.
    score: dict[str, float] = {fid: 0.0 for fid in layer}

    # --- pairwise evidence ---
    if pairwise_scores:
        for i, a in enumerate(layer):
            for b in layer[i + 1 :]:
                s_ab = pairwise_scores.get((a, b), 0.0)
                s_ba = pairwise_scores.get((b, a), 0.0)
                net = s_ab - s_ba
                score[a] += net
                score[b] -= net

    # --- weak/medium constraints from the partial order itself ---
    for i, a in enumerate(layer):
        for b in layer[i + 1 :]:
            c_ab = partial.get_confidence(a, b)  # a before b
            c_ba = partial.get_confidence(b, a)  # b before a
            if c_ab is not None:
                score[a] += c_ab * 0.5  # half weight — these are soft
                score[b] -= c_ab * 0.5
            if c_ba is not None:
                score[b] += c_ba * 0.5
                score[a] -= c_ba * 0.5

    # --- heuristic tie-break ---
    for fid in layer:
        name = _resolve_name(fid, mapper)
        score[fid] += _heuristic_score(name, partial, fid)

    # Sort: lower score = comes first (more "base").
    return sorted(layer, key=lambda fid: (score[fid], fid))


def _resolve_name(fid: str, mapper: Optional["FileIDMapper"]) -> str:
    """Get a filename string for heuristic analysis.

    If a mapper is available, use it.  Otherwise, the file ID itself
    may contain the filename (e.g. ``"utils.py"`` or ``"F3:utils.py"``).
    """
    if mapper is not None:
        try:
            return mapper.id_to_file(fid)  # type: ignore[union-attr]
        except (AttributeError, KeyError):
            pass

    # Best-effort: if the ID contains a recognisable filename, use it.
    if ".py" in fid:
        return fid.split("/")[-1]
    return fid


def _heuristic_score(name: str, partial: PartialOrder, fid: str) -> float:
    """Assign a small heuristic bias based on filename patterns.

    Negative = earlier (more base).  Positive = later.

    These biases are intentionally small so that actual evidence
    (pairwise scores, weak constraints) dominates.
    """
    s = 0.0

    # test_ files go last — they almost always depend on non-test code.
    basename = name.split("/")[-1] if "/" in name else name
    if basename.startswith("test_"):
        s += 5.0
    elif basename == "conftest.py":
        s += 4.0

    # __init__.py usually aggregates — goes late, but not always.
    if basename == "__init__.py":
        s += 2.0

    # Files with "base", "core", "utils", "common" in the name
    # are more likely to be foundational.
    lower = basename.lower()
    if re.search(r"(base|core|utils|common|constants|config)", lower):
        s -= 1.5

    # Files with "main", "app", "cli", "run" are entry points — late.
    if re.search(r"(main|app|cli|run|server)", lower):
        s += 2.5

    # Out-degree in strong graph as a proxy for importance.
    # More dependents → more foundational → should come earlier.
    strong = partial.get_strong_constraints()
    out_count = sum(1 for (b, _a) in strong if b == fid)
    s -= out_count * 0.3

    return s


def _severity_label(confidence: float) -> str:
    """Human-readable severity label for a constraint violation."""
    if confidence > 0.7:
        return "STRONG VIOLATION"
    if confidence > 0.3:
        return "MEDIUM violation"
    return "weak violation"
