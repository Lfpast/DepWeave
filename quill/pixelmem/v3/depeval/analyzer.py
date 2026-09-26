"""Pipeline result analysis and error attribution.

Tracks detailed statistics for every pipeline run and classifies errors
into actionable categories.  This enables systematic debugging: instead
of staring at wrong answers, we know exactly *which stage* failed.

Error types:
  - ``"correct"``       — prediction matches ground truth.
  - ``"resolver"``      — import resolver failed to find a real edge.
  - ``"ambiguous"``     — edge existed but confidence was too low.
  - ``"decoder"``       — strong edges existed but decoder misordered.
  - ``"dup_output"``    — LLM produced duplicate file IDs in output.
  - ``"llm_mistake"``   — LLM explicitly changed a correct computed order.
  - ``"underdetermined"`` — too few constraints to determine a unique order.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .file_id_mapper import FileIDMapper
from .partial_order import PartialOrder


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Full result of a single pipeline run with instrumentation data."""

    # Predicted output
    file_ids: list[str]         # predicted file-ID order
    basenames: list[str]        # predicted basenames (derived from IDs)

    # Ground truth
    expected: list[str]         # expected basenames

    # Top-level correctness
    exact: bool                 # basenames == expected (exact match)

    # Pipeline statistics
    n_files: int                # number of input files
    n_candidates: int           # import candidates found by resolver
    n_strong_edges: int         # constraints with confidence > 0.7
    n_medium_edges: int         # constraints with confidence in (0.3, 0.7]
    n_weak_edges: int           # constraints with confidence <= 0.3
    n_ambiguous_pairs: int      # pairs with no or very weak evidence
    n_pairwise_queries: int     # LLM pairwise ranking calls made
    has_duplicates: bool        # True if input has duplicate basenames
    has_cycle: bool             # True if strong edges form a cycle
    n_constraint_violations: int  # strong constraints violated by final order
    tokens_used: int            # total LLM tokens (in + out)

    # Error attribution
    error_type: str             # see module docstring for categories


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_result(
    predicted_ids: list[str],
    expected_bn: list[str],
    evidence: Optional[dict] = None,
    partial: Optional[PartialOrder] = None,
    mapper: Optional[FileIDMapper] = None,
    *,
    n_candidates: int = 0,
    n_pairwise_queries: int = 0,
    tokens_used: int = 0,
) -> PipelineResult:
    """Build a PipelineResult with error attribution.

    Args:
        predicted_ids: Final file-ID order produced by the pipeline.
        expected_bn: Ground-truth basenames in correct order.
        evidence: Optional dict of edge stats from build_evidence_graph.
        partial: Optional PartialOrder used for constraint analysis.
        mapper: Optional FileIDMapper for converting IDs to basenames.
        n_candidates: Number of import candidates found.
        n_pairwise_queries: Number of pairwise LLM calls made.
        tokens_used: Total LLM tokens consumed.

    Returns:
        Fully populated PipelineResult.
    """
    # Derive basenames from IDs
    if mapper is not None:
        predicted_bn = mapper.ids_to_basenames(predicted_ids)
    else:
        predicted_bn = list(predicted_ids)

    n_files = len(expected_bn)
    exact = predicted_bn == expected_bn

    # Edge statistics from partial order
    n_strong = 0
    n_medium = 0
    n_weak = 0
    n_ambiguous = 0
    has_cycle = False
    n_violations = 0

    if partial is not None:
        strong_edges = partial.get_strong_constraints()
        n_strong = len(strong_edges)
        ambiguous_pairs = partial.get_ambiguous_pairs()
        n_ambiguous = len(ambiguous_pairs)
        has_cycle = partial.has_cycle()

        # Count medium and weak edges
        for (b, a), c in partial._constraints.items():
            if c > 0.7:
                pass  # already counted as strong
            elif c > 0.3:
                n_medium += 1
            else:
                n_weak += 1

        # Check violations of the predicted order
        violations = partial.check_violations(predicted_ids)
        n_violations = sum(1 for _, _, c in violations if c > 0.7)

    # Duplicate basenames in input?
    has_dups = len(set(expected_bn)) < len(expected_bn)

    # Determine error type
    error_type = _classify_error(
        exact=exact,
        predicted_bn=predicted_bn,
        expected_bn=expected_bn,
        predicted_ids=predicted_ids,
        n_strong=n_strong,
        n_ambiguous=n_ambiguous,
        n_violations=n_violations,
        has_cycle=has_cycle,
        has_dups=has_dups,
        partial=partial,
        mapper=mapper,
    )

    return PipelineResult(
        file_ids=predicted_ids,
        basenames=predicted_bn,
        expected=expected_bn,
        exact=exact,
        n_files=n_files,
        n_candidates=n_candidates,
        n_strong_edges=n_strong,
        n_medium_edges=n_medium,
        n_weak_edges=n_weak,
        n_ambiguous_pairs=n_ambiguous,
        n_pairwise_queries=n_pairwise_queries,
        has_duplicates=has_dups,
        has_cycle=has_cycle,
        n_constraint_violations=n_violations,
        tokens_used=tokens_used,
        error_type=error_type,
    )


def _classify_error(
    *,
    exact: bool,
    predicted_bn: list[str],
    expected_bn: list[str],
    predicted_ids: list[str],
    n_strong: int,
    n_ambiguous: int,
    n_violations: int,
    has_cycle: bool,
    has_dups: bool,
    partial: Optional[PartialOrder],
    mapper: Optional[FileIDMapper],
) -> str:
    """Classify the error type for a single prediction.

    The classification follows a priority chain: check the most specific
    (and actionable) failure mode first, then fall through to more
    general ones.
    """
    if exact:
        return "correct"

    # Check for duplicate IDs in the prediction (LLM repeated a file)
    if len(set(predicted_ids)) < len(predicted_ids):
        return "dup_output"

    # If there's a cycle in strong edges, the partial order is broken
    if has_cycle:
        return "resolver"

    # If the decoder violated strong constraints, it's a decoder bug
    if n_violations > 0:
        return "decoder"

    # If the partial order alone would give the right answer but the
    # LLM changed it, that's an LLM mistake
    if partial is not None:
        topo = partial.decode_total_order()
        if mapper is not None:
            topo_bn = mapper.ids_to_basenames(topo)
        else:
            topo_bn = topo
        if topo_bn == expected_bn and predicted_bn != expected_bn:
            return "llm_mistake"

    # If the correct pair-orderings exist but confidence was too low
    # to be strong, it's an ambiguity problem
    if n_ambiguous > 0 and n_strong < len(expected_bn) - 1:
        return "ambiguous"

    # If we have very few edges at all, the resolver missed connections
    if n_strong == 0:
        return "resolver"

    # If there are enough strong edges but the answer is still wrong,
    # the problem is under-determined (e.g. multiple valid topo orders)
    return "underdetermined"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def export_summary(results: list[PipelineResult]) -> dict:
    """Aggregate a list of PipelineResults into a summary report.

    Returns:
        Dict with keys:
        - ``total``: number of results
        - ``accuracy``: fraction exact-match
        - ``error_distribution``: ``{error_type: count}``
        - ``by_file_count``: ``{n_files: {total, correct, accuracy}}``
        - ``dup_stats``: ``{dup_total, dup_correct, nondup_total, nondup_correct}``
        - ``avg_pairwise_queries``: mean pairwise LLM calls per question
        - ``avg_tokens``: mean total tokens per question
        - ``avg_strong_edges``: mean strong edges per question
        - ``avg_ambiguous_pairs``: mean ambiguous pairs per question
    """
    if not results:
        return {"total": 0, "accuracy": 0.0}

    total = len(results)
    correct = sum(1 for r in results if r.exact)

    # Error distribution
    error_dist = Counter(r.error_type for r in results)

    # Accuracy by file count
    by_count: dict[int, dict] = {}
    for r in results:
        bucket = by_count.setdefault(r.n_files, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if r.exact:
            bucket["correct"] += 1
    for bucket in by_count.values():
        bucket["accuracy"] = (
            bucket["correct"] / bucket["total"] if bucket["total"] > 0 else 0.0
        )

    # Duplicate vs non-duplicate
    dup_results = [r for r in results if r.has_duplicates]
    nondup_results = [r for r in results if not r.has_duplicates]

    return {
        "total": total,
        "accuracy": correct / total,
        "error_distribution": dict(error_dist.most_common()),
        "by_file_count": {
            k: v for k, v in sorted(by_count.items())
        },
        "dup_stats": {
            "dup_total": len(dup_results),
            "dup_correct": sum(1 for r in dup_results if r.exact),
            "nondup_total": len(nondup_results),
            "nondup_correct": sum(1 for r in nondup_results if r.exact),
        },
        "avg_pairwise_queries": (
            sum(r.n_pairwise_queries for r in results) / total
        ),
        "avg_tokens": sum(r.tokens_used for r in results) / total,
        "avg_strong_edges": sum(r.n_strong_edges for r in results) / total,
        "avg_ambiguous_pairs": (
            sum(r.n_ambiguous_pairs for r in results) / total
        ),
    }
