"""ExtractorTester — quality check for a synthesized Extractor.

The synthesis loop develops → tests → refines extractors in a tight inner
loop. This module is the "test" half: given a (possibly just-synthesized)
Extractor and the TaskCard's few-shot documents, it runs the extractor and
reports whether it meets the TaskCard's success signals, plus actionable
feedback the LLM can consume on the next iteration.

Success signals are declared on the TaskCard under
``options.extractor_expectations``::

    options:
      extractor_expectations:
        min_primitives_per_doc: 2
        required_relations: ["imports_symbol"]
        min_total_primitives: 12
        min_cross_doc_edges: 1    # primitives whose object references another doc
        max_primitives_per_doc: 200
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from shortmem.plugins import Extractor
from shortmem.types import Example, Primitive, TaskSpec


@dataclass
class ExtractorTestResult:
    passed: bool
    n_primitives: int
    n_docs: int
    relation_counts: dict           # {relation: count}
    per_doc_primitives: dict        # {doc_id: count}
    cross_doc_edges: int            # primitives where object references another doc
    samples: list[tuple]            # first 20 primitives as 4-tuples
    issues: list[str] = field(default_factory=list)
    # Human-readable feedback block for the LLM refiner
    feedback: str = ""


def _object_references_other_doc(
    prim: Primitive,
    doc_ids: set[str],
    doc_basenames: set[str],
) -> bool:
    """Heuristic: does the primitive's object name another document?"""
    obj = prim.object or ""
    if "@" in obj:
        tail = obj.split("@", 1)[1]
        return tail in doc_ids or tail in doc_basenames
    bn = obj.split("/")[-1].split("\\")[-1]
    return obj in doc_ids or bn in doc_basenames


def _docs_from_example(ex: Example) -> dict[str, str]:
    """Pull ``{doc_id: text}`` out of any reasonable TaskCard input shape."""
    src = ex.input or {}
    if "documents" in src and isinstance(src["documents"], dict):
        return {str(k): str(v) for k, v in src["documents"].items()}
    if "file_contents" in src and isinstance(src["file_contents"], dict):
        return {str(k): str(v) for k, v in src["file_contents"].items()}
    # Fallback: treat input dict as docs.
    return {str(k): str(v) for k, v in src.items() if isinstance(v, str)}


def _doc_basenames(doc_ids) -> set[str]:
    return {d.split("/")[-1].split("\\")[-1] for d in doc_ids}


def test_extractor(
    extractor: Extractor,
    task: TaskSpec,
    examples: list[Example],
) -> ExtractorTestResult:
    """Run ``extractor`` over the docs in ``examples`` and grade the output.

    Thresholds come from ``task.options['extractor_expectations']``; any
    missing key uses a sensible default.
    """
    exp = dict(task.options.get("extractor_expectations", {}) or {})
    min_per_doc = int(exp.get("min_primitives_per_doc", 1))
    max_per_doc = int(exp.get("max_primitives_per_doc", 800))
    min_total = int(exp.get("min_total_primitives", 0))
    required_rels: list[str] = list(exp.get("required_relations", []))
    min_cross_doc = int(exp.get("min_cross_doc_edges", 0))

    all_primitives: list[Primitive] = []
    per_doc_counts: dict[str, int] = {}
    per_doc_seen: set[str] = set()

    for ex in examples:
        docs = _docs_from_example(ex)
        per_doc_seen.update(docs.keys())
        try:
            prims = list(extractor.extract(docs))
        except Exception as e:
            return ExtractorTestResult(
                passed=False,
                n_primitives=0,
                n_docs=len(docs),
                relation_counts={},
                per_doc_primitives={},
                cross_doc_edges=0,
                samples=[],
                issues=[f"extractor raised: {type(e).__name__}: {e}"],
                feedback=(
                    "The extractor raised an exception when run on the few-shot "
                    f"documents:\n  {type(e).__name__}: {e}\n"
                    "Revise the patterns so extract() runs cleanly on all documents."
                ),
            )
        all_primitives.extend(prims)
        by_doc: dict[str, int] = Counter()
        for p in prims:
            doc = (p.provenance or {}).get("doc_id")
            if doc is None:
                # Subject often == doc_id for file-level primitives.
                doc = p.subject
            by_doc[doc] += 1
        for d in docs:
            per_doc_counts[d] = per_doc_counts.get(d, 0) + by_doc.get(d, 0)

    rel_counter: Counter = Counter(p.relation for p in all_primitives)

    doc_ids = set(per_doc_seen)
    doc_bn = _doc_basenames(doc_ids)
    cross_doc = sum(
        1 for p in all_primitives
        if _object_references_other_doc(p, doc_ids, doc_bn)
    )

    issues: list[str] = []
    if len(all_primitives) < min_total:
        issues.append(
            f"too few primitives overall ({len(all_primitives)} < {min_total})"
        )
    for d, c in per_doc_counts.items():
        if c < min_per_doc:
            issues.append(
                f"document '{d}' produced {c} primitives "
                f"(expected >= {min_per_doc})"
            )
        if c > max_per_doc:
            issues.append(
                f"document '{d}' produced {c} primitives "
                f"(max {max_per_doc}; patterns too loose — emitting line-by-line?)"
            )
    for rel in required_rels:
        if rel_counter.get(rel, 0) == 0:
            issues.append(f"missing required relation '{rel}'")
    if cross_doc < min_cross_doc:
        issues.append(
            f"only {cross_doc} cross-document edges "
            f"(expected >= {min_cross_doc}); "
            "the extractor needs to emit primitives whose object names "
            "another document."
        )

    passed = len(issues) == 0

    samples = [p.as_tuple() for p in all_primitives[:20]]
    feedback = _build_feedback(
        passed=passed,
        n_primitives=len(all_primitives),
        n_docs=len(doc_ids),
        per_doc_counts=per_doc_counts,
        rel_counter=rel_counter,
        cross_doc=cross_doc,
        samples=samples,
        issues=issues,
        expectations=exp,
    )

    return ExtractorTestResult(
        passed=passed,
        n_primitives=len(all_primitives),
        n_docs=len(doc_ids),
        relation_counts=dict(rel_counter),
        per_doc_primitives=per_doc_counts,
        cross_doc_edges=cross_doc,
        samples=samples,
        issues=issues,
        feedback=feedback,
    )


def _build_feedback(
    *,
    passed: bool,
    n_primitives: int,
    n_docs: int,
    per_doc_counts: dict,
    rel_counter: Counter,
    cross_doc: int,
    samples: list,
    issues: list[str],
    expectations: dict,
) -> str:
    lines = []
    verdict = "PASSED" if passed else "FAILED"
    lines.append(f"Extractor test: {verdict}")
    lines.append(f"  docs: {n_docs}  primitives: {n_primitives}  "
                 f"cross-doc edges: {cross_doc}")
    if rel_counter:
        rel_summary = ", ".join(
            f"{r}={c}" for r, c in rel_counter.most_common(8)
        )
        lines.append(f"  relations: {rel_summary}")
    if per_doc_counts:
        per_doc_brief = ", ".join(
            f"{d.split('/')[-1]}={c}"
            for d, c in list(per_doc_counts.items())[:6]
        )
        lines.append(f"  per-doc counts (first 6): {per_doc_brief}")
    if samples:
        lines.append("  first 5 primitives:")
        for s in samples[:5]:
            lines.append(f"    {s}")
    if expectations:
        lines.append(f"  expectations: {expectations}")
    if issues:
        lines.append("Issues:")
        for i in issues:
            lines.append(f"  - {i}")
    if not passed:
        lines.append(
            "Revise the extraction patterns to address the issues above. "
            "Keep the schema's relations and conditions; adjust only the "
            "regexes and field_mapping."
        )
    return "\n".join(lines)
