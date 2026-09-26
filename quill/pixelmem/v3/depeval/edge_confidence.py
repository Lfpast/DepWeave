"""Build a scored dependency graph from import resolution and PixelMem evidence.

Combines static import analysis (``import_resolver``) with optional
PixelMem pixel-matrix ``depends_on`` triples to produce a unified
list of ``DependencyEvidence`` edges with calibrated confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pixelmem.v3.depeval.file_id_mapper import FileIDMapper
from pixelmem.v3.depeval.import_resolver import resolve_all_imports, ImportCandidate

if TYPE_CHECKING:
    from pixelmem.shard_manager import ShardManager


# ── Data class ─────────────────────────────────────────────────────


@dataclass
class DependencyEvidence:
    """One directed dependency edge with confidence metadata."""
    source_id: str      # e.g. "F0"
    target_id: str      # e.g. "F1"
    raw_import: str      # original import text (or "pixel:depends_on")
    score: float         # calibrated 0-1
    reason: str          # human-readable provenance
    ambiguity: int       # number of competing candidates (0 for pixel evidence)


# ── Graph construction ─────────────────────────────────────────────


def _candidates_to_evidence(
    candidates: list[ImportCandidate],
    mapper: FileIDMapper,
) -> list[DependencyEvidence]:
    """Convert import candidates to dependency evidence entries."""
    evidence: list[DependencyEvidence] = []
    for c in candidates:
        try:
            src_id = mapper.id_for(c.source_file)
            tgt_id = mapper.id_for(c.target_file)
        except KeyError:
            # File not in mapper (external dependency) — skip
            continue
        evidence.append(DependencyEvidence(
            source_id=src_id,
            target_id=tgt_id,
            raw_import=c.raw_import,
            score=c.score,
            reason=c.reason,
            ambiguity=c.ambiguity,
        ))
    return evidence


def _pixel_evidence(
    mapper: FileIDMapper,
    mgr: ShardManager,
) -> list[DependencyEvidence]:
    """Query PixelMem for ``depends_on`` triples among mapped files.

    Looks up each file ID as an entity and collects any
    ``depends_on`` relations whose object is also a known file ID.
    """
    evidence: list[DependencyEvidence] = []
    all_ids = mapper.all_ids()
    id_set = set(all_ids)

    # Build lookup: basename (no ext) -> file IDs, for fuzzy entity matching
    basename_to_ids: dict[str, list[str]] = {}
    for fid in all_ids:
        bname = mapper.basename_for(fid).replace(".py", "")
        basename_to_ids.setdefault(bname, []).append(fid)

    # Query each file entity for depends_on triples
    for fid in all_ids:
        basename = mapper.basename_for(fid).replace(".py", "")
        entity_names = [fid, basename]

        try:
            triples = mgr.query(
                query=f"{fid} depends_on",
                entity_names=entity_names,
                max_shards=2,
                max_hops=1,
            )
        except Exception:
            continue

        for triple in triples:
            rel = triple.relation.lower().replace(" ", "_") if hasattr(triple, "relation") else ""
            if "depends" not in rel:
                continue

            obj_text = triple.object if hasattr(triple, "object") else ""
            # Try to resolve the object to a file ID
            target_ids: list[str] = []
            # Direct ID reference (e.g. "F3")
            if obj_text in id_set:
                target_ids = [obj_text]
            else:
                # Try basename matching
                obj_clean = obj_text.replace(".py", "").strip()
                if obj_clean in basename_to_ids:
                    target_ids = basename_to_ids[obj_clean]

            for tid in target_ids:
                if tid == fid:
                    continue
                evidence.append(DependencyEvidence(
                    source_id=fid,
                    target_id=tid,
                    raw_import="pixel:depends_on",
                    score=0.6,  # moderate confidence from pixel memory
                    reason="pixel_depends_on",
                    ambiguity=0,
                ))

    return evidence


def build_evidence_graph(
    files: list[str],
    file_contents: dict[str, str],
    mapper: FileIDMapper,
    mgr: ShardManager | None = None,
) -> list[DependencyEvidence]:
    """Build a unified dependency evidence list from all sources.

    Sources:
        1. Static import resolution (``import_resolver.resolve_all_imports``).
        2. PixelMem ``depends_on`` triples (if *mgr* is provided).

    Deduplication: when the same ``(source_id, target_id)`` pair
    appears from multiple sources, the entry with the **highest**
    score is kept.

    Args:
        files: All project file paths.
        file_contents: ``{path: source_text}`` for every file.
        mapper: Pre-built ``FileIDMapper``.
        mgr: Optional ``ShardManager`` for pixel-memory evidence.

    Returns:
        Deduplicated list of ``DependencyEvidence``, sorted by score
        descending.
    """
    # Source 1: static imports
    candidates = resolve_all_imports(files, file_contents, mapper)
    evidence = _candidates_to_evidence(candidates, mapper)

    # Source 2: pixel memory (optional)
    if mgr is not None:
        evidence.extend(_pixel_evidence(mapper, mgr))

    # Dedup: (source, target) -> keep highest score
    best: dict[tuple[str, str], DependencyEvidence] = {}
    for ev in evidence:
        key = (ev.source_id, ev.target_id)
        if key not in best or ev.score > best[key].score:
            best[key] = ev

    result = list(best.values())
    result.sort(key=lambda e: e.score, reverse=True)
    return result


# ── Edge classification ────────────────────────────────────────────


def classify_edges(
    evidence: list[DependencyEvidence],
) -> tuple[list[DependencyEvidence], list[DependencyEvidence], list[DependencyEvidence]]:
    """Partition evidence into confidence tiers.

    Returns:
        ``(strong, medium, weak)`` where:
        - **strong**: score > 0.7
        - **medium**: 0.3 <= score <= 0.7
        - **weak**: score < 0.3
    """
    strong: list[DependencyEvidence] = []
    medium: list[DependencyEvidence] = []
    weak: list[DependencyEvidence] = []

    for ev in evidence:
        if ev.score > 0.7:
            strong.append(ev)
        elif ev.score >= 0.3:
            medium.append(ev)
        else:
            weak.append(ev)

    return strong, medium, weak
