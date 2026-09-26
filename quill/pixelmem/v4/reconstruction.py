"""Reconstruction layer: build compact evidence from primitive facts.

The LLM should NOT receive raw primitive triples. This layer
reconstructs higher-level evidence objects that are compact and
human-readable.

Three reconstruction types:
    A. File summary: functions, classes, imports
    B. Dependency explanation: why A depends on B
    C. Duplicate disambiguation: which a.py is which
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pixelmem.triple_extractor import Triple
from pixelmem.v4.alias_namespace import AliasNamespace
from pixelmem.v4.dependency_graph import DependencyGraph
from pixelmem.v4.natural_labels import NaturalLabeler
from pixelmem.v4.primitive_extractor import (
    REL_CONTAINS_FUNCTION,
    REL_CONTAINS_CLASS,
    REL_IMPORTS_SYMBOL,
    REL_IMPORTS_MODULE,
    REL_DEFINES_CONSTANT,
    COND_INTERNAL,
    COND_EXTERNAL,
    COND_INFERRED,
)


# ------------------------------------------------------------------
# Evidence objects
# ------------------------------------------------------------------


@dataclass
class FileSummaryEvidence:
    """Compact summary of a file's contents."""
    file_alias: str
    natural_label: str
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    internal_imports: list[str] = field(default_factory=list)
    external_imports: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        parts = [f"{self.natural_label}:"]
        if self.functions:
            parts.append(f"  defines: {', '.join(self.functions[:8])}")
        if self.classes:
            parts.append(f"  classes: {', '.join(self.classes[:8])}")
        if self.internal_imports:
            parts.append(f"  imports (local): {', '.join(self.internal_imports[:6])}")
        if self.external_imports:
            parts.append(f"  imports (external): {', '.join(self.external_imports[:4])}")
        return "\n".join(parts)


@dataclass
class DependencyEvidence:
    """Why one file depends on another."""
    source_label: str
    target_label: str
    support: list[str]
    confidence: float
    edge_type: str

    def to_text(self) -> str:
        support_str = "; ".join(self.support)
        return f"{self.source_label} depends on {self.target_label}: {support_str}"


@dataclass
class DisambiguationEvidence:
    """Which duplicate-named file is which."""
    basename: str
    candidates: list[dict]  # [{natural_label, functions, classes}]

    def to_text(self) -> str:
        lines = [f"Multiple files named {self.basename}:"]
        for c in self.candidates:
            desc_parts = []
            if c.get("functions"):
                desc_parts.append(f"defines {', '.join(c['functions'][:3])}")
            if c.get("classes"):
                desc_parts.append(f"has class {', '.join(c['classes'][:2])}")
            if c.get("parent_dir"):
                desc_parts.append(f"in {c['parent_dir']}/")
            desc = "; ".join(desc_parts) if desc_parts else "empty"
            lines.append(f"  - {c['natural_label']}: {desc}")
        return "\n".join(lines)


@dataclass
class OrderingEvidence:
    """Evidence for file ordering."""
    ordered_labels: list[str]
    strong_edges: list[str]  # "A before B" descriptions
    ambiguous_pairs: list[str]  # "A vs B: no evidence"

    def to_text(self) -> str:
        lines = ["Dependency order (base first):"]
        for i, label in enumerate(self.ordered_labels, 1):
            lines.append(f"  {i}. {label}")
        if self.strong_edges:
            lines.append("\nConfirmed dependencies:")
            for e in self.strong_edges:
                lines.append(f"  - {e}")
        if self.ambiguous_pairs:
            lines.append("\nAmbiguous (no direct dependency):")
            for p in self.ambiguous_pairs:
                lines.append(f"  - {p}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Reconstructor
# ------------------------------------------------------------------


class EvidenceReconstructor:
    """Build compact evidence objects from primitives + graph.

    Usage::

        recon = EvidenceReconstructor(ns, graph, labeler, triples)
        summary = recon.file_summary("main.py")
        dep = recon.dependency("main.py", "a(1).py")
        disambig = recon.disambiguation("a.py")
        ordering = recon.ordering(["main.py", "a(1).py", "base.py"])
    """

    def __init__(
        self,
        ns: AliasNamespace,
        graph: DependencyGraph,
        labeler: NaturalLabeler,
        triples: list[Triple],
    ) -> None:
        self._ns = ns
        self._graph = graph
        self._labeler = labeler
        # Index triples by subject for fast lookup
        self._by_subject: dict[str, list[Triple]] = {}
        for t in triples:
            self._by_subject.setdefault(t.subject, []).append(t)

    # ------------------------------------------------------------------
    # File summary
    # ------------------------------------------------------------------

    def file_summary(self, file_alias: str) -> FileSummaryEvidence:
        """Reconstruct a compact file summary from primitives."""
        label = self._labeler.label(file_alias)
        funcs, classes, constants = [], [], []
        internal_imports, external_imports = [], []

        for t in self._by_subject.get(file_alias, []):
            if t.relation == REL_CONTAINS_FUNCTION:
                name = t.object.split("@")[0] if "@" in t.object else t.object
                funcs.append(name)
            elif t.relation == REL_CONTAINS_CLASS:
                name = t.object.split("@")[0] if "@" in t.object else t.object
                classes.append(name)
            elif t.relation == REL_DEFINES_CONSTANT:
                name = t.object.split("@")[0] if "@" in t.object else t.object
                constants.append(name)
            elif t.relation == REL_IMPORTS_MODULE:
                if t.condition == COND_INTERNAL:
                    internal_imports.append(self._labeler.label(t.object))
                elif t.condition == COND_EXTERNAL:
                    external_imports.append(t.object)
            elif t.relation == REL_IMPORTS_SYMBOL:
                if t.condition in (COND_INTERNAL, COND_INFERRED):
                    sym_label = self._labeler.label_symbol(t.object)
                    internal_imports.append(sym_label)
                elif t.condition == COND_EXTERNAL:
                    external_imports.append(t.object)

        return FileSummaryEvidence(
            file_alias=file_alias,
            natural_label=label,
            functions=funcs,
            classes=classes,
            constants=constants,
            internal_imports=list(dict.fromkeys(internal_imports)),  # dedup, preserve order
            external_imports=list(dict.fromkeys(external_imports)),
        )

    # ------------------------------------------------------------------
    # Dependency explanation
    # ------------------------------------------------------------------

    def dependency(
        self,
        source_alias: str,
        target_alias: str,
    ) -> Optional[DependencyEvidence]:
        """Reconstruct an explanation of why source depends on target."""
        edge = self._graph.edge(source_alias, target_alias)
        if not edge:
            return None

        return DependencyEvidence(
            source_label=self._labeler.label(source_alias),
            target_label=self._labeler.label(target_alias),
            support=list(edge.evidence),
            confidence=edge.confidence,
            edge_type=edge.edge_type,
        )

    # ------------------------------------------------------------------
    # Disambiguation
    # ------------------------------------------------------------------

    def disambiguation(self, basename: str) -> Optional[DisambiguationEvidence]:
        """Reconstruct disambiguation info for duplicate basenames."""
        aliases = self._ns.aliases_for_basename(basename)
        if len(aliases) <= 1:
            return None

        candidates = []
        for alias in aliases:
            entry = self._ns.file_entry(alias)
            summary = self.file_summary(alias)
            candidates.append({
                "natural_label": self._labeler.label(alias),
                "functions": summary.functions,
                "classes": summary.classes,
                "parent_dir": entry.parent_dir,
            })

        return DisambiguationEvidence(
            basename=basename,
            candidates=candidates,
        )

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def ordering(self, file_aliases: list[str]) -> OrderingEvidence:
        """Reconstruct ordering evidence for a set of files."""
        order = self._graph.topological_sort()
        # Filter to requested files, preserving topo order
        ordered = [f for f in order if f in set(file_aliases)]

        # Strong edges
        strong = []
        for edge in self._graph.strong_edges():
            if edge.source in set(file_aliases) and edge.target in set(file_aliases):
                src_label = self._labeler.label(edge.source)
                tgt_label = self._labeler.label(edge.target)
                strong.append(f"{src_label} depends on {tgt_label}")

        # Ambiguous pairs
        amb = []
        for a, b in self._graph.ambiguous_pairs():
            if a in set(file_aliases) and b in set(file_aliases):
                amb.append(f"{self._labeler.label(a)} vs {self._labeler.label(b)}")

        return OrderingEvidence(
            ordered_labels=self._labeler.label_list(ordered),
            strong_edges=strong,
            ambiguous_pairs=amb,
        )
