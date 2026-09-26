"""Default python-deps plugin set — thin V4 adapter.

Exposes V4's extraction + resolver + graph as V5 plugins. This module is the
*only* place V5 imports V4; any other V5 file doing `from pixelmem.v4` is a
layering violation.

Adapter contract:

- ``V4Extractor`` runs V4's AST extractor and stdlib-aware resolver, converts
  ``v4.Triple`` to V5 ``Primitive``.
- ``V4DependencyEngine`` wraps V4's ``DependencyGraph`` so existing topo-sort
  and re-export tracing behavior is preserved byte-for-byte.
- ``V4OrderingPrompt`` builds V4's hybrid prompt and parses the JSON response.

Building a plugin set with these gets you V4's current behavior on
DependEval, accessible through V5's generic pipeline / harness.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# V4 imports — confined to this module only.
from pixelmem.triple_extractor import Triple as V4Triple
from pixelmem.v4.alias_namespace import AliasNamespace as V4AliasNamespace
from pixelmem.v4.dependency_graph import DependencyGraph as V4DependencyGraph
from pixelmem.v4.natural_labels import NaturalLabeler as V4NaturalLabeler
from pixelmem.v4.primitive_extractor import (
    extract_primitives as v4_extract_primitives,
    REL_IMPORTS_SYMBOL,
    REL_IMPORTS_MODULE,
    COND_INTERNAL,
    COND_INFERRED,
)
from pixelmem.v4.symbol_resolver import SymbolResolver as V4SymbolResolver

from quill.core.plugins import (
    DerivationEngine,
    DerivationRule,
    Extractor,
    PluginSet,
    PromptTemplate,
)
from quill.core.types import EvidenceBundle, Primitive, TaskSpec


# ---------------------------------------------------------------------------
# Translation helpers between V4 Triple and V5 Primitive
# ---------------------------------------------------------------------------


def _triple_to_primitive(t: V4Triple, provenance: Optional[dict] = None) -> Primitive:
    return Primitive(
        subject=t.subject,
        relation=t.relation,
        object=t.object,
        condition=t.condition or "",
        provenance=provenance,
    )


def _primitive_to_triple(p: Primitive) -> V4Triple:
    return V4Triple(p.subject, p.relation, p.object, p.condition)


# ---------------------------------------------------------------------------
# Extractor — V4 extractor + stdlib-aware symbol resolver
# ---------------------------------------------------------------------------


class V4Extractor(Extractor):
    """V5 Extractor that delegates to V4's AST extractor + SymbolResolver.

    The result is a namespace-anchored list of Primitives. The namespace
    itself is cached on the extractor so the derivation engine can reuse it.
    """

    def __init__(self, language: str = "python") -> None:
        self._language = language
        self._last_ns: Optional[V4AliasNamespace] = None
        self._last_resolver: Optional[V4SymbolResolver] = None
        self._last_files: Optional[list[str]] = None

    def extract(self, documents: dict[str, str], **kwargs: Any) -> list[Primitive]:
        files = list(documents.keys())
        if not files:
            return []

        ns = V4AliasNamespace()
        ns.register_files(files)

        triples = v4_extract_primitives(files, documents, ns, language=self._language)

        resolver = V4SymbolResolver(ns)
        resolver.build_tables(files, documents)
        triples = _filter_stdlib_edges(triples, files, documents, ns, resolver)

        self._last_ns = ns
        self._last_resolver = resolver
        self._last_files = files

        return [
            _triple_to_primitive(t, provenance={"source": "v4_extractor"})
            for t in triples
        ]

    # Introspection used by V4DependencyEngine.
    @property
    def last_namespace(self) -> Optional[V4AliasNamespace]:
        return self._last_ns

    @property
    def last_resolver(self) -> Optional[V4SymbolResolver]:
        return self._last_resolver


def _filter_stdlib_edges(
    triples: list[V4Triple],
    files: list[str],
    file_contents: dict[str, str],
    ns: V4AliasNamespace,
    resolver: V4SymbolResolver,
) -> list[V4Triple]:
    """Port of V4's ``RetrievalPipeline._fix_edges_with_resolver`` (stdlib-only).

    Finds the import line in source code that matches each edge, then drops
    the edge if it's clearly a stdlib import (e.g. ``from types import
    SimpleNamespace`` colliding with a local ``types.py``).
    """
    corrected: list[V4Triple] = []
    all_aliases = set(ns.all_file_aliases())

    for t in triples:
        if t.relation not in (REL_IMPORTS_SYMBOL, REL_IMPORTS_MODULE):
            corrected.append(t)
            continue
        if t.condition not in (COND_INTERNAL, COND_INFERRED):
            corrected.append(t)
            continue
        if t.subject not in all_aliases:
            corrected.append(t)
            continue

        src_path = ns.original_path_for(t.subject)
        code = file_contents.get(src_path, "")

        sym_name = t.object.split("@")[0] if "@" in t.object else t.object
        import_line = _find_import_line(code, sym_name)

        if import_line and resolver.is_stdlib_import(import_line):
            continue  # drop stdlib edge

        corrected.append(t)

    return corrected


def _find_import_line(code: str, sym_name: str) -> Optional[str]:
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("from ", "import ")):
            continue
        if sym_name in stripped:
            return stripped
    return None


# ---------------------------------------------------------------------------
# Derivation engine — wraps V4 DependencyGraph for topo sort parity
# ---------------------------------------------------------------------------


class V4DependencyEngine(DerivationEngine):
    """Derivation engine backed by ``v4.DependencyGraph``.

    Ignores the ``rules`` argument (V4's graph has its own chain logic) and
    instead pulls strong/ambiguous edges + topo sort straight from V4. This
    gives us byte-for-byte V4 parity on DependEval through a V5 pipeline.
    """

    def __init__(self, extractor: V4Extractor) -> None:
        self._extractor = extractor
        self._labeler: Optional[V4NaturalLabeler] = None
        self._graph: Optional[V4DependencyGraph] = None

    def derive(
        self,
        primitives: list[Primitive],
        rules: list[DerivationRule],
        task: TaskSpec,
    ) -> EvidenceBundle:
        ns = self._extractor.last_namespace
        resolver = self._extractor.last_resolver
        if ns is None:
            raise RuntimeError(
                "V4DependencyEngine requires the extractor to have run first"
            )

        v4_triples = [_primitive_to_triple(p) for p in primitives]

        graph = V4DependencyGraph(ns, resolver=resolver)
        graph.build_from_primitives(v4_triples)

        topo = graph.topological_sort()
        labeler = V4NaturalLabeler(ns, graph)

        strong = [
            Primitive(
                subject=edge.source,
                relation="depends_on",
                object=edge.target,
                condition="resolved",
                provenance={
                    "edge_type": edge.edge_type,
                    "confidence": edge.confidence,
                    "evidence": edge.evidence[:3],
                },
            )
            for edge in graph.strong_edges()
        ]
        ambiguous_pairs = graph.ambiguous_pairs()
        ambiguous = [
            Primitive(
                subject=a,
                relation="maybe_depends_on",
                object=b,
                condition="ambiguous",
                provenance={"pair_from": "ambiguous_pairs"},
            )
            for (a, b) in ambiguous_pairs
        ]

        ordering_hint = [labeler.label(a) for a in topo]

        self._graph = graph
        self._labeler = labeler

        return EvidenceBundle(
            strong=strong,
            ambiguous=ambiguous,
            raw_primitives=[],
            ordering_hint=ordering_hint,
            metadata={
                "n_strong": len(strong),
                "n_ambiguous": len(ambiguous),
                "has_cycle": graph.has_cycle(),
                "topo_aliases": topo,
                "basename_map": {
                    a: ns.file_entry(a).basename for a in topo
                },
                "label_to_alias": {labeler.label(a): a for a in topo},
            },
        )

    @property
    def last_labeler(self) -> Optional[V4NaturalLabeler]:
        return self._labeler


# ---------------------------------------------------------------------------
# Prompt template — V4 hybrid prompt + JSON parser
# ---------------------------------------------------------------------------


class V4OrderingPrompt(PromptTemplate):
    """Builds V4's hybrid prompt (raw imports + confirmed edges + topo order)
    and parses the JSON-array response into basenames.
    """

    _DEFAULT_BUDGET = 500

    def build(
        self,
        task: TaskSpec,
        query_input: dict,
        evidence: EvidenceBundle,
    ) -> str:
        files = query_input.get("files") or []
        file_contents = query_input.get("file_contents") or {}

        ordered_labels: list[str] = evidence.ordering_hint
        basename_map: dict[str, str] = evidence.metadata.get("basename_map", {})
        topo_aliases: list[str] = evidence.metadata.get("topo_aliases", [])

        # File summaries: raw import lines + top-level definitions.
        summaries = []
        for alias in topo_aliases:
            path = _alias_to_path(alias, files, basename_map)
            code = file_contents.get(path, "")
            imports = _extract_import_lines(code)
            defines = _extract_top_level_defs(code)
            label = ordered_labels[topo_aliases.index(alias)] if alias in topo_aliases else alias
            summaries.append(
                f"{label}:\n"
                f"  imports: {'; '.join(imports) or '(none)'}\n"
                f"  defines: {', '.join(defines) or '(none)'}"
            )

        # Confirmed dependency edges — only strong ones.
        labels = {a: ordered_labels[i] for i, a in enumerate(topo_aliases)}
        edge_lines = []
        for p in evidence.strong:
            s = labels.get(p.subject, p.subject)
            t = labels.get(p.object, p.object)
            edge_lines.append(f"  {s} depends on {t}")
        confirmed_text = "\n".join(edge_lines) if edge_lines else "  (none detected)"

        computed_order_json = json.dumps(ordered_labels)

        prompt = (
            "You are given a small set of Python source files. Return a "
            "dependency ordering where base files (imported by others) "
            "come first.\n\n"
            f"File summaries:\n{chr(10).join(summaries)}\n\n"
            f"Confirmed dependencies:\n{confirmed_text}\n\n"
            f"Computed dependency order: {computed_order_json}\n\n"
            "Is this correct? If yes, return it. If not, fix and return the "
            "corrected order. Return ONLY a JSON array of filenames."
        )
        return prompt

    def parse(self, completion: str, task: TaskSpec) -> list[str]:
        m = re.search(r"\[.*\]", completion, re.DOTALL)
        if not m:
            raise ValueError("no JSON array found in completion")
        items = json.loads(m.group(0))
        if not isinstance(items, list):
            raise ValueError("parsed JSON is not a list")
        # Normalize to basenames — the caller may hand us paths or labels.
        return [_normalize_basename(str(x)) for x in items]


def _alias_to_path(alias: str, files: list[str], basename_map: dict[str, str]) -> str:
    bn = basename_map.get(alias, alias)
    for p in files:
        if p.split("/")[-1] == bn:
            return p
    return alias


def _extract_import_lines(code: str, max_lines: int = 6) -> list[str]:
    lines = []
    for ln in code.splitlines():
        stripped = ln.strip()
        if stripped.startswith(("import ", "from ")):
            lines.append(stripped)
            if len(lines) >= max_lines:
                break
    return lines


def _extract_top_level_defs(code: str, max_defs: int = 8) -> list[str]:
    names: list[str] = []
    for ln in code.splitlines():
        m = re.match(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", ln)
        if m:
            names.append(m.group(1))
            if len(names) >= max_defs:
                break
    return names


def _normalize_basename(x: str) -> str:
    # Accept "the foo.py that defines bar" natural labels too.
    m = re.search(r"([A-Za-z0-9_\-]+\.py)", x)
    if m:
        return m.group(1)
    return x.split("/")[-1]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_python_deps_plugins() -> PluginSet:
    """Return a PluginSet wired up to V4 defaults — byte-for-byte V4 parity."""
    extractor = V4Extractor(language="python")
    engine = V4DependencyEngine(extractor)
    return PluginSet(
        name="python_deps_v4_adapter",
        extractor=extractor,
        resolver=None,  # stdlib filter already folded into V4Extractor
        derivation_rules=[],  # V4DependencyEngine ignores rules
        derivation_engine=engine,
        prompt_template=V4OrderingPrompt(),
    )


__all__ = [
    "V4Extractor",
    "V4DependencyEngine",
    "V4OrderingPrompt",
    "build_python_deps_plugins",
]
