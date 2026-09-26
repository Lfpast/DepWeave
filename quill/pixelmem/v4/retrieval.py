"""Four-stage retrieval pipeline over primitive quadruples.

Stage 1: Candidate grounding
    Map query to candidate files/symbols, expand duplicates.

Stage 2: Subgraph extraction
    Extract relevant primitives and/or cached derived edges.

Stage 3: Reconstruction
    Build compact evidence objects from primitives.

Stage 4: Final prompt packaging
    Produce natural, human-readable, alias-free explanation.

Query modes:
    - file_contents: what does X contain?
    - symbol_lookup: where is X defined?
    - file_dependency: what depends on X? / what does X import?
    - disambiguation: which X contains Y?
    - ordering: what is the dependency order?
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from pixelmem.shard_manager import ShardManager
from pixelmem.triple_extractor import Triple

from pixelmem.v4.alias_namespace import AliasNamespace
from pixelmem.v4.primitive_extractor import extract_primitives
from pixelmem.v4.dependency_graph import DependencyGraph
from pixelmem.v4.natural_labels import NaturalLabeler
from pixelmem.v4.symbol_resolver import SymbolResolver
from pixelmem.v4.reconstruction import (
    EvidenceReconstructor,
    FileSummaryEvidence,
    DependencyEvidence,
    DisambiguationEvidence,
    OrderingEvidence,
)


class QueryMode(Enum):
    FILE_CONTENTS = "file_contents"
    SYMBOL_LOOKUP = "symbol_lookup"
    FILE_DEPENDENCY = "file_dependency"
    DISAMBIGUATION = "disambiguation"
    ORDERING = "ordering"


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""
    mode: QueryMode
    prompt_text: str         # final LLM-ready text (no internal aliases)
    evidence_objects: list   # raw evidence objects for programmatic access
    n_primitives_used: int
    n_derived_used: int


class RetrievalPipeline:
    """End-to-end retrieval over primitive quadruples stored in PixelMem.

    Orchestrates: namespace → extraction → PixelMem storage → graph →
    reconstruction → natural labels → final prompt.

    Usage::

        pipeline = RetrievalPipeline(debug=False)
        pipeline.index(files, file_contents)

        # Query
        result = pipeline.query("ordering", files)
        print(result.prompt_text)  # LLM-ready, no internal aliases
    """

    def __init__(
        self,
        shard_size: int = 128,
        debug: bool = False,
    ) -> None:
        self._shard_size = shard_size
        self._debug = debug

        # Components (built during index())
        self._ns: Optional[AliasNamespace] = None
        self._graph: Optional[DependencyGraph] = None
        self._labeler: Optional[NaturalLabeler] = None
        self._reconstructor: Optional[EvidenceReconstructor] = None
        self._primitives: list[Triple] = []
        self._mgr: Optional[ShardManager] = None
        self._td: Optional[str] = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        files: list[str],
        file_contents: dict[str, str],
        language: str = "python",
    ) -> None:
        """Build the full index: namespace, primitives, graph, caches.

        Args:
            files: List of original file paths.
            file_contents: ``{path: source_code}`` dict.
            language: Programming language.
        """
        # Step 1: Build alias namespace
        self._ns = AliasNamespace()
        self._ns.register_files(files)

        # Step 2: Extract primitive quadruples
        self._primitives = extract_primitives(files, file_contents, self._ns, language)

        # Step 3: Compiler-style symbol resolution
        # Trace re-exports through __init__.py, filter stdlib, fix false edges
        self._resolver = SymbolResolver(self._ns)
        self._resolver.build_tables(files, file_contents)
        self._primitives = self._fix_edges_with_resolver(
            self._primitives, files, file_contents,
        )

        # Step 4: Store in PixelMem (encode)
        self._td = tempfile.mkdtemp(prefix="v4_")
        self._mgr = ShardManager(self._td, shard_size=self._shard_size)
        for i in range(0, len(self._primitives), 10):
            self._mgr.encode("", triples=self._primitives[i:i + 10])

        # Step 5: Build dependency graph from corrected primitives
        # Pass the resolver so the graph can trace re-exports through __init__.py
        self._graph = DependencyGraph(self._ns, resolver=self._resolver)
        self._graph.build_from_primitives(self._primitives)

        # Step 6: Store derived triples in PixelMem (cache)
        derived = self._graph.derived_triples()
        if derived:
            for i in range(0, len(derived), 10):
                self._mgr.encode("", triples=derived[i:i + 10])

        # Step 7: Save and rebuild index
        self._mgr.save()
        self._mgr._rebuild_entity_index()

        # Step 8: Build labeler and reconstructor
        self._labeler = NaturalLabeler(self._ns, self._graph, debug=self._debug)
        self._reconstructor = EvidenceReconstructor(
            self._ns, self._graph, self._labeler, self._primitives,
        )

    # ------------------------------------------------------------------
    # Compiler-style edge correction
    # ------------------------------------------------------------------

    def _fix_edges_with_resolver(
        self,
        triples: list[Triple],
        files: list[str],
        file_contents: dict[str, str],
    ) -> list[Triple]:
        """Use the symbol resolver to filter stdlib false edges.

        Only stdlib filtering — remove edges where the import clearly
        references a stdlib module (e.g. ``from types import SimpleNamespace``).
        Re-export tracing is too aggressive and removes valid edges.
        """
        from pixelmem.v4.primitive_extractor import (
            REL_IMPORTS_SYMBOL, REL_IMPORTS_MODULE,
            COND_INTERNAL, COND_INFERRED,
        )

        corrected: list[Triple] = []
        all_aliases = set(self._ns.all_file_aliases())

        for t in triples:
            # Only check cross-file import edges
            if t.relation not in (REL_IMPORTS_SYMBOL, REL_IMPORTS_MODULE):
                corrected.append(t)
                continue

            if t.condition not in (COND_INTERNAL, COND_INFERRED):
                corrected.append(t)
                continue

            src_alias = t.subject
            if src_alias not in all_aliases:
                corrected.append(t)
                continue

            # Find the matching import line to check for stdlib
            src_path = self._ns.original_path_for(src_alias)
            code = file_contents.get(src_path, "")

            sym_name = t.object.split("@")[0] if "@" in t.object else t.object
            import_line = self._find_import_line(code, sym_name)

            if import_line and self._resolver.is_stdlib_import(import_line):
                # Drop stdlib edge
                continue

            corrected.append(t)

        return corrected

    def _find_import_line(self, code: str, sym_name: str) -> Optional[str]:
        """Find the import line in source code that imports sym_name."""
        for line in code.split("\n"):
            stripped = line.strip()
            if not stripped.startswith(("from ", "import ")):
                continue
            if sym_name in stripped:
                return stripped
        return None

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        mode: str,
        targets: Optional[list[str]] = None,
        symbol_name: Optional[str] = None,
    ) -> RetrievalResult:
        """Run a retrieval query.

        Args:
            mode: One of "file_contents", "symbol_lookup",
                "file_dependency", "disambiguation", "ordering".
            targets: File paths or aliases to query about.
            symbol_name: Symbol name for symbol_lookup mode.
        """
        qmode = QueryMode(mode)

        # Stage 1: Candidate grounding
        file_aliases = self._ground_candidates(targets)

        # Stage 2-4: Mode-specific retrieval
        if qmode == QueryMode.FILE_CONTENTS:
            return self._query_file_contents(file_aliases)
        elif qmode == QueryMode.SYMBOL_LOOKUP:
            return self._query_symbol_lookup(symbol_name or "")
        elif qmode == QueryMode.FILE_DEPENDENCY:
            return self._query_file_dependency(file_aliases)
        elif qmode == QueryMode.DISAMBIGUATION:
            return self._query_disambiguation(file_aliases)
        elif qmode == QueryMode.ORDERING:
            return self._query_ordering(file_aliases)
        else:
            raise ValueError(f"Unknown query mode: {mode}")

    # ------------------------------------------------------------------
    # Ordering (primary use case for DependEval)
    # ------------------------------------------------------------------

    def run_ordering(
        self,
        files: list[str],
        file_contents: dict[str, str],
        ask_fn: Callable,
        language: str = "python",
    ) -> tuple[str, dict]:
        """Full ordering pipeline: index → graph → LLM verify → answer.

        Drop-in replacement for ``file_summary_store.index_and_query``
        and ``depeval.pipeline.run_dependeval``.

        Returns:
            (json_answer, stats_dict)
        """
        # Index everything
        self.index(files, file_contents, language)

        # Get graph-based ordering
        result = self.query("ordering", files)

        # Build structural evidence prompt
        # The graph already knows dependencies — tell the LLM what we found
        # instead of making it re-derive from raw imports.
        topo_order = self._graph.topological_sort()
        ordered_labels = self._labeler.label_list(topo_order)

        prompt = self._build_structural_prompt(
            topo_order, ordered_labels, file_contents
        )

        # Ask LLM
        raw_answer = ask_fn(prompt)
        if isinstance(raw_answer, tuple):
            answer_text, in_tok, out_tok = raw_answer
            tokens_used = in_tok + out_tok
        else:
            answer_text = raw_answer
            tokens_used = 0

        # Parse response
        try:
            m = re.search(r"\[.*\]", answer_text, re.DOTALL)
            pred_labels = json.loads(m.group(0)) if m else ordered_labels
        except (json.JSONDecodeError, AttributeError):
            pred_labels = ordered_labels

        # Map labels back to aliases, then to basenames
        label_to_alias = {self._labeler.label(a): a for a in topo_order}
        pred_aliases = []
        for label in pred_labels:
            alias = label_to_alias.get(label)
            if alias:
                pred_aliases.append(alias)
            else:
                # Fuzzy match
                for a in topo_order:
                    if self._ns.file_entry(a).basename == label:
                        if a not in pred_aliases:
                            pred_aliases.append(a)
                            break

        # Convert to basenames
        pred_basenames = [self._ns.file_entry(a).basename for a in pred_aliases]

        # Fill in any missing
        all_basenames = [self._ns.file_entry(a).basename for a in topo_order]
        for bn in all_basenames:
            if bn not in pred_basenames:
                pred_basenames.append(bn)

        json_answer = json.dumps(pred_basenames)

        stats = {
            "n_primitives": len(self._primitives),
            "n_derived": len(self._graph.derived_triples()),
            "n_strong_edges": len(self._graph.strong_edges()),
            "n_ambiguous_pairs": len(self._graph.ambiguous_pairs()),
            "has_cycle": self._graph.has_cycle(),
            "tokens_used": tokens_used,
            "graph": repr(self._graph),
        }

        return json_answer, stats

    # ------------------------------------------------------------------
    # Mode-specific implementations
    # ------------------------------------------------------------------

    def _ground_candidates(self, targets: Optional[list[str]]) -> list[str]:
        """Stage 1: Map query targets to file aliases."""
        if not targets:
            return self._ns.all_file_aliases()

        aliases = []
        for t in targets:
            # Try as full path
            try:
                aliases.append(self._ns.file_alias_for(t))
                continue
            except KeyError:
                pass
            # Try as alias directly
            try:
                self._ns.file_entry(t)
                aliases.append(t)
                continue
            except KeyError:
                pass
            # Try as basename (may expand to multiple)
            found = self._ns.aliases_for_basename(t)
            if found:
                aliases.extend(found)
            else:
                # Fuzzy: check if basename matches
                for alias in self._ns.all_file_aliases():
                    if self._ns.file_entry(alias).basename == t:
                        aliases.append(alias)
        return aliases

    def _query_file_contents(self, aliases: list[str]) -> RetrievalResult:
        """What does file X contain?"""
        evidence = []
        for alias in aliases:
            summary = self._reconstructor.file_summary(alias)
            evidence.append(summary)

        text = "\n\n".join(e.to_text() for e in evidence)
        return RetrievalResult(
            mode=QueryMode.FILE_CONTENTS,
            prompt_text=text,
            evidence_objects=evidence,
            n_primitives_used=sum(
                len(self._reconstructor._by_subject.get(a, []))
                for a in aliases
            ),
            n_derived_used=0,
        )

    def _query_symbol_lookup(self, name: str) -> RetrievalResult:
        """Where is symbol X defined?"""
        entries = self._ns.find_symbol(name)
        evidence = []
        for e in entries:
            file_label = self._labeler.label(e.file_alias)
            evidence.append({
                "symbol": e.name,
                "kind": e.kind,
                "file": file_label,
            })

        if evidence:
            lines = [f"Symbol '{name}' is defined in:"]
            for e in evidence:
                lines.append(f"  - {e['file']} ({e['kind']})")
            text = "\n".join(lines)
        else:
            text = f"Symbol '{name}' not found in any indexed file."

        return RetrievalResult(
            mode=QueryMode.SYMBOL_LOOKUP,
            prompt_text=text,
            evidence_objects=evidence,
            n_primitives_used=len(entries),
            n_derived_used=0,
        )

    def _query_file_dependency(self, aliases: list[str]) -> RetrievalResult:
        """What does X depend on? / What depends on X?"""
        evidence = []
        for alias in aliases:
            deps = self._graph.dependencies_of(alias)
            for d in deps:
                dep_ev = self._reconstructor.dependency(alias, d.target)
                if dep_ev:
                    evidence.append(dep_ev)

        text = "\n".join(e.to_text() for e in evidence) if evidence else "No dependencies found."
        return RetrievalResult(
            mode=QueryMode.FILE_DEPENDENCY,
            prompt_text=text,
            evidence_objects=evidence,
            n_primitives_used=0,
            n_derived_used=len(evidence),
        )

    def _query_disambiguation(self, aliases: list[str]) -> RetrievalResult:
        """Which X contains Y? Disambiguate duplicate basenames."""
        seen_basenames: set[str] = set()
        evidence = []
        for alias in aliases:
            bn = self._ns.file_entry(alias).basename
            if bn in seen_basenames:
                continue
            seen_basenames.add(bn)
            disambig = self._reconstructor.disambiguation(bn)
            if disambig:
                evidence.append(disambig)

        text = "\n\n".join(e.to_text() for e in evidence) if evidence else "No duplicate basenames."
        return RetrievalResult(
            mode=QueryMode.DISAMBIGUATION,
            prompt_text=text,
            evidence_objects=evidence,
            n_primitives_used=0,
            n_derived_used=0,
        )

    def _query_ordering(self, aliases: list[str]) -> RetrievalResult:
        """What is the dependency order?"""
        ordering = self._reconstructor.ordering(aliases)
        text = ordering.to_text()
        n_strong = len(self._graph.strong_edges())
        return RetrievalResult(
            mode=QueryMode.ORDERING,
            prompt_text=text,
            evidence_objects=[ordering],
            n_primitives_used=0,
            n_derived_used=n_strong,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Structural prompt builder
    # ------------------------------------------------------------------

    def _build_structural_prompt(
        self,
        topo_order: list[str],
        ordered_labels: list[str],
        file_contents: dict[str, str],
    ) -> str:
        """Build a hybrid prompt: raw imports + structural graph evidence.

        Combines:
          1. Per-file summaries with raw import lines (so LLM can catch
             edges the graph missed)
          2. Confirmed dependency edges from the graph (so LLM doesn't
             need to re-derive what we already know)
          3. The computed order as a suggestion
        """
        file_set = set(topo_order)

        # Section 1: Per-file summaries with raw imports + defines
        summary_lines = []
        for alias in topo_order:
            label = self._labeler.label(alias)
            path = self._ns.original_path_for(alias)
            code = file_contents.get(path, "")
            parts = [f"{label}:"]

            # Raw import lines (compact)
            imports = []
            for line in code.split("\n"):
                stripped = line.strip()
                if re.match(r"^\s*(?:from|import)\s", stripped):
                    imports.append(stripped)
            if imports:
                parts.append(f"  imports: {'; '.join(imports[:8])}")

            # Definitions
            defs = []
            for line in code.split("\n"):
                m = re.match(r"^\s*(?:def|class)\s+(\w+)", line.strip())
                if m:
                    defs.append(m.group(1))
            if defs:
                parts.append(f"  defines: {', '.join(defs[:8])}")

            summary_lines.append("\n".join(parts))

        # Section 2: Confirmed dependencies from graph
        dep_lines = []
        for edge in self._graph.strong_edges():
            if edge.source in file_set and edge.target in file_set:
                src_label = self._labeler.label(edge.source)
                tgt_label = self._labeler.label(edge.target)
                reason = edge.evidence[0] if edge.evidence else edge.edge_type
                dep_lines.append(f"  {src_label} depends on {tgt_label} ({reason})")

        # Build prompt
        prompt_parts = [
            "File summaries:\n" + "\n".join(summary_lines),
        ]

        if dep_lines:
            prompt_parts.append(
                "\nConfirmed dependencies:\n" + "\n".join(dep_lines)
            )

        order_json = json.dumps(ordered_labels)
        prompt_parts.append(
            f"\nComputed dependency order (base first, dependent last): {order_json}\n\n"
            "Is this correct? If yes, return it. If not, fix and return the corrected order.\n"
            "Return ONLY a JSON array of filenames, preserving exact case."
        )

        return "\n".join(prompt_parts)

    def cleanup(self) -> None:
        """Remove temporary storage."""
        if self._td:
            shutil.rmtree(self._td, ignore_errors=True)
            self._td = None

    def __del__(self) -> None:
        self.cleanup()
