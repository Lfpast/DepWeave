"""File-level dependency graph derived from primitive quadruples.

Builds a directed graph of file dependencies by chaining
symbol-level primitives:

    imports_symbol + defined_in  =>  file A depends_on file B
    calls + defined_in           =>  file A depends_on file B
    extends + defined_in         =>  file A depends_on file B
    imports_module (internal)    =>  file A depends_on file B

The graph is cached after first build and invalidated only when
primitives change.

Supports:
    - Forward dependency lookup (what does A depend on?)
    - Reverse dependency lookup (what depends on A?)
    - Topological sort
    - Connected component extraction
    - Ambiguity detection (pairs with no edge)
    - Explanation generation (why does A depend on B?)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from pixelmem.triple_extractor import Triple
from pixelmem.v4.alias_namespace import AliasNamespace
from pixelmem.v4.primitive_extractor import (
    REL_IMPORTS_SYMBOL,
    REL_IMPORTS_MODULE,
    REL_CALLS,
    REL_EXTENDS,
    REL_DEFINED_IN,
    COND_INTERNAL,
    COND_INFERRED,
)


@dataclass
class DependencyEdge:
    """A single file-level dependency with evidence."""
    source: str          # file alias (depends ON target)
    target: str          # file alias (depended upon)
    evidence: list[str]  # human-readable evidence strings
    confidence: float    # 0.0-1.0
    edge_type: str       # "import", "call", "extends", "module", "heuristic"


@dataclass
class GraphStats:
    """Summary statistics for a dependency graph."""
    n_files: int
    n_edges: int
    n_strong_edges: int
    n_ambiguous_pairs: int
    has_cycle: bool
    n_components: int


class DependencyGraph:
    """Directed file-level dependency graph with caching.

    Build from primitive triples, then query for dependencies,
    topological ordering, and explanations.

    Optionally accepts a ``SymbolResolver`` to filter false edges
    where a file imports from ``__init__.py`` but the symbol is
    re-exported (not defined there).

    Usage::

        graph = DependencyGraph(namespace)
        graph.build_from_primitives(triples)

        # Queries
        graph.dependencies_of("main.py")    # what main.py depends on
        graph.dependents_of("base.py")      # what depends on base.py
        graph.topological_sort()             # full ordering
        graph.explain("main.py", "a(1).py") # why this edge exists
    """

    def __init__(self, ns: AliasNamespace, resolver=None) -> None:
        self._ns = ns
        self._resolver = resolver  # Reserved for future use
        # Forward: source -> {target -> DependencyEdge}
        self._forward: dict[str, dict[str, DependencyEdge]] = defaultdict(dict)
        # Reverse: target -> set of sources
        self._reverse: dict[str, set[str]] = defaultdict(set)
        # All file aliases in the graph
        self._files: set[str] = set()
        # Cache state
        self._built = False
        # Derived triples (for storage)
        self._derived_triples: list[Triple] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_primitives(self, triples: list[Triple]) -> None:
        """Build the file-level dependency graph from primitive triples.

        Chains:
            1. imports_symbol(A, sym_ref) + defined_in(sym, B) => A depends_on B
            2. imports_module(A, B, internal) => A depends_on B
            3. calls(sym_A, sym_B) + defined_in(sym_B, B) => file_of(sym_A) depends_on B
            4. extends(cls_A, base) + defined_in(base, B) => file_of(cls_A) depends_on B
        """
        self._forward.clear()
        self._reverse.clear()
        self._files.clear()
        self._derived_triples.clear()

        # Index primitives
        defined_in: dict[str, str] = {}     # sym_alias -> file_alias
        sym_file: dict[str, str] = {}       # sym_alias -> file_alias (same as defined_in)
        imports_sym: list[tuple[str, str]] = []   # (file_alias, sym_ref)
        imports_mod: list[tuple[str, str]] = []   # (file_alias, target_alias)
        calls_list: list[tuple[str, str]] = []    # (sym_alias, target_ref)
        extends_list: list[tuple[str, str]] = []  # (sym_alias, base_ref)

        all_file_aliases = set(self._ns.all_file_aliases())

        for t in triples:
            if t.relation == REL_DEFINED_IN:
                defined_in[t.subject] = t.object
                sym_file[t.subject] = t.object

            elif t.relation == REL_IMPORTS_SYMBOL:
                if t.condition in (COND_INTERNAL, COND_INFERRED):
                    imports_sym.append((t.subject, t.object))

            elif t.relation == REL_IMPORTS_MODULE:
                if t.condition == COND_INTERNAL and t.object in all_file_aliases:
                    imports_mod.append((t.subject, t.object))

            elif t.relation == REL_CALLS:
                calls_list.append((t.subject, t.object))

            elif t.relation == REL_EXTENDS:
                extends_list.append((t.subject, t.object))

        # Register all files
        self._files = set(all_file_aliases)

        # Chain 1: imports_symbol -> defined_in
        for src_file, sym_ref in imports_sym:
            target_file = defined_in.get(sym_ref)
            if not target_file:
                if "@" in sym_ref:
                    target_file = sym_ref.split("@", 1)[1]
                    if target_file not in all_file_aliases:
                        target_file = None
            if target_file and target_file != src_file:
                sym_name = sym_ref.split("@")[0] if "@" in sym_ref else sym_ref
                self._add_edge(
                    src_file, target_file,
                    f"imports symbol {sym_name}",
                    confidence=0.9, edge_type="import",
                )

        # Chain 2: imports_module (internal, target is a file alias)
        for src_file, target_file in imports_mod:
            if target_file != src_file:
                self._add_edge(
                    src_file, target_file,
                    f"imports module {target_file}",
                    confidence=0.85, edge_type="module",
                )

        # Chain 3: calls -> defined_in
        for caller_sym, target_ref in calls_list:
            caller_file = sym_file.get(caller_sym)
            target_file = defined_in.get(target_ref)
            if caller_file and target_file and caller_file != target_file:
                self._add_edge(
                    caller_file, target_file,
                    f"calls {target_ref.split('@')[0] if '@' in target_ref else target_ref}",
                    confidence=0.8, edge_type="call",
                )

        # Chain 4: extends -> defined_in (or symbol lookup)
        for cls_sym, base_ref in extends_list:
            cls_file = sym_file.get(cls_sym)
            if not cls_file:
                continue

            # Try direct defined_in lookup
            base_file = defined_in.get(base_ref)
            if not base_file:
                # Try symbol name matching across all files
                matches = self._ns.find_symbol(base_ref)
                if len(matches) == 1:
                    base_file = matches[0].file_alias

            if base_file and base_file != cls_file:
                self._add_edge(
                    cls_file, base_file,
                    f"{cls_sym.split('@')[0]} extends {base_ref}",
                    confidence=0.95, edge_type="extends",
                )

        # Add heuristic edges
        self._add_heuristic_edges()

        # Build derived triples for caching/storage
        self._build_derived_triples()

        self._built = True

    def _trace_reexport(
        self,
        target_file: str,
        sym_name: str,
        defined_in: dict[str, str],
        all_file_aliases: set[str],
    ) -> Optional[str]:
        """Trace through __init__.py re-exports to find the real source.

        If target_file is an __init__.py and has a SymbolResolver, check
        whether the symbol is defined there or re-exported from another file.
        If re-exported, return the actual defining file instead.

        Returns target_file unchanged if:
        - target is not __init__.py
        - no resolver available
        - symbol is actually defined in __init__.py
        - re-export source is not in our file set
        """
        entry = self._ns.file_entry(target_file)
        if entry.basename != "__init__.py":
            return target_file

        if not self._resolver:
            return target_file

        table = self._resolver.get_file_table(target_file)
        if not table:
            return target_file

        # Is the symbol defined in __init__.py itself?
        if sym_name in table.defined:
            return target_file

        # Is it re-exported from a file in our set?
        if sym_name in table.re_exported:
            prov = table.re_exported[sym_name]
            if prov.defined_in and prov.defined_in in all_file_aliases:
                # Redirect to the actual source file
                return prov.defined_in

        # Symbol not found in __init__.py's table — could be from an
        # external package imported through the package, or from a file
        # not in our subset. Keep the original edge (conservative).
        return target_file

    def _add_edge(
        self,
        source: str,
        target: str,
        evidence: str,
        confidence: float,
        edge_type: str,
    ) -> None:
        """Add or strengthen a dependency edge."""
        existing = self._forward[source].get(target)
        if existing:
            existing.evidence.append(evidence)
            existing.confidence = max(existing.confidence, confidence)
        else:
            self._forward[source][target] = DependencyEdge(
                source=source,
                target=target,
                evidence=[evidence],
                confidence=confidence,
                edge_type=edge_type,
            )
            self._reverse[target].add(source)

    def _add_heuristic_edges(self) -> None:
        """Add weak edges from naming conventions."""
        for file_alias in self._files:
            entry = self._ns.file_entry(file_alias)
            bn = entry.basename

            # test_X.py depends on X.py
            if bn.startswith("test_"):
                tested_bn = bn[5:]
                for candidate in self._ns.aliases_for_basename(tested_bn):
                    self._add_edge(
                        file_alias, candidate,
                        f"test file for {tested_bn}",
                        confidence=0.5, edge_type="heuristic",
                    )


    def _build_derived_triples(self) -> None:
        """Generate derived triples for PixelMem caching."""
        for source, targets in self._forward.items():
            for target, edge in targets.items():
                self._derived_triples.append(Triple(
                    source, "depends_on", target, "derived_dependency"
                ))
                if edge.edge_type == "call":
                    self._derived_triples.append(Triple(
                        source, "calls_into", target, "derived_dependency"
                    ))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def dependencies_of(self, file_alias: str) -> list[DependencyEdge]:
        """What does this file depend on? (forward edges)"""
        return list(self._forward.get(file_alias, {}).values())

    def dependents_of(self, file_alias: str) -> list[DependencyEdge]:
        """What files depend on this one? (reverse edges)"""
        return [
            self._forward[src][file_alias]
            for src in self._reverse.get(file_alias, set())
            if file_alias in self._forward.get(src, {})
        ]

    def has_edge(self, source: str, target: str) -> bool:
        return target in self._forward.get(source, {})

    def edge(self, source: str, target: str) -> Optional[DependencyEdge]:
        return self._forward.get(source, {}).get(target)

    def all_edges(self) -> list[DependencyEdge]:
        return [
            edge
            for targets in self._forward.values()
            for edge in targets.values()
        ]

    def strong_edges(self, threshold: float = 0.7) -> list[DependencyEdge]:
        return [e for e in self.all_edges() if e.confidence > threshold]

    # ------------------------------------------------------------------
    # Graph algorithms
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """Topological sort (base files first, dependents last).

        Uses Kahn's algorithm. Files with no dependencies come first.
        Tie-breaking: alphabetical, with test_ and __init__ pushed later.
        """
        in_degree: dict[str, int] = {f: 0 for f in self._files}
        adj: dict[str, list[str]] = {f: [] for f in self._files}

        for src, targets in self._forward.items():
            for tgt in targets:
                if tgt in self._files:
                    adj[tgt].append(src)  # tgt -> src (tgt must come before src)
                    in_degree[src] = in_degree.get(src, 0) + 1

        queue = sorted(
            [f for f in self._files if in_degree[f] == 0],
            key=_sort_key,
        )
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for dependent in sorted(adj.get(node, []), key=_sort_key):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    queue.sort(key=_sort_key)

        # Remaining (cycle): append
        remaining = [f for f in self._files if f not in set(result)]
        result.extend(sorted(remaining, key=_sort_key))

        return result

    def topological_layers(self) -> list[list[str]]:
        """Partition files into layers by dependency depth.

        Layer 0: no dependencies. Layer 1: depends only on layer 0. Etc.
        """
        in_degree: dict[str, int] = {f: 0 for f in self._files}
        adj: dict[str, set[str]] = {f: set() for f in self._files}

        for src, targets in self._forward.items():
            for tgt in targets:
                if tgt in self._files and self._forward[src][tgt].confidence > 0.7:
                    adj[tgt].add(src)
                    in_degree[src] += 1

        layers: list[list[str]] = []
        queue = sorted([f for f in self._files if in_degree[f] == 0])

        while queue:
            layers.append(list(queue))
            next_q: list[str] = []
            for node in queue:
                for dep in adj.get(node, set()):
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        next_q.append(dep)
            queue = sorted(next_q)

        # Remaining (cycle)
        visited = {f for layer in layers for f in layer}
        remaining = sorted(f for f in self._files if f not in visited)
        if remaining:
            layers.append(remaining)

        return layers

    def ambiguous_pairs(self) -> list[tuple[str, str]]:
        """File pairs with no strong edge in either direction."""
        files = sorted(self._files)
        pairs = []
        for i, a in enumerate(files):
            for b in files[i + 1:]:
                fwd = self._forward.get(a, {}).get(b)
                rev = self._forward.get(b, {}).get(a)
                max_conf = max(
                    fwd.confidence if fwd else 0.0,
                    rev.confidence if rev else 0.0,
                )
                if max_conf < 0.3:
                    pairs.append((a, b))
        return pairs

    def connected_components(self) -> list[set[str]]:
        """Undirected connected components."""
        visited: set[str] = set()
        components: list[set[str]] = []

        undirected: dict[str, set[str]] = defaultdict(set)
        for src, targets in self._forward.items():
            for tgt in targets:
                undirected[src].add(tgt)
                undirected[tgt].add(src)

        for f in self._files:
            if f in visited:
                continue
            component: set[str] = set()
            queue = deque([f])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor in undirected.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if component:
                components.append(component)

        # Add isolated files
        for f in self._files:
            if f not in visited:
                components.append({f})

        return components

    def has_cycle(self) -> bool:
        """Check for cycles in strong edges."""
        WHITE, GREY, BLACK = 0, 1, 2
        color: dict[str, int] = {f: WHITE for f in self._files}

        strong_adj: dict[str, set[str]] = defaultdict(set)
        for src, targets in self._forward.items():
            for tgt, edge in targets.items():
                if edge.confidence > 0.7:
                    strong_adj[src].add(tgt)

        def dfs(u: str) -> bool:
            color[u] = GREY
            for v in strong_adj.get(u, set()):
                if color.get(v) == GREY:
                    return True
                if color.get(v) == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False

        return any(dfs(f) for f in self._files if color[f] == WHITE)

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain(self, source: str, target: str) -> Optional[list[str]]:
        """Why does source depend on target? Returns evidence list or None."""
        edge = self._forward.get(source, {}).get(target)
        if edge:
            return list(edge.evidence)
        return None

    # ------------------------------------------------------------------
    # Stats / derived triples
    # ------------------------------------------------------------------

    def stats(self) -> GraphStats:
        all_e = self.all_edges()
        return GraphStats(
            n_files=len(self._files),
            n_edges=len(all_e),
            n_strong_edges=sum(1 for e in all_e if e.confidence > 0.7),
            n_ambiguous_pairs=len(self.ambiguous_pairs()),
            has_cycle=self.has_cycle(),
            n_components=len(self.connected_components()),
        )

    def derived_triples(self) -> list[Triple]:
        """Derived triples ready for PixelMem storage/caching."""
        return list(self._derived_triples)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"DependencyGraph({s.n_files} files, {s.n_edges} edges, "
            f"{s.n_strong_edges} strong)"
        )


def _sort_key(file_alias: str) -> tuple[int, str]:
    """Sort key pushing test_ and __init__ later."""
    bn = file_alias.split("(")[0] if "(" in file_alias else file_alias
    priority = 0
    if bn.startswith("test_"):
        priority = 2
    elif bn == "__init__.py":
        priority = 1
    return (priority, file_alias)
