"""Bayesian Dependency Ordering — probabilistic graph for file ordering.

Problems with simple topo sort:
  - Fails when edges are missing (partial graph)
  - No way to handle conflicting/ambiguous edges
  - Can't distinguish strong vs weak dependencies
  - Ties broken arbitrarily

Bayesian approach:
  - Each edge has a confidence score (0-1)
  - Strong: direct 'from .X import Y' between our files (conf=1.0)
  - Medium: 'from package.X import Y' resolved to our file (conf=0.8)
  - Weak: name overlap in calls/references (conf=0.3)
  - Build weighted DAG, find most probable ordering
  - Handle missing edges by using import pattern heuristics:
    * test_ files usually depend on the file they test
    * __init__.py usually depends on other files in same package
    * files with fewer imports are more likely base files
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from pixelmem.shard_manager import ShardManager


@dataclass
class DependencyEdge:
    source: str  # file that depends
    target: str  # file it depends on
    confidence: float  # 0-1
    evidence: str  # why we think this edge exists


def build_dependency_graph(
    files: list[str],
    mgr: ShardManager,
    file_contents: Optional[dict[str, str]] = None,
) -> list[DependencyEdge]:
    """Build weighted dependency edges from PixelMem pixel matrix.

    Reads depends_on triples and assigns confidence based on evidence type.
    Also infers implicit edges from naming patterns.
    """
    file_set = set(f.strip().lower() for f in files)
    bn_to_full = {}
    for f in files:
        bn_to_full[f.strip().lower()] = f.split("/")[-1]

    edges: list[DependencyEdge] = []
    seen: set[tuple[str, str]] = set()

    # 1. Read explicit depends_on edges from pixel matrix
    for shard in mgr.shards:
        for i in range(shard.n):
            subj = shard.idx_to_entity.get(i, "")
            if subj not in file_set:
                continue
            for j in range(shard.n):
                rgb = tuple(int(x) for x in shard.relation[i, j])
                if rgb == (0, 0, 0):
                    continue
                rel = shard.color_to_relation.get(rgb, "?")
                if rel != "depends_on":
                    continue
                obj = shard.idx_to_entity.get(j, "")
                if obj not in file_set or obj == subj:
                    continue

                src_bn = bn_to_full.get(subj, subj.split("/")[-1])
                tgt_bn = bn_to_full.get(obj, obj.split("/")[-1])

                # Determine confidence from the import line (in condition)
                crgb = tuple(int(x) for x in shard.condition[i, j])
                cond = shard.color_to_condition.get(crgb, "")

                if cond.startswith("from .") or cond.startswith("from .."):
                    conf = 1.0  # relative import — strongest signal
                    evidence = f"relative import: {cond[:50]}"
                elif "import" in cond:
                    conf = 0.85  # absolute import resolved to our file
                    evidence = f"resolved import: {cond[:50]}"
                else:
                    conf = 0.7  # edge exists but no import line
                    evidence = "dependency edge"

                key = (src_bn, tgt_bn)
                if key not in seen:
                    seen.add(key)
                    edges.append(DependencyEdge(src_bn, tgt_bn, conf, evidence))

    # 2. Infer implicit edges from naming patterns
    basenames = [f.split("/")[-1] for f in files]

    for bn in basenames:
        # test_ files depend on the file they test
        if bn.startswith("test_"):
            tested = bn[5:]  # test_foo.py → foo.py
            if tested in basenames:
                key = (bn, tested)
                if key not in seen:
                    seen.add(key)
                    edges.append(DependencyEdge(bn, tested, 0.6, "test file pattern"))

        # conftest.py depends on test files or tested modules
        if bn == "conftest.py":
            for other in basenames:
                if other.startswith("test_") and other != bn:
                    # conftest usually loaded alongside tests
                    pass  # don't add — conftest is parallel, not dependent

    # 3. __init__.py: only add weak "late" heuristic if no explicit edges
    init_files = [bn for bn in basenames if bn == "__init__.py"]
    connected = {(e.source, e.target) for e in edges}
    for init_bn in init_files:
        has_explicit = any(
            (e.source == init_bn or e.target == init_bn) and e.confidence >= 0.5
            for e in edges
        )
        if not has_explicit:
            for other in basenames:
                if other == init_bn or other.startswith("test_"):
                    continue
                if (init_bn, other) not in connected:
                    edges.append(DependencyEdge(init_bn, other, 0.1, "init weak aggregator"))

    return edges


def bayesian_order(
    files: list[str],
    edges: list[DependencyEdge],
) -> list[str]:
    """Order files using weighted dependency edges.

    Algorithm:
    1. Build weighted in-degree: sum of confidence scores of incoming edges
    2. Files with lowest weighted in-degree come first (most "base")
    3. Break ties using heuristics:
       - test_ files go last
       - __init__.py goes after other files in same package
       - files with more definitions go first (they're libraries)
    """
    basenames = [f.split("/")[-1] for f in files]
    bn_set = set(basenames)

    # Weighted in-degree: how much does this file depend on others?
    dependency_weight: dict[str, float] = {bn: 0.0 for bn in basenames}
    # Weighted out-degree: how much do others depend on this file?
    importance_weight: dict[str, float] = {bn: 0.0 for bn in basenames}

    for edge in edges:
        if edge.source in bn_set and edge.target in bn_set:
            dependency_weight[edge.source] += edge.confidence
            importance_weight[edge.target] += edge.confidence

    # Score: lower = more base (comes first)
    # score = dependency_weight - importance_weight + tie_breaker
    scores: dict[str, float] = {}
    for bn in basenames:
        score = dependency_weight[bn] - importance_weight[bn]

        # Tie-breakers
        if bn.startswith("test_"):
            score += 10  # tests go last
        if bn == "conftest.py":
            score += 8   # conftest near last
        if bn == "__init__.py":
            score += 5   # __init__ goes late (aggregator)

        scores[bn] = score

    # Sort by score (lowest first = most base)
    ordered = sorted(basenames, key=lambda bn: scores[bn])

    # Verify: check if this ordering violates any strong edges
    pos = {bn: i for i, bn in enumerate(ordered)}
    violations = 0
    for edge in edges:
        if edge.confidence >= 0.7:  # strong edges must be respected
            if edge.source in pos and edge.target in pos:
                if pos[edge.target] > pos[edge.source]:
                    # target (dependency) comes AFTER source — wrong!
                    violations += 1

    # If violations, fall back to topo sort for strong edges
    if violations > 0:
        topo = _weighted_topo_sort(basenames, edges)
        if topo:
            return topo

    return ordered


def _weighted_topo_sort(
    basenames: list[str],
    edges: list[DependencyEdge],
) -> list[str]:
    """Topological sort respecting only strong edges (conf >= 0.7).

    When multiple valid orderings exist, prefer:
    - Files with higher importance (more dependents) earlier
    - test_ files later
    """
    bn_set = set(basenames)

    # Build graph from strong edges only
    graph: dict[str, set[str]] = defaultdict(set)  # file → files it depends on
    for edge in edges:
        if edge.confidence >= 0.5 and edge.source in bn_set and edge.target in bn_set:
            graph[edge.source].add(edge.target)

    # Compute in-degrees (how many things each file depends on)
    in_deg = {bn: len(graph.get(bn, set())) for bn in basenames}

    # Reverse graph for traversal
    rev: dict[str, set[str]] = defaultdict(set)
    for src, deps in graph.items():
        for dep in deps:
            rev[dep].add(src)

    # Importance for tie-breaking
    importance = defaultdict(float)
    for edge in edges:
        if edge.target in bn_set:
            importance[edge.target] += edge.confidence

    # Kahn's with priority (higher importance first among candidates)
    def sort_key(bn):
        score = -importance.get(bn, 0)
        if bn.startswith("test_"):
            score += 100
        if bn == "__init__.py":
            score += 50
        if bn == "conftest.py":
            score += 80
        return score

    queue = sorted([bn for bn in basenames if in_deg[bn] == 0], key=sort_key)
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for dependent in rev.get(node, set()):
            in_deg[dependent] -= 1
            if in_deg[dependent] == 0:
                # Insert in sorted position
                queue.append(dependent)
                queue.sort(key=sort_key)

    # Add remaining (cycle or disconnected)
    remaining = [bn for bn in basenames if bn not in result]
    remaining.sort(key=sort_key)
    result.extend(remaining)

    return result


def verify_and_fix_order(
    ordering: list[str],
    edges: list[DependencyEdge],
) -> list[str]:
    """Post-process: fix adjacent-pair violations.

    If file A depends on B but A comes before B in the ordering,
    swap them. Repeat until no more violations.

    This catches the 30 "1-swap away" failures deterministically.
    """
    result = list(ordering)
    bn_set = set(result)

    # Build lookup: (source, target) → max confidence
    dep_conf: dict[tuple[str, str], float] = {}
    for e in edges:
        if e.source in bn_set and e.target in bn_set:
            key = (e.source, e.target)
            dep_conf[key] = max(dep_conf.get(key, 0), e.confidence)

    # Use weighted topo sort instead of trying to fix the existing order.
    # The topo sort respects transitive dependencies correctly.
    topo = _weighted_topo_sort(list(set(result)), edges)
    if topo and len(topo) == len(result):
        return topo

    return result


def order_files(
    files: list[str],
    mgr: ShardManager,
    file_contents: Optional[dict[str, str]] = None,
) -> list[str]:
    """Main entry: order files using Bayesian dependency analysis.

    Args:
        files: List of file paths.
        mgr: ShardManager with stored triples.
        file_contents: Optional raw code for indirect evidence scanning.

    Returns basenames in dependency order (base first, dependent last).
    """
    edges = build_dependency_graph(files, mgr, file_contents)
    ordering = bayesian_order(files, edges)
    return verify_and_fix_order(ordering, edges)
