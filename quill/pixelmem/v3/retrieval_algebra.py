"""PixelMem V3 Retrieval Algebra -- deterministic operators over PNG pixel matrices.

A complete toolkit of 20+ operators for entity resolution, scanning, filtering,
aggregation, graph traversal, workflow analysis, chunk retrieval, set operations,
and projection. Every operator is a standalone pure function that works against
real ShardManager instances.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pixelmem.memory import PixelMemUnit, BLACK
from pixelmem.shard_manager import ShardManager


# ---------------------------------------------------------------------------
# Fact dataclass
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """A single structured fact extracted from the pixel matrix.

    Fields:
        subject:    Source entity name (canonical, lowercase).
        relation:   Relation label decoded from the pixel color.
        object:     Target entity name (canonical, lowercase).
        condition:  Optional metadata string (date, qualifier, etc.).
        shard_idx:  Index of the shard this fact came from (-1 if unknown).
        chunk_idx:  Index of the chunk within the shard (-1 if flat scan).
        confidence: Confidence score in [0, 1]. Always 1.0 for matrix reads.
    """

    subject: str
    relation: str
    object: str
    condition: str = ""
    shard_idx: int = -1
    chunk_idx: int = -1
    confidence: float = 1.0

    def key(self) -> tuple[str, str, str]:
        """Return the (subject, relation, object) identity key."""
        return (self.subject, self.relation, self.object)

    def to_text(self) -> str:
        cond = f" [{self.condition}]" if self.condition else ""
        return f"{self.subject} {self.relation} {self.object}{cond}"


# ===================================================================
#  Internal helpers
# ===================================================================

def _canonicalize(name: str) -> str:
    """Lowercase and strip a surface form."""
    return name.strip().lower()


def _scan_entity_flat(mgr: ShardManager, entity: str) -> list[Fact]:
    """Core flat matrix reader -- scan row and column for *entity* across all shards.

    For each shard that contains the entity, reads every non-BLACK pixel in
    the entity's row (outgoing edges) and column (incoming edges).

    Returns:
        List of Fact objects with shard_idx populated.
    """
    canonical = _canonicalize(entity)
    facts: list[Fact] = []

    for si, shard in enumerate(mgr.shards):
        if canonical not in shard.entity_to_idx:
            continue
        idx = shard.entity_to_idx[canonical]

        # Row scan (entity as subject)
        for j in range(shard.n):
            rgb = tuple(int(x) for x in shard.relation[idx, j])
            if rgb == BLACK:
                continue
            obj = shard.idx_to_entity.get(j, f"entity_{j}")
            rel = shard.color_to_relation.get(rgb, f"rel_{rgb}")
            crgb = tuple(int(x) for x in shard.condition[idx, j])
            cond = shard.color_to_condition.get(crgb, "")
            facts.append(Fact(
                subject=canonical, relation=rel, object=obj,
                condition=cond, shard_idx=si,
            ))

        # Column scan (entity as object)
        for i in range(shard.n):
            if i == idx:
                continue
            rgb = tuple(int(x) for x in shard.relation[i, idx])
            if rgb == BLACK:
                continue
            subj = shard.idx_to_entity.get(i, f"entity_{i}")
            rel = shard.color_to_relation.get(rgb, f"rel_{rgb}")
            crgb = tuple(int(x) for x in shard.condition[i, idx])
            cond = shard.color_to_condition.get(crgb, "")
            facts.append(Fact(
                subject=subj, relation=rel, object=canonical,
                condition=cond, shard_idx=si,
            ))

    return facts


def _scan_entity_chunks(mgr: ShardManager, entity: str) -> list[Fact]:
    """Chunk-based scan -- reconstruct chunks from tapes and collect those
    that mention *entity*."""
    from pixelmem.decoder import reconstruct_chunks

    canonical = _canonicalize(entity)
    facts: list[Fact] = []

    for si, shard in enumerate(mgr.shards):
        if canonical not in shard.entity_to_idx:
            continue

        chunks = reconstruct_chunks(shard)
        for ci, chunk in enumerate(chunks):
            # Check if entity appears in any triple of the chunk
            dominated = False
            for t in chunk:
                if _canonicalize(t.subject) == canonical or _canonicalize(t.object) == canonical:
                    dominated = True
                    break
            if not dominated:
                continue
            for t in chunk:
                facts.append(Fact(
                    subject=_canonicalize(t.subject),
                    relation=t.relation,
                    object=_canonicalize(t.object),
                    condition=t.condition or "",
                    shard_idx=si,
                    chunk_idx=ci,
                ))

    return facts


def _all_entity_names(mgr: ShardManager) -> set[str]:
    """Collect every entity name from every shard."""
    names: set[str] = set()
    for shard in mgr.shards:
        names.update(shard.entity_to_idx.keys())
    return names


# ===================================================================
#  Entity Resolution
# ===================================================================

def resolve(mgr: ShardManager, surface: str) -> list[str]:
    """Resolve a surface form to canonical entity names.

    Strategy (in priority order):
      1. Exact match
      2. Substring match (both directions)
      3. Word-overlap match (split on _ and spaces)
      4. Alias lookup via ``is`` / ``alias`` / ``also_known_as`` / ``same_as``

    Returns:
        Sorted list of unique matching entity names.
    """
    canonical = _canonicalize(surface)
    all_ents = _all_entity_names(mgr)
    matches: set[str] = set()

    # 1. Exact
    if canonical in all_ents:
        matches.add(canonical)

    # 2. File-path basename match (shard_manager.py → pixelmem/shard_manager.py)
    for ent in all_ents:
        if ent.endswith("/" + canonical) or ent.endswith(os.sep + canonical):
            matches.add(ent)
        # Also match without extension: "shard_manager" → "pixelmem/shard_manager.py"
        ent_base = ent.rsplit("/", 1)[-1].rsplit(".", 1)[0] if ("/" in ent or "." in ent) else ent
        if canonical == ent_base or canonical.replace("_", "") == ent_base.replace("_", ""):
            matches.add(ent)

    # 3. Substring match — only if canonical is a substantial part of the entity
    #    (avoid matching "shard" to every entity containing "shard")
    if len(canonical) > 4:
        for ent in all_ents:
            if ent in matches:
                continue
            # Only match if canonical is >50% of the entity or entity is >50% of canonical
            if canonical in ent and len(canonical) > len(ent) * 0.4:
                matches.add(ent)
            elif ent in canonical and len(ent) > len(canonical) * 0.4:
                matches.add(ent)

    # 4. Word overlap — require >50% of query words to match
    if len(canonical) > 2 and not matches:
        query_words = set(canonical.replace("_", " ").split())
        for ent in all_ents:
            ent_words = set(ent.replace("_", " ").split())
            overlap = len(query_words & ent_words)
            if overlap > 0 and overlap >= len(query_words) * 0.5:
                matches.add(ent)

    # 4. Alias relations
    if canonical in all_ents:
        alias_rels = {"is", "alias", "also_known_as", "same_as"}
        facts = _scan_entity_flat(mgr, canonical)
        for f in facts:
            if f.relation in alias_rels:
                target = f.object if f.subject == canonical else f.subject
                matches.add(target)

    # Sort by match quality: exact first, then basename match, then others
    def _match_score(ent: str) -> int:
        if ent == canonical:
            return 0  # exact
        ent_base = ent.rsplit("/", 1)[-1].rsplit(".", 1)[0] if ("/" in ent or "." in ent) else ent
        if canonical == ent_base:
            return 1  # basename exact
        if canonical in ent_base or ent_base in canonical:
            return 2  # basename substring
        return 3  # other
    return sorted(matches, key=lambda e: (_match_score(e), e))


_PRONOUN_MAP = {"i", "my", "me", "mine", "myself", "we", "our"}


def resolve_pronoun(token: str) -> str:
    """Map first-person pronouns to the canonical ``user`` entity.

    Returns:
        ``"user"`` if the token is a first-person pronoun, otherwise
        the lowercased token unchanged.
    """
    lower = token.strip().lower()
    if lower in _PRONOUN_MAP:
        return "user"
    return lower


# ===================================================================
#  Scan Operators
# ===================================================================

def scan_entity(mgr: ShardManager, entity: str, limit: int = 50) -> list[Fact]:
    """Row + column scan across all shards for *entity*.

    Tries both chunk-based and flat-matrix scanning and returns whichever
    produces more results (flat is more complete for workflow entities where
    chunk tapes may not index everything).

    Args:
        mgr:    The ShardManager to scan.
        entity: Surface form of the entity to look up.
        limit:  Maximum number of facts to return.

    Returns:
        List of Fact objects, at most *limit* entries.
    """
    canonical = resolve_pronoun(entity)
    flat = _scan_entity_flat(mgr, canonical)
    chunk = _scan_entity_chunks(mgr, canonical)
    facts = chunk if len(chunk) > len(flat) else flat
    return facts[:limit]


def scan_relation(mgr: ShardManager, relation: str) -> list[Fact]:
    """Find all facts across all shards that carry the given relation.

    Scans every shard's relation color map to identify the RGB encoding,
    then reads every non-BLACK pixel in the relation matrix that matches.

    Returns:
        List of Fact objects for every occurrence of *relation*.
    """
    rel_lower = _canonicalize(relation)
    facts: list[Fact] = []

    for si, shard in enumerate(mgr.shards):
        # Find RGB for this relation in the shard's color space
        rgb = shard.relation_color_map.get(rel_lower)
        if rgb is None:
            # Try substring match on relation names
            for rname, rrgb in shard.relation_color_map.items():
                if rel_lower in rname or rname in rel_lower:
                    rgb = rrgb
                    break
        if rgb is None:
            continue

        # Full matrix scan for this color
        for i in range(shard.n):
            for j in range(shard.n):
                pixel = tuple(int(x) for x in shard.relation[i, j])
                if pixel != rgb:
                    continue
                subj = shard.idx_to_entity.get(i, f"entity_{i}")
                obj = shard.idx_to_entity.get(j, f"entity_{j}")
                crgb = tuple(int(x) for x in shard.condition[i, j])
                cond = shard.color_to_condition.get(crgb, "")
                facts.append(Fact(
                    subject=subj, relation=shard.color_to_relation.get(rgb, rel_lower),
                    object=obj, condition=cond, shard_idx=si,
                ))

    return facts


def scan_pair(mgr: ShardManager, entity_a: str, entity_b: str) -> list[Fact]:
    """Find all direct relations between two entities (both directions).

    Checks (a -> b) and (b -> a) in every shard that contains both.

    Returns:
        List of Fact objects for direct edges between a and b.
    """
    a = _canonicalize(entity_a)
    b = _canonicalize(entity_b)
    facts: list[Fact] = []

    for si, shard in enumerate(mgr.shards):
        if a not in shard.entity_to_idx or b not in shard.entity_to_idx:
            continue
        idx_a = shard.entity_to_idx[a]
        idx_b = shard.entity_to_idx[b]

        # a -> b
        rgb = tuple(int(x) for x in shard.relation[idx_a, idx_b])
        if rgb != BLACK:
            rel = shard.color_to_relation.get(rgb, f"rel_{rgb}")
            crgb = tuple(int(x) for x in shard.condition[idx_a, idx_b])
            cond = shard.color_to_condition.get(crgb, "")
            facts.append(Fact(a, rel, b, cond, si))

        # b -> a
        rgb = tuple(int(x) for x in shard.relation[idx_b, idx_a])
        if rgb != BLACK:
            rel = shard.color_to_relation.get(rgb, f"rel_{rgb}")
            crgb = tuple(int(x) for x in shard.condition[idx_b, idx_a])
            cond = shard.color_to_condition.get(crgb, "")
            facts.append(Fact(b, rel, a, cond, si))

    return facts


def scan_submatrix(mgr: ShardManager, entities: list[str]) -> list[Fact]:
    """Decode the |E| x |E| sub-matrix for a set of entities.

    For every pair (i, j) where both i and j are in *entities*, reads
    the relation pixel. This is the pixel-native analog of a multi-entity
    join.

    Returns:
        List of Fact objects from the sub-matrix intersections.
    """
    canonical_set = {_canonicalize(e) for e in entities}
    facts: list[Fact] = []

    for si, shard in enumerate(mgr.shards):
        # Find indices present in this shard
        present = {}
        for ent in canonical_set:
            if ent in shard.entity_to_idx:
                present[ent] = shard.entity_to_idx[ent]
        if len(present) < 2:
            continue

        idx_list = list(present.values())
        name_map = {v: k for k, v in present.items()}

        for i in idx_list:
            for j in idx_list:
                rgb = tuple(int(x) for x in shard.relation[i, j])
                if rgb == BLACK:
                    continue
                rel = shard.color_to_relation.get(rgb, f"rel_{rgb}")
                crgb = tuple(int(x) for x in shard.condition[i, j])
                cond = shard.color_to_condition.get(crgb, "")
                facts.append(Fact(
                    name_map[i], rel, name_map[j], cond, si,
                ))

    return facts


# ===================================================================
#  Filter Operators
# ===================================================================

def filter_relation(facts: list[Fact], relation: str) -> list[Fact]:
    """Keep only facts whose relation matches (substring, case-insensitive).

    Args:
        facts:    Input fact list.
        relation: Relation pattern to match.

    Returns:
        Filtered list of Fact objects.
    """
    rel_lower = _canonicalize(relation)
    return [f for f in facts if rel_lower in f.relation or f.relation in rel_lower]


def filter_condition(facts: list[Fact], pattern: str) -> list[Fact]:
    """Keep only facts whose condition matches a regex pattern.

    Args:
        facts:   Input fact list.
        pattern: Regular expression to match against the condition string.

    Returns:
        Filtered list of Fact objects.
    """
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        # Fall back to plain substring match
        pat_lower = pattern.lower()
        return [f for f in facts if pat_lower in f.condition.lower()]
    return [f for f in facts if regex.search(f.condition)]


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def filter_date_range(facts: list[Fact], start: str, end: str) -> list[Fact]:
    """Keep facts whose condition contains a date within [start, end].

    Dates are expected in ``YYYY-MM-DD`` format. Facts without a parseable
    date in their condition are excluded.

    Args:
        facts: Input fact list.
        start: Start date inclusive (``YYYY-MM-DD``).
        end:   End date inclusive (``YYYY-MM-DD``).

    Returns:
        Filtered list of Fact objects.
    """
    results: list[Fact] = []
    for f in facts:
        m = _DATE_RE.search(f.condition)
        if m:
            d = m.group(1)
            if start <= d <= end:
                results.append(f)
    return results


def filter_shard(facts: list[Fact], shard_idx: int) -> list[Fact]:
    """Keep only facts from a specific shard.

    Args:
        facts:     Input fact list.
        shard_idx: Shard index to keep.

    Returns:
        Filtered list of Fact objects.
    """
    return [f for f in facts if f.shard_idx == shard_idx]


# ===================================================================
#  Aggregation
# ===================================================================

def count_facts(facts: list[Fact], group_by: str = "relation") -> dict[str, int]:
    """Count facts grouped by a field.

    Args:
        facts:    Input fact list.
        group_by: One of ``"relation"``, ``"subject"``, ``"object"``,
                  ``"condition"``, ``"shard_idx"``.

    Returns:
        Dict mapping group key to count.
    """
    counts: dict[str, int] = defaultdict(int)
    for f in facts:
        key = str(getattr(f, group_by, "unknown"))
        counts[key] += 1
    return dict(counts)


_NUMBER_RE = re.compile(r"[\d,]+\.?\d*")


def sum_numeric(facts: list[Fact]) -> dict:
    """Extract and sum numeric values from fact objects and conditions.

    Scans the ``object`` and ``condition`` fields for numbers.

    Returns:
        Dict with ``total``, ``count``, and ``parsed_values``.
    """
    total = 0.0
    parsed: list[float] = []
    for f in facts:
        for text in (f.object, f.condition):
            nums = _NUMBER_RE.findall(text.replace(",", ""))
            for n in nums:
                try:
                    val = float(n)
                    total += val
                    parsed.append(val)
                except ValueError:
                    continue
    return {
        "total": total,
        "count": len(parsed),
        "parsed_values": parsed,
    }


def date_diff(date_a: str, date_b: str) -> dict:
    """Compute the difference between two dates.

    Accepts dates in ``YYYY-MM-DD`` format.

    Returns:
        Dict with ``days``, ``weeks``, ``months``, and ``description``.
        On parse failure, returns a dict with an ``error`` key.
    """
    try:
        da = datetime.strptime(date_a.strip(), "%Y-%m-%d")
        db = datetime.strptime(date_b.strip(), "%Y-%m-%d")
    except ValueError:
        return {"error": f"Cannot parse dates: {date_a}, {date_b}"}

    delta = abs((da - db).days)
    weeks = round(delta / 7, 1)
    months = round(delta / 30.44, 1)

    return {
        "days": delta,
        "weeks": weeks,
        "months": months,
        "description": f"{delta} days ({weeks} weeks)",
    }


def most_recent(facts: list[Fact]) -> list[Fact]:
    """Sort facts by the date found in their condition, most recent first.

    Facts without a parseable date are placed at the end.

    Returns:
        A new list of Fact objects sorted descending by date.
    """
    def _date_key(f: Fact) -> str:
        m = _DATE_RE.search(f.condition)
        return m.group(1) if m else "0000-00-00"

    return sorted(facts, key=_date_key, reverse=True)


# ===================================================================
#  Graph Traversal
# ===================================================================

def neighbors(mgr: ShardManager, entity: str, depth: int = 1) -> set[str]:
    """Multi-hop neighbor discovery.

    Starting from *entity*, follows outgoing and incoming edges up to
    *depth* hops. Returns the set of all discovered entity names
    (excluding the starting entity itself).

    Args:
        mgr:    ShardManager to query.
        entity: Starting entity.
        depth:  Maximum number of hops (1 = direct neighbors only).

    Returns:
        Set of entity names reachable within *depth* hops.
    """
    canonical = _canonicalize(entity)
    visited: set[str] = {canonical}
    frontier: set[str] = {canonical}

    for _ in range(depth):
        next_frontier: set[str] = set()
        for ent in frontier:
            facts = _scan_entity_flat(mgr, ent)
            for f in facts:
                for name in (f.subject, f.object):
                    if name not in visited:
                        visited.add(name)
                        next_frontier.add(name)
        frontier = next_frontier
        if not frontier:
            break

    visited.discard(canonical)
    return visited


def path_between(
    mgr: ShardManager,
    entity_a: str,
    entity_b: str,
    max_hops: int = 3,
) -> list[list[Fact]]:
    """BFS shortest path(s) between two entities.

    Returns all shortest paths (as lists of Fact edges) from *entity_a*
    to *entity_b*, up to *max_hops* edges long.

    Args:
        mgr:      ShardManager to query.
        entity_a: Start entity.
        entity_b: End entity.
        max_hops: Maximum path length.

    Returns:
        List of paths. Each path is a list of Fact objects representing
        edges along the path. Empty list if no path found.
    """
    a = _canonicalize(entity_a)
    b = _canonicalize(entity_b)

    if a == b:
        return [[]]

    # BFS: each queue entry is (current_entity, path_so_far)
    queue: deque[tuple[str, list[Fact]]] = deque()
    queue.append((a, []))
    visited: set[str] = {a}
    found_paths: list[list[Fact]] = []
    found_depth: int | None = None

    while queue:
        current, path = queue.popleft()

        # Prune if we already found shorter paths
        if found_depth is not None and len(path) >= found_depth:
            continue
        if len(path) >= max_hops:
            continue

        facts = _scan_entity_flat(mgr, current)
        for f in facts:
            # Determine the neighbor from this edge
            if _canonicalize(f.subject) == current:
                neighbor = _canonicalize(f.object)
            elif _canonicalize(f.object) == current:
                neighbor = _canonicalize(f.subject)
            else:
                continue

            new_path = path + [f]

            if neighbor == b:
                found_paths.append(new_path)
                found_depth = len(new_path)
                continue

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, new_path))

    return found_paths


def connected_component(
    mgr: ShardManager,
    entity: str,
    max_size: int = 100,
) -> set[str]:
    """Find the connected component containing *entity*.

    Performs an unbounded BFS (up to *max_size* nodes) following all
    edges in both directions.

    Args:
        mgr:      ShardManager to query.
        entity:   Starting entity.
        max_size: Safety limit on component size.

    Returns:
        Set of all entity names in the connected component.
    """
    canonical = _canonicalize(entity)
    visited: set[str] = {canonical}
    queue: deque[str] = deque([canonical])

    while queue and len(visited) < max_size:
        current = queue.popleft()
        facts = _scan_entity_flat(mgr, current)
        for f in facts:
            for name in (f.subject, f.object):
                if name not in visited:
                    visited.add(name)
                    queue.append(name)
                    if len(visited) >= max_size:
                        break
            if len(visited) >= max_size:
                break

    return visited


# ===================================================================
#  Workflow-Specific Operators
# ===================================================================

def topo_sort(mgr: ShardManager, relation: str = "imports") -> list[str]:
    """Topological sort on dependency edges defined by *relation*.

    Scans all facts with the given relation and builds a DAG. Returns
    entities in topological order (dependencies before dependents).
    If cycles are detected, returns partial ordering with remaining
    entities appended.

    Returns:
        List of entity names in topological order.
    """
    edges = scan_relation(mgr, relation)

    # Build adjacency list: subject depends on object
    # (subject imports object => object must come first)
    graph: dict[str, set[str]] = defaultdict(set)
    in_degree: dict[str, int] = defaultdict(int)
    all_nodes: set[str] = set()

    for f in edges:
        graph[f.object].add(f.subject)
        in_degree.setdefault(f.object, 0)
        in_degree[f.subject] = in_degree.get(f.subject, 0) + 1
        all_nodes.add(f.subject)
        all_nodes.add(f.object)

    # Kahn's algorithm
    queue: deque[str] = deque()
    for node in all_nodes:
        if in_degree.get(node, 0) == 0:
            queue.append(node)

    result: list[str] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.get(node, set()):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Append any remaining nodes (cycle members)
    remaining = all_nodes - set(result)
    result.extend(sorted(remaining))

    return result


def find_subclasses(mgr: ShardManager, class_name: str) -> list[str]:
    """Reverse lookup: find all entities that extend *class_name*.

    Searches for ``extends``, ``inherits``, ``subclass_of``, and
    ``is_a`` relations pointing to *class_name*.

    Returns:
        Sorted list of subclass entity names.
    """
    canonical = _canonicalize(class_name)
    extend_rels = {"extends", "inherits", "subclass_of", "is_a"}
    subclasses: set[str] = set()

    # Scan the parent entity for incoming edges
    facts = _scan_entity_flat(mgr, canonical)
    for f in facts:
        if f.relation in extend_rels and _canonicalize(f.object) == canonical:
            subclasses.add(f.subject)

    # Also scan all shards for these relations explicitly
    for rel in extend_rels:
        rel_facts = scan_relation(mgr, rel)
        for f in rel_facts:
            if _canonicalize(f.object) == canonical:
                subclasses.add(f.subject)

    return sorted(subclasses)


def find_overrides(mgr: ShardManager, child: str, parent: str) -> list[str]:
    """Find methods defined in *child* that are not defined in *parent*.

    Looks for ``has_method``, ``defines``, or ``method`` relations on both
    entities and returns the set difference (child methods minus parent methods).

    Returns:
        Sorted list of method names that the child overrides or adds.
    """
    method_rels = {"has_method", "defines", "method"}

    child_facts = _scan_entity_flat(mgr, _canonicalize(child))
    parent_facts = _scan_entity_flat(mgr, _canonicalize(parent))

    child_methods: set[str] = set()
    parent_methods: set[str] = set()

    for f in child_facts:
        if f.relation in method_rels:
            child_methods.add(f.object)
    for f in parent_facts:
        if f.relation in method_rels:
            parent_methods.add(f.object)

    return sorted(child_methods - parent_methods)


def resolve_imports(mgr: ShardManager, file_path: str) -> list[Fact]:
    """Follow the import chain from *file_path* to the definitions it depends on.

    Recursively follows ``imports`` edges to collect the transitive closure
    of all imported entities, up to 5 levels deep.

    Returns:
        List of Fact objects representing the full import chain.
    """
    canonical = _canonicalize(file_path)
    all_facts: list[Fact] = []
    visited: set[str] = set()
    frontier: set[str] = {canonical}
    max_depth = 5

    for _ in range(max_depth):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for ent in frontier:
            if ent in visited:
                continue
            visited.add(ent)
            facts = _scan_entity_flat(mgr, ent)
            import_facts = [f for f in facts if "import" in f.relation]
            all_facts.extend(import_facts)
            for f in import_facts:
                target = f.object if f.subject == ent else f.subject
                if target not in visited:
                    next_frontier.add(target)
        frontier = next_frontier

    return all_facts


def search_docstrings(mgr: ShardManager, keywords: list[str]) -> list[Fact]:
    """Search stored docstring/description facts for keyword matches.

    Looks for facts whose relation contains ``doc``, ``description``,
    ``docstring``, or ``comment``, and whose object or condition text
    matches any of the provided keywords.

    Returns:
        List of matching Fact objects sorted by number of keyword hits
        (descending).
    """
    doc_rels = {"doc", "description", "docstring", "comment", "documentation"}
    kw_lower = [k.lower() for k in keywords]
    candidates: list[tuple[int, Fact]] = []

    for si, shard in enumerate(mgr.shards):
        # Identify doc-like relations in this shard
        for rel_name, rgb in shard.relation_color_map.items():
            is_doc = any(d in rel_name for d in doc_rels)
            if not is_doc:
                continue
            # Scan full matrix for this color
            for i in range(shard.n):
                for j in range(shard.n):
                    pixel = tuple(int(x) for x in shard.relation[i, j])
                    if pixel != rgb:
                        continue
                    subj = shard.idx_to_entity.get(i, f"entity_{i}")
                    obj = shard.idx_to_entity.get(j, f"entity_{j}")
                    crgb = tuple(int(x) for x in shard.condition[i, j])
                    cond = shard.color_to_condition.get(crgb, "")

                    text = f"{obj} {cond}".lower()
                    hits = sum(1 for kw in kw_lower if kw in text)
                    if hits > 0:
                        fact = Fact(subj, rel_name, obj, cond, si)
                        candidates.append((hits, fact))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in candidates]


# ===================================================================
#  Chunk Operators
# ===================================================================

def get_chunks(mgr: ShardManager, entity: str) -> list[list[Fact]]:
    """Chunk-preserving retrieval for *entity*.

    Reconstructs chunks from tapes and returns all chunks that contain
    a triple mentioning the entity. Each chunk is a list of Facts that
    were encoded together from the same knowledge source.

    Returns:
        List of chunks (each chunk is a list of Fact objects).
    """
    from pixelmem.decoder import reconstruct_chunks

    canonical = _canonicalize(entity)
    result_chunks: list[list[Fact]] = []

    for si, shard in enumerate(mgr.shards):
        if canonical not in shard.entity_to_idx:
            continue

        chunks = reconstruct_chunks(shard)
        for ci, chunk in enumerate(chunks):
            match = False
            for t in chunk:
                if _canonicalize(t.subject) == canonical or _canonicalize(t.object) == canonical:
                    match = True
                    break
            if not match:
                continue

            fact_chunk: list[Fact] = []
            for t in chunk:
                fact_chunk.append(Fact(
                    subject=_canonicalize(t.subject),
                    relation=t.relation,
                    object=_canonicalize(t.object),
                    condition=t.condition or "",
                    shard_idx=si,
                    chunk_idx=ci,
                ))
            result_chunks.append(fact_chunk)

    return result_chunks


def chunk_search(mgr: ShardManager, query_tokens: list[str]) -> list[list[Fact]]:
    """BM25-style scoring over chunk content.

    Reconstructs all chunks from all shards, scores each chunk by
    term-frequency / inverse-document-frequency of the query tokens
    against the concatenated text of each chunk's facts.

    Returns:
        List of chunks sorted by descending BM25 score, each chunk
        a list of Fact objects.
    """
    from pixelmem.decoder import reconstruct_chunks

    tokens_lower = [t.lower() for t in query_tokens]
    if not tokens_lower:
        return []

    # Collect all chunks across shards
    all_chunks: list[tuple[list[Fact], str]] = []  # (facts, text)
    for si, shard in enumerate(mgr.shards):
        chunks = reconstruct_chunks(shard)
        for ci, chunk in enumerate(chunks):
            facts: list[Fact] = []
            text_parts: list[str] = []
            for t in chunk:
                facts.append(Fact(
                    subject=_canonicalize(t.subject),
                    relation=t.relation,
                    object=_canonicalize(t.object),
                    condition=t.condition or "",
                    shard_idx=si,
                    chunk_idx=ci,
                ))
                text_parts.extend([t.subject, t.relation, t.object, t.condition or ""])
            text = " ".join(text_parts).lower()
            all_chunks.append((facts, text))

    if not all_chunks:
        return []

    # IDF: count how many chunks contain each token
    n_docs = len(all_chunks)
    doc_freq: dict[str, int] = defaultdict(int)
    for _, text in all_chunks:
        for tok in set(tokens_lower):
            if tok in text:
                doc_freq[tok] += 1

    # BM25 scoring (k1=1.5, b=0.75)
    k1 = 1.5
    b = 0.75
    avg_len = sum(len(text.split()) for _, text in all_chunks) / n_docs

    scored: list[tuple[float, list[Fact]]] = []
    for facts, text in all_chunks:
        words = text.split()
        doc_len = len(words)
        score = 0.0
        for tok in tokens_lower:
            tf = words.count(tok) if tok in text else 0
            if tf == 0:
                continue
            df = doc_freq.get(tok, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
            score += idf * tf_norm
        if score > 0:
            scored.append((score, facts))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [facts for _, facts in scored]


# ===================================================================
#  Set Operators
# ===================================================================

def union(a: list[Fact], b: list[Fact]) -> list[Fact]:
    """Set union of two fact lists (preserves order, deduplicates by key).

    Returns:
        Combined list with duplicates removed (first occurrence wins).
    """
    seen: set[tuple[str, str, str]] = set()
    result: list[Fact] = []
    for f in a + b:
        k = f.key()
        if k not in seen:
            seen.add(k)
            result.append(f)
    return result


def intersect(a: list[Fact], b: list[Fact]) -> list[Fact]:
    """Set intersection by (subject, relation, object) key.

    Returns:
        Facts present in both *a* and *b* (uses *a*'s Fact instances).
    """
    b_keys = {f.key() for f in b}
    return [f for f in a if f.key() in b_keys]


def difference(a: list[Fact], b: list[Fact]) -> list[Fact]:
    """Set difference: facts in *a* but not in *b*.

    Returns:
        Facts from *a* whose key does not appear in *b*.
    """
    b_keys = {f.key() for f in b}
    return [f for f in a if f.key() not in b_keys]


def deduplicate(facts: list[Fact]) -> list[Fact]:
    """Remove duplicate facts (by subject+relation+object key).

    Preserves the first occurrence of each unique key.

    Returns:
        De-duplicated list of Fact objects.
    """
    seen: set[tuple[str, str, str]] = set()
    result: list[Fact] = []
    for f in facts:
        k = f.key()
        if k not in seen:
            seen.add(k)
            result.append(f)
    return result


# ===================================================================
#  Projection
# ===================================================================

def project(facts: list[Fact], fields: list[str]) -> list[dict]:
    """Project facts onto a subset of fields.

    Args:
        facts:  Input fact list.
        fields: Field names to keep (e.g. ``["subject", "object"]``).

    Returns:
        List of dicts, each containing only the specified fields.
    """
    result: list[dict] = []
    for f in facts:
        row = {}
        for field_name in fields:
            if hasattr(f, field_name):
                row[field_name] = getattr(f, field_name)
        result.append(row)
    return result


def group_by(facts: list[Fact], key: str) -> dict[str, list[Fact]]:
    """Group facts by a field value.

    Args:
        facts: Input fact list.
        key:   Field name to group by (e.g. ``"relation"``, ``"subject"``).

    Returns:
        Dict mapping field values to lists of Fact objects.
    """
    groups: dict[str, list[Fact]] = defaultdict(list)
    for f in facts:
        group_key = str(getattr(f, key, "unknown"))
        groups[group_key].append(f)
    return dict(groups)
