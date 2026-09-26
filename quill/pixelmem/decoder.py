"""PixelMem Decoding Pipeline (Algorithm 2 from the paper).

Reads pixel-encoded knowledge and reconstructs triples as text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pixelmem.memory import PixelMemUnit, BLACK
from pixelmem.triple_extractor import Triple


@dataclass
class DecodedTriple:
    """A decoded triple with full text labels."""
    subject: str
    relation: str
    object: str
    condition: str = ""
    chunk_id: Optional[int] = None

    def to_text(self) -> str:
        cond = f" [{self.condition}]" if self.condition else ""
        return f"{self.subject} --{self.relation}--> {self.object}{cond}"

    def __repr__(self) -> str:
        return self.to_text()


def _scan_entity(idx: int, unit: PixelMemUnit, visited: set[int]) -> list[DecodedTriple]:
    """Scan row and column for a single entity index. Returns decoded triples."""
    results: list[DecodedTriple] = []

    # Scan row i (entity as subject)
    for j in range(unit.n):
        rgb = tuple(int(x) for x in unit.relation[idx, j])
        if rgb == BLACK:
            continue
        obj_name = unit.idx_to_entity.get(j, f"entity_{j}")
        subj_name = unit.idx_to_entity.get(idx, f"entity_{idx}")
        relation = unit.color_to_relation.get(rgb, f"rel_{rgb}")
        cond_rgb = tuple(int(x) for x in unit.condition[idx, j])
        condition = unit.color_to_condition.get(cond_rgb, "")
        results.append(DecodedTriple(
            subject=subj_name, relation=relation,
            object=obj_name, condition=condition,
        ))

    # Scan column i (entity as object)
    for i in range(unit.n):
        if i in visited:
            continue
        rgb = tuple(int(x) for x in unit.relation[i, idx])
        if rgb == BLACK:
            continue
        subj_name = unit.idx_to_entity.get(i, f"entity_{i}")
        obj_name = unit.idx_to_entity.get(idx, f"entity_{idx}")
        relation = unit.color_to_relation.get(rgb, f"rel_{rgb}")
        cond_rgb = tuple(int(x) for x in unit.condition[i, idx])
        condition = unit.color_to_condition.get(cond_rgb, "")
        results.append(DecodedTriple(
            subject=subj_name, relation=relation,
            object=obj_name, condition=condition,
        ))

    return results


def decode_query(
    query_entities: list[str],
    unit: PixelMemUnit,
    include_chunks: bool = False,
    max_hops: int = 1,
) -> list[DecodedTriple]:
    """Decode triples related to the given entities from a PixelMem unit.

    Implements Algorithm 2 with multi-hop following:
      1. For each query entity, resolve to index
      2. Scan row i and column i in the relation matrix
      3. For non-black pixels, decode color -> relation name
      4. Read corresponding condition
      5. If max_hops > 1, follow discovered entities for additional hops
      6. Optionally reconstruct chunk context from tapes

    Multi-hop example (max_hops=2):
      Query: "person_5" -> finds "person_5 --manages--> person_0"
      Hop 2: follows "person_0" -> finds "person_0 --works_at--> org_4"
      Result: both triples returned, enabling "where does person_5's
      manager work?" to be answered.

    Args:
        query_entities: Entity names to query for.
        unit: The PixelMemUnit to read from.
        include_chunks: If True, also scan tapes to reconstruct chunk groupings.
        max_hops: Maximum number of hops to follow (1 = direct only,
            2 = follow one level of neighbors, etc.)

    Returns:
        List of decoded triples.
    """
    results: list[DecodedTriple] = []
    visited_indices: set[int] = set()

    # Resolve initial query entities — fuzzy matching
    # "commute" matches "commute_duration", "car" matches "car_gps"
    # "I"/"my" maps to "user"
    # Also searches relation names: "commute" finds entities linked by "daily_commute"
    _PRONOUNS = {"i", "my", "me", "mine", "myself", "we", "our"}
    frontier: set[int] = set()
    for name in query_entities:
        canonical = PixelMemUnit._canonicalize(name)
        if canonical in _PRONOUNS:
            canonical = "user"
        # Exact entity match
        if canonical in unit.entity_to_idx:
            frontier.add(unit.entity_to_idx[canonical])
            continue
        # Fuzzy entity match: substring both directions
        for ent_name, ent_idx in unit.entity_to_idx.items():
            if canonical in ent_name or ent_name in canonical:
                frontier.add(ent_idx)
            elif len(canonical) > 3:
                ent_words = set(ent_name.split("_"))
                if canonical in ent_words:
                    frontier.add(ent_idx)

        # Relation name match: "commute" finds all entities connected by
        # relations containing "commute" (e.g., "daily_commute")
        if len(canonical) > 3 and not frontier:
            for rel_name, rgb in unit.relation_color_map.items():
                if canonical in rel_name or rel_name in canonical:
                    # Find all entities connected by this relation
                    for i in range(unit.n):
                        for j in range(unit.n):
                            if tuple(unit.relation[i, j]) == rgb:
                                frontier.add(i)
                                frontier.add(j)

    for hop in range(max_hops):
        if not frontier:
            break

        next_frontier: set[int] = set()
        for idx in frontier:
            if idx in visited_indices:
                continue
            visited_indices.add(idx)

            hop_results = _scan_entity(idx, unit, visited_indices)
            results.extend(hop_results)

            # Collect neighbor entities for next hop
            if hop < max_hops - 1:
                for t in hop_results:
                    # Add discovered entities to next frontier
                    for name in [t.subject, t.object]:
                        canonical = PixelMemUnit._canonicalize(name)
                        if canonical in unit.entity_to_idx:
                            neighbor_idx = unit.entity_to_idx[canonical]
                            if neighbor_idx not in visited_indices:
                                next_frontier.add(neighbor_idx)

        frontier = next_frontier

    if include_chunks:
        _attach_chunk_ids(results, unit)

    return results


def decode_submatrix(
    entity_indices: set[int],
    unit: PixelMemUnit,
) -> list[DecodedTriple]:
    """Decode only the sub-matrix defined by the given entity indices.

    Instead of scanning the full N×N matrix, this reads only the
    intersections of the specified rows and columns — an |E|×|E|
    sub-matrix where E is the set of relevant entities.

    This is the pixel-native equivalent of text_relevant: it retrieves
    exactly the triples between known-relevant entities, nothing more.

    Args:
        entity_indices: Set of matrix indices to include.
        unit: The PixelMemUnit to read from.

    Returns:
        List of decoded triples within the sub-matrix.
    """
    results: list[DecodedTriple] = []
    for i in entity_indices:
        for j in entity_indices:
            rgb = tuple(int(x) for x in unit.relation[i, j])
            if rgb == BLACK:
                continue
            subj = unit.idx_to_entity.get(i, f"entity_{i}")
            obj = unit.idx_to_entity.get(j, f"entity_{j}")
            relation = unit.color_to_relation.get(rgb, f"rel_{rgb}")
            cond_rgb = tuple(int(x) for x in unit.condition[i, j])
            condition = unit.color_to_condition.get(cond_rgb, "")
            results.append(DecodedTriple(
                subject=subj, relation=relation, object=obj, condition=condition,
            ))
    return results


def decode_submatrix_multihop(
    query_entities: list[str],
    unit: PixelMemUnit,
    max_hops: int = 2,
) -> list[DecodedTriple]:
    """Decode sub-matrix with multi-hop expansion.

    1. Resolve query entities to indices
    2. Hop 1: scan their full rows/columns to discover neighbors
    3. Collect all discovered entity indices (query + neighbors)
    4. Decode only the sub-matrix of those indices

    This gives text_relevant-level precision with PixelMem storage:
    - Hop 1 finds which entities are connected to the query
    - Sub-matrix decode gets all relationships between those entities
    - Result: complete local neighborhood, no irrelevant triples

    Args:
        query_entities: Entity names to start from.
        unit: The PixelMemUnit to read from.
        max_hops: How many expansion hops (1=direct neighbors only).

    Returns:
        List of decoded triples in the expanded sub-matrix.
    """
    # Resolve initial entities
    relevant_indices: set[int] = set()
    for name in query_entities:
        canonical = PixelMemUnit._canonicalize(name)
        if canonical in unit.entity_to_idx:
            relevant_indices.add(unit.entity_to_idx[canonical])

    if not relevant_indices:
        return []

    # Expand via row/column scanning to discover neighbors
    frontier = set(relevant_indices)
    for hop in range(max_hops):
        if not frontier:
            break
        next_frontier: set[int] = set()
        for idx in frontier:
            # Scan row (outgoing edges)
            for j in range(unit.n):
                if tuple(unit.relation[idx, j]) != BLACK:
                    next_frontier.add(j)
            # Scan column (incoming edges)
            for i in range(unit.n):
                if tuple(unit.relation[i, idx]) != BLACK:
                    next_frontier.add(i)
        new = next_frontier - relevant_indices
        relevant_indices.update(next_frontier)
        frontier = new

    # Now decode only the sub-matrix of relevant entities
    return decode_submatrix(relevant_indices, unit)


def decode_all(unit: PixelMemUnit) -> list[DecodedTriple]:
    """Decode ALL non-black triples from the relation matrix."""
    results = []
    for i in range(unit.n):
        for j in range(unit.n):
            rgb = tuple(int(x) for x in unit.relation[i, j])
            if rgb == BLACK:
                continue
            subj = unit.idx_to_entity.get(i, f"entity_{i}")
            obj = unit.idx_to_entity.get(j, f"entity_{j}")
            relation = unit.color_to_relation.get(rgb, f"rel_{rgb}")
            cond_rgb = tuple(int(x) for x in unit.condition[i, j])
            condition = unit.color_to_condition.get(cond_rgb, "")
            results.append(DecodedTriple(
                subject=subj, relation=relation, object=obj, condition=condition,
            ))
    return results


def reconstruct_chunks(unit: PixelMemUnit) -> list[list[DecodedTriple]]:
    """Reconstruct all chunks from the tape images.

    Reads tapes position-by-position. Black (0,0,0) on all three tapes
    simultaneously marks chunk boundaries.
    """
    chunks: list[list[DecodedTriple]] = []
    current_chunk: list[DecodedTriple] = []

    tape_len = min(len(unit.tape_r), len(unit.tape_c), len(unit.tape_e))

    for k in range(tape_len):
        r_px = unit.tape_r[k]
        c_px = unit.tape_c[k]
        e_px = unit.tape_e[k]

        # Check for stop signal (all three tapes are black)
        if r_px == BLACK and c_px == BLACK and e_px == BLACK:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
            continue

        # Decode this tape position
        i, j = e_px[0], e_px[1]  # entity coordinates from Tape E
        subj = unit.idx_to_entity.get(i, f"entity_{i}")
        obj = unit.idx_to_entity.get(j, f"entity_{j}")

        r_tuple = tuple(r_px) if not isinstance(r_px, tuple) else r_px
        c_tuple = tuple(c_px) if not isinstance(c_px, tuple) else c_px

        relation = unit.color_to_relation.get(r_tuple, f"rel_{r_tuple}")
        condition = unit.color_to_condition.get(c_tuple, "")

        current_chunk.append(DecodedTriple(
            subject=subj,
            relation=relation,
            object=obj,
            condition=condition,
            chunk_id=len(chunks),
        ))

    # Don't forget the last chunk if tape doesn't end with a delimiter
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def scan_entity_chunks(
    entity: str,
    unit: PixelMemUnit,
) -> list[list[DecodedTriple]]:
    """Scan for an entity and return all CHUNKS containing it.

    Like APR's continuous chunk retrieval: instead of returning isolated
    triples, returns the full chunk (all triples encoded together from
    the same knowledge source). This preserves context.

    Example: if searching for "chandelier", returns the entire chunk:
      - user met_with aunt [march 2026]
      - aunt gave crystal chandelier [as gift]
      - chandelier is vintage piece
      - user plans_to hang chandelier [in dining room]

    Instead of just: user received crystal chandelier
    """
    canonical = PixelMemUnit._canonicalize(entity)
    if canonical not in unit.entity_to_idx:
        return []

    idx = unit.entity_to_idx[canonical]

    # Reconstruct all chunks from tapes
    all_chunks = reconstruct_chunks(unit)

    # Find chunks that contain any triple involving this entity
    matching_chunks = []
    for chunk in all_chunks:
        for t in chunk:
            subj_canon = PixelMemUnit._canonicalize(t.subject)
            obj_canon = PixelMemUnit._canonicalize(t.object)
            if subj_canon == canonical or obj_canon == canonical:
                matching_chunks.append(chunk)
                break

    return matching_chunks


def _attach_chunk_ids(
    results: list[DecodedTriple], unit: PixelMemUnit
) -> None:
    """Attach chunk IDs to results by scanning tapes for matching coordinates."""
    chunks = reconstruct_chunks(unit)
    # Build a lookup: (subject, relation, object) -> chunk_id
    lookup: dict[tuple[str, str, str], int] = {}
    for chunk_id, chunk in enumerate(chunks):
        for dt in chunk:
            lookup[(dt.subject, dt.relation, dt.object)] = chunk_id

    for r in results:
        key = (r.subject, r.relation, r.object)
        if key in lookup:
            r.chunk_id = lookup[key]


def triples_to_apr_json(triples: list[DecodedTriple]) -> dict:
    """Convert decoded quadruples to APR-style compact JSON with indices.

    Like AutoPrunedRetriever: dictionaries + indexed quadruples (e, r, e, c).
    Each string stored once in its dictionary, quadruples reference by index.

    Output format:
      {
        "e": ["user", "acme", ...],                  # entity dictionary
        "r": ["works_at", "lives_in", ...],           # relation dictionary
        "c": ["since 2022", "each way", ...],         # condition dictionary
        "q": [[0,0,1,0], [0,1,2,-1], ...],            # quadruples [s,r,o,c] indices
      }

    -1 in condition position means no condition.
    """
    entities: list[str] = []
    relations: list[str] = []
    conditions: list[str] = []
    e2i: dict[str, int] = {}
    r2i: dict[str, int] = {}
    c2i: dict[str, int] = {}

    def _intern_e(s):
        if s not in e2i:
            e2i[s] = len(entities)
            entities.append(s)
        return e2i[s]

    def _intern_r(s):
        if s not in r2i:
            r2i[s] = len(relations)
            relations.append(s)
        return r2i[s]

    def _intern_c(s):
        if not s:
            return -1
        if s not in c2i:
            c2i[s] = len(conditions)
            conditions.append(s)
        return c2i[s]

    quads = []
    for t in triples:
        si = _intern_e(t.subject)
        ri = _intern_r(t.relation)
        oi = _intern_e(t.object)
        ci = _intern_c(t.condition)
        quads.append([si, ri, oi, ci])

    return {
        "e": entities,
        "r": relations,
        "c": conditions,
        "q": quads,
    }


def triples_to_image_context(
    triples: list[DecodedTriple],
    unit: "PixelMemUnit",
    entity_indices: set[int] | None = None,
    block_size: int = 8,
) -> tuple[list[str], str]:
    """Render PixelMem as images for VLM consumption — the three-tape format from the paper.

    Generates:
      1. relation.png — sub-matrix of relevant entities (NxN, colored pixels)
      2. condition.png — parallel condition matrix
      3. tapes.png — sequential tape visualization (R, C, E stacked)

    Plus a text legend mapping entity indices, colors to names.

    Returns (list_of_image_paths, legend_text).
    """
    import tempfile
    from PIL import Image
    import numpy as np

    if entity_indices is None:
        entity_indices = set()
        for t in triples:
            canonical = unit._canonicalize(t.subject)
            if canonical in unit.entity_to_idx:
                entity_indices.add(unit.entity_to_idx[canonical])
            canonical = unit._canonicalize(t.object)
            if canonical in unit.entity_to_idx:
                entity_indices.add(unit.entity_to_idx[canonical])

    idx_list = sorted(entity_indices)
    n = len(idx_list)
    if n == 0:
        return [], "No entities found."

    tmpdir = tempfile.mkdtemp(prefix="pixelmem_img_")
    paths = []

    # 1. Relation matrix (sub-matrix)
    img_size = n * block_size
    rel_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cond_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    for ii, i in enumerate(idx_list):
        for jj, j in enumerate(idx_list):
            rgb = tuple(int(x) for x in unit.relation[i, j])
            if rgb != (0, 0, 0):
                y0, x0 = ii * block_size, jj * block_size
                rel_img[y0:y0+block_size, x0:x0+block_size] = rgb
            crgb = tuple(int(x) for x in unit.condition[i, j])
            if crgb != (0, 0, 0):
                y0, x0 = ii * block_size, jj * block_size
                cond_img[y0:y0+block_size, x0:x0+block_size] = crgb

    rel_path = f"{tmpdir}/relation.png"
    cond_path = f"{tmpdir}/condition.png"
    Image.fromarray(rel_img).save(rel_path)
    Image.fromarray(cond_img).save(cond_path)
    paths.extend([rel_path, cond_path])

    # 3. Tape visualization — three horizontal strips stacked
    # Collect tape entries that correspond to our entity subset
    tape_entries = []
    tape_len = min(len(unit.tape_r), len(unit.tape_c), len(unit.tape_e))
    for k in range(tape_len):
        r_px = unit.tape_r[k]
        c_px = unit.tape_c[k]
        e_px = unit.tape_e[k]
        if r_px == BLACK and c_px == BLACK and e_px == BLACK:
            tape_entries.append(((0,0,0), (0,0,0), (0,0,0)))  # stop marker
            continue
        i, j = e_px[0], e_px[1]
        if i in entity_indices or j in entity_indices:
            tape_entries.append((tuple(r_px), tuple(c_px), tuple(e_px)))

    if tape_entries:
        tape_w = len(tape_entries)
        tape_h = 3 * block_size  # 3 tapes stacked
        tape_img = np.zeros((tape_h, tape_w * block_size, 3), dtype=np.uint8)
        for k, (r, c, e) in enumerate(tape_entries):
            x0 = k * block_size
            tape_img[0:block_size, x0:x0+block_size] = r               # Tape R
            tape_img[block_size:2*block_size, x0:x0+block_size] = c     # Tape C
            tape_img[2*block_size:3*block_size, x0:x0+block_size] = e   # Tape E
        tape_path = f"{tmpdir}/tapes.png"
        Image.fromarray(tape_img).save(tape_path)
        paths.append(tape_path)

    # Build compact legend — minimal tokens
    # Entity map: "0:user,1:acme,2:alice"
    ent_map = ",".join(f"{ii}:{unit.idx_to_entity.get(i, f'e{i}')}" for ii, i in enumerate(idx_list))

    # Color map: only used colors, compact format
    # "R=works_at,G=lives_in,B=graduated_with"
    used_rels = {}
    used_conds = {}
    for i in idx_list:
        for j in idx_list:
            rgb = tuple(int(x) for x in unit.relation[i, j])
            if rgb != (0, 0, 0):
                used_rels[rgb] = unit.color_to_relation.get(rgb, "?")
            crgb = tuple(int(x) for x in unit.condition[i, j])
            if crgb != (0, 0, 0):
                used_conds[crgb] = unit.color_to_condition.get(crgb, "?")

    rel_map = ",".join(f"{r},{g},{b}={name}" for (r,g,b), name in sorted(used_rels.items()))
    cond_map = ",".join(f"{r},{g},{b}={name}" for (r,g,b), name in sorted(used_conds.items())) if used_conds else ""

    legend = f"Entities:{ent_map}\nRelations:{rel_map}"
    if cond_map:
        legend += f"\nConditions:{cond_map}"

    return paths, legend


def triples_to_text(
    triples: list[DecodedTriple],
    fmt: str = "quadruple",
) -> str:
    """Convert decoded triples to a readable text summary.

    Args:
        triples: Decoded triples to format.
        fmt: Output format:
            "quadruple" — "Alice works_at Acme [since 2022]"
            "condition_triple" — splits condition into separate triple:
                "Alice works_at Acme" + "Alice works_at_condition since_2022"
            "legacy" — "alice --[works_at]--> acme (context: since_2022)"
    """
    if not triples:
        return "No relevant knowledge found."

    lines = []
    for t in triples:
        if fmt == "quadruple":
            cond = f" [{t.condition}]" if t.condition else ""
            lines.append(f"- {t.subject} {t.relation} {t.object}{cond}")
        elif fmt == "condition_triple":
            lines.append(f"- {t.subject} {t.relation} {t.object}")
            if t.condition:
                lines.append(f"- {t.subject} {t.relation}_condition {t.condition}")
        else:  # legacy
            cond = f" (context: {t.condition})" if t.condition else ""
            lines.append(f"- {t.subject} --[{t.relation}]--> {t.object}{cond}")
    return "\n".join(lines)
