"""PixelMem Encoding Pipeline (Algorithm 1 from the paper).

Converts raw text into pixel-encoded knowledge graph triples.
"""

from __future__ import annotations

from typing import Optional

from pixelmem.memory import PixelMemUnit, BLACK
from pixelmem.triple_extractor import extract_triples, Triple


def encode_text(
    text: str,
    unit: PixelMemUnit,
    triples: Optional[list[Triple]] = None,
) -> list[Triple]:
    """Encode text into a PixelMem unit.

    Implements Algorithm 1: for each triple (subject, relation, object, condition):
      1. Resolve entity indices for subject and object
      2. Resolve RGB colors for relation and condition
      3. Write to relation and condition matrices at (i, j)
      4. Append to all three tapes
      5. Append chunk delimiter (black on all tapes)

    Args:
        text: Raw text to extract triples from.
        unit: The PixelMemUnit to write into.
        triples: Pre-extracted triples. If None, uses the triple extractor.

    Returns:
        The list of triples that were encoded.
    """
    if triples is None:
        triples = extract_triples(text)

    if not triples:
        return []

    for triple in triples:
        i = unit.resolve_entity(triple.subject)
        j = unit.resolve_entity(triple.object)
        rgb_r = unit.resolve_relation_color(triple.relation)
        rgb_c = unit.resolve_condition_color(triple.condition)

        # Write to matrices
        unit.relation[i, j] = rgb_r
        unit.condition[i, j] = rgb_c

        # Append to tapes
        unit.tape_r.append(rgb_r)
        unit.tape_c.append(rgb_c)
        unit.tape_e.append((i, j, 0))

    # Chunk delimiter — black on all three tapes
    unit.tape_r.append(BLACK)
    unit.tape_c.append(BLACK)
    unit.tape_e.append(BLACK)

    # Update summary with entity names from this chunk
    entities_in_chunk = set()
    for t in triples:
        entities_in_chunk.add(t.subject.strip().lower())
        entities_in_chunk.add(t.object.strip().lower())
    if unit.summary:
        unit.summary += "; "
    unit.summary += ", ".join(sorted(entities_in_chunk))

    return triples


def encode_triples(
    triples: list[Triple],
    unit: PixelMemUnit,
) -> None:
    """Encode pre-extracted triples directly (no LLM extraction step)."""
    encode_text("", unit, triples=triples)
