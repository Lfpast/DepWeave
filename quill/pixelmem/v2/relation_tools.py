"""Relation-Specific Mini-Tools — deterministic helpers for targeted queries.

Small pure functions that operate directly on the pixel matrix.
The LLM calls these instead of doing arithmetic or filtering itself.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from pixelmem.memory import PixelMemUnit, BLACK
from pixelmem.shard_manager import ShardManager


@dataclass
class FactEntry:
    """Compact structured fact."""
    subject: str
    relation: str
    object: str
    condition: str = ""
    shard_idx: int = -1

    def to_dict(self) -> dict:
        d = {"s": self.subject, "r": self.relation, "o": self.object}
        if self.condition:
            d["c"] = self.condition
        return d

    def to_compact(self) -> str:
        c = f" [{self.condition}]" if self.condition else ""
        return f"{self.subject} {self.relation} {self.object}{c}"


def _scan_entity_facts(mgr: ShardManager, entity: str) -> list[FactEntry]:
    """Scan all facts about an entity — chunk-based first, flat fallback.

    Tries chunk-based scan (preserves context grouping). If that finds
    fewer facts than the flat matrix scan, uses flat scan instead.
    This handles workflow entities where chunk tapes may not index all triples.
    """
    from pixelmem.decoder import scan_entity_chunks

    # Try chunk-based scan
    chunk_facts = []
    seen_chunks = set()
    for si, shard in enumerate(mgr.shards):
        chunks = scan_entity_chunks(entity, shard)
        for chunk in chunks:
            chunk_key = tuple((t.subject, t.relation, t.object) for t in chunk)
            if chunk_key in seen_chunks:
                continue
            seen_chunks.add(chunk_key)
            for t in chunk:
                chunk_facts.append(FactEntry(t.subject, t.relation, t.object, t.condition, si))

    # Also do flat matrix scan
    flat_facts = _scan_entity_facts_flat(mgr, entity)

    # Use whichever found more (flat is more complete for workflow entities)
    return flat_facts if len(flat_facts) > len(chunk_facts) else chunk_facts


def _scan_entity_facts_flat(mgr: ShardManager, entity: str) -> list[FactEntry]:
    """Flat scan (no chunks) — kept for comparison."""
    canonical = entity.strip().lower()
    facts = []
    for si, shard in enumerate(mgr.shards):
        if canonical not in shard.entity_to_idx:
            continue
        idx = shard.entity_to_idx[canonical]
        # Row scan (entity as subject)
        for j in range(shard.n):
            rgb = tuple(int(x) for x in shard.relation[idx, j])
            if rgb == BLACK:
                continue
            obj = shard.idx_to_entity.get(j, f"e{j}")
            rel = shard.color_to_relation.get(rgb, "?")
            crgb = tuple(int(x) for x in shard.condition[idx, j])
            cond = shard.color_to_condition.get(crgb, "")
            facts.append(FactEntry(canonical, rel, obj, cond, si))
        # Column scan (entity as object)
        for i in range(shard.n):
            if i == idx:
                continue
            rgb = tuple(int(x) for x in shard.relation[i, idx])
            if rgb == BLACK:
                continue
            subj = shard.idx_to_entity.get(i, f"e{i}")
            rel = shard.color_to_relation.get(rgb, "?")
            crgb = tuple(int(x) for x in shard.condition[i, idx])
            cond = shard.color_to_condition.get(crgb, "")
            facts.append(FactEntry(subj, rel, canonical, cond, si))
    return facts


# ── Tool 1: resolve_alias ───────────────────────────────────────

def resolve_alias(mgr: ShardManager, surface_form: str) -> list[str]:
    """Find canonical entity names matching a surface form.

    Matches by: exact, substring, word overlap, known alias relations.
    """
    canonical = surface_form.strip().lower()
    all_entities = set()
    for shard in mgr.shards:
        all_entities.update(shard.entity_to_idx.keys())

    matches = []
    # Exact match
    if canonical in all_entities:
        matches.append(canonical)

    # Substring match (both directions)
    for ent in all_entities:
        if ent == canonical:
            continue
        if canonical in ent or ent in canonical:
            matches.append(ent)
        # Word overlap
        elif len(canonical) > 3:
            ent_words = set(ent.replace("_", " ").split())
            query_words = set(canonical.replace("_", " ").split())
            if query_words & ent_words:
                matches.append(ent)

    # Check alias/is relations
    if canonical in all_entities:
        facts = _scan_entity_facts(mgr, canonical)
        for f in facts:
            if f.relation in ("is", "alias", "also_known_as", "same_as"):
                if f.subject == canonical:
                    matches.append(f.object)
                else:
                    matches.append(f.subject)

    return sorted(set(matches))


# ── Tool 2: find_rel ────────────────────────────────────────────

def find_rel(
    mgr: ShardManager,
    entity: str,
    relation: Optional[str] = None,
    limit: int = 20,
    direct_only: bool = False,
) -> list[FactEntry]:
    """Find all facts for an entity, optionally filtered by relation type.

    Args:
        direct_only: If True, only return facts where the entity is
            directly the subject or object (not just in the same chunk).
    """
    facts = _scan_entity_facts(mgr, entity)
    canonical = entity.strip().lower()

    if direct_only:
        facts = [f for f in facts if canonical in f.subject or canonical in f.object]

    if relation:
        rel_lower = relation.strip().lower()
        facts = [f for f in facts if rel_lower in f.relation or f.relation in rel_lower]

    # Prioritize: facts directly mentioning the entity come first
    direct = [f for f in facts if canonical in f.subject or canonical in f.object]
    indirect = [f for f in facts if f not in direct]
    prioritized = direct + indirect

    return prioritized[:limit]


# ── Tool 3: find_between ────────────────────────────────────────

def find_between(
    mgr: ShardManager,
    entity_a: str,
    entity_b: str,
) -> list[FactEntry]:
    """Find all direct relations between two entities."""
    a = entity_a.strip().lower()
    b = entity_b.strip().lower()
    results = []

    for si, shard in enumerate(mgr.shards):
        if a not in shard.entity_to_idx or b not in shard.entity_to_idx:
            continue
        idx_a = shard.entity_to_idx[a]
        idx_b = shard.entity_to_idx[b]

        # a -> b
        rgb = tuple(int(x) for x in shard.relation[idx_a, idx_b])
        if rgb != BLACK:
            rel = shard.color_to_relation.get(rgb, "?")
            crgb = tuple(int(x) for x in shard.condition[idx_a, idx_b])
            cond = shard.color_to_condition.get(crgb, "")
            results.append(FactEntry(a, rel, b, cond, si))

        # b -> a
        rgb = tuple(int(x) for x in shard.relation[idx_b, idx_a])
        if rgb != BLACK:
            rel = shard.color_to_relation.get(rgb, "?")
            crgb = tuple(int(x) for x in shard.condition[idx_b, idx_a])
            cond = shard.color_to_condition.get(crgb, "")
            results.append(FactEntry(b, rel, a, cond, si))

    return results


# ── Tool 4: find_recent ─────────────────────────────────────────

_DATE_PATTERN = re.compile(r'(\d{4}-\d{2}-\d{2})')


def find_recent(
    mgr: ShardManager,
    entity: str,
    relation: Optional[str] = None,
) -> list[FactEntry]:
    """Find most recent facts for an entity (sorted by date in condition)."""
    facts = find_rel(mgr, entity, relation, limit=100)

    def _extract_date(fact: FactEntry) -> str:
        m = _DATE_PATTERN.search(fact.condition)
        return m.group(1) if m else "0000-00-00"

    facts.sort(key=_extract_date, reverse=True)
    return facts


# ── Tool 5: filter_chunks ───────────────────────────────────────

def filter_chunks(
    mgr: ShardManager,
    entity: str,
    relation: Optional[str] = None,
    date_range: Optional[tuple[str, str]] = None,
) -> list[list[FactEntry]]:
    """Find chunks containing an entity, optionally filtered."""
    from pixelmem.decoder import scan_entity_chunks

    chunks = []
    for shard in mgr.shards:
        raw_chunks = scan_entity_chunks(entity, shard)
        for chunk in raw_chunks:
            facts = []
            for t in chunk:
                f = FactEntry(t.subject, t.relation, t.object, t.condition or "")
                if relation and relation.lower() not in f.relation:
                    continue
                if date_range:
                    m = _DATE_PATTERN.search(f.condition)
                    if m:
                        d = m.group(1)
                        if d < date_range[0] or d > date_range[1]:
                            continue
                facts.append(f)
            if facts:
                chunks.append(facts)
    return chunks


# ── Tool 6: date_diff ───────────────────────────────────────────

def date_diff(date_a: str, date_b: str) -> dict:
    """Compute the difference between two dates.

    Accepts: "2026-04-03", "2026-03-20", etc.
    Returns: {"days": int, "weeks": float, "months": float, "description": str}
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


# ── Tool 7: sum_values ──────────────────────────────────────────

_NUMBER_PATTERN = re.compile(r'[\d,]+\.?\d*')


def sum_values(values: list[str]) -> dict:
    """Sum numeric values extracted from strings.

    Handles: "$400,000", "3.5 weeks", "45 minutes", "22 movies"
    """
    total = 0.0
    parsed = []
    for v in values:
        nums = _NUMBER_PATTERN.findall(v.replace(",", ""))
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


# ── Tool 8: count_relations ─────────────────────────────────────

def count_relations(
    mgr: ShardManager,
    entity: str,
    relation: Optional[str] = None,
    direction: str = "both",
) -> dict:
    """Count edges for an entity, optionally filtered by relation and direction."""
    facts = _scan_entity_facts(mgr, entity)
    canonical = entity.strip().lower()

    if relation:
        rel_lower = relation.strip().lower()
        facts = [f for f in facts if rel_lower in f.relation]

    if direction == "out":
        facts = [f for f in facts if f.subject == canonical]
    elif direction == "in":
        facts = [f for f in facts if f.object == canonical]

    return {
        "entity": canonical,
        "count": len(facts),
        "direction": direction,
        "relation_filter": relation,
    }
