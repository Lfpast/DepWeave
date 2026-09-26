"""Entity and relation normalization with alias tracking for PixelMem V3.

The Canonicalizer scans existing shards to learn entity names, then
provides fast normalization of entity names and relation labels.  It
maintains an alias map so that surface-form variations ("Alice Smith",
"alice", "alice_smith") all resolve to a single canonical name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from pixelmem.memory import BLACK, PixelMemUnit
from pixelmem.shard_manager import ShardManager
from pixelmem.triple_extractor import Triple


class Canonicalizer:
    """Normalize entity names and relation labels across shards.

    On construction the canonicalizer scans every shard in *mgr* to
    learn existing entity names and build initial alias / relation maps.

    Attributes
    ----------
    _alias_map : dict[str, str]
        Lowercased surface form -> canonical entity name.
    _relation_map : dict[str, str]
        Variant relation label -> canonical relation.
    _entity_variants : dict[str, set[str]]
        Canonical entity name -> set of known surface variants.
    """

    # ── Relation synonym table ──────────────────────────────────────

    RELATION_SYNONYMS: dict[str, str] = {
        "employed_by": "works_at",
        "resides_in": "lives_in",
        "born_in_city": "born_in",
        "partner_with": "partner_of",
        "collaborator": "collaborates_with",
        "studied": "studied_at",
        "prefers": "prefers",
        "likes": "likes",
        "contains_func": "contains_function",
        "has_class": "contains_class",
        "import": "imports",
        "call": "calls",
        "extend": "extends",
        "depend_on": "depends_on",
        "configure": "configures",
        "test": "tests",
    }

    # ── Construction ────────────────────────────────────────────────

    def __init__(self, mgr: ShardManager) -> None:
        """Initialise the canonicalizer from an existing ShardManager.

        Scans all shards to populate the alias map, relation map, and
        entity variant index.

        Parameters
        ----------
        mgr : ShardManager
            The shard manager whose shards should be indexed.
        """
        self._mgr = mgr
        self._alias_map: dict[str, str] = {}
        self._relation_map: dict[str, str] = dict(self.RELATION_SYNONYMS)
        self._entity_variants: dict[str, set[str]] = {}
        self._rebuild_maps()

    # ── Public API: single-item canonicalization ────────────────────

    def canonicalize_entity(self, name: str) -> str:
        """Normalize an entity name to its canonical form.

        Steps:
          1. Strip whitespace and lowercase.
          2. Replace interior whitespace with underscores
             (e.g. ``"alice smith"`` -> ``"alice_smith"``).
          3. Look up the alias map; return the canonical if found.
          4. Otherwise return the cleaned name as-is.

        Parameters
        ----------
        name : str
            Raw entity name.

        Returns
        -------
        str
            Canonical entity name.
        """
        cleaned = self._clean_name(name)
        if cleaned in self._alias_map:
            return self._alias_map[cleaned]
        return cleaned

    def canonicalize_relation(self, relation: str) -> str:
        """Normalize a relation label.

        Lowercases, strips whitespace, replaces spaces with underscores,
        and resolves known synonyms.

        Parameters
        ----------
        relation : str
            Raw relation label.

        Returns
        -------
        str
            Canonical relation label.
        """
        cleaned = relation.strip().lower().replace(" ", "_")
        return self._relation_map.get(cleaned, cleaned)

    def canonicalize_triple(self, triple: Triple) -> Triple:
        """Normalize all fields of a Triple.

        Parameters
        ----------
        triple : Triple
            The triple to normalize.

        Returns
        -------
        Triple
            A new Triple with canonicalized subject, relation, and object.
            The condition field is preserved unchanged.
        """
        return Triple(
            subject=self.canonicalize_entity(triple.subject),
            relation=self.canonicalize_relation(triple.relation),
            object=self.canonicalize_entity(triple.object),
            condition=triple.condition,
        )

    def canonicalize_batch(self, triples: list[Triple]) -> list[Triple]:
        """Normalize a list of triples.

        Parameters
        ----------
        triples : list[Triple]
            Triples to normalize.

        Returns
        -------
        list[Triple]
            New list of canonicalized triples.
        """
        return [self.canonicalize_triple(t) for t in triples]

    # ── Public API: alias management ────────────────────────────────

    def register_alias(self, alias: str, canonical: str) -> None:
        """Register *alias* as an alternative surface form for *canonical*.

        Both names are cleaned (lowercased, underscored) before storage.

        Parameters
        ----------
        alias : str
            The new alias to register.
        canonical : str
            The canonical entity name the alias resolves to.
        """
        clean_alias = self._clean_name(alias)
        clean_canonical = self._clean_name(canonical)
        self._alias_map[clean_alias] = clean_canonical
        self._entity_variants.setdefault(clean_canonical, set()).add(clean_alias)

    def resolve_aliases(self, names: list[str]) -> list[str]:
        """Batch-resolve a list of entity names through the alias map.

        Parameters
        ----------
        names : list[str]
            Raw entity names.

        Returns
        -------
        list[str]
            Canonical names in the same order.
        """
        return [self.canonicalize_entity(n) for n in names]

    # ── Persistence ─────────────────────────────────────────────────

    def save_aliases(self, path: str | Path) -> None:
        """Save the alias map and entity variants to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        path = Path(path)
        data = {
            "alias_map": self._alias_map,
            "relation_map": {
                k: v
                for k, v in self._relation_map.items()
                if k not in self.RELATION_SYNONYMS or self._relation_map[k] != self.RELATION_SYNONYMS[k]
            },
            "entity_variants": {
                k: sorted(v) for k, v in self._entity_variants.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def load_aliases(self, path: str | Path) -> None:
        """Load alias map and entity variants from a JSON file.

        Merges loaded data with any existing entries (loaded data wins on
        conflict).

        Parameters
        ----------
        path : str or Path
            Source file path.
        """
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        # Merge alias map
        loaded_aliases: dict[str, str] = raw.get("alias_map", {})
        self._alias_map.update(loaded_aliases)

        # Merge custom relation synonyms
        loaded_relations: dict[str, str] = raw.get("relation_map", {})
        self._relation_map.update(loaded_relations)

        # Merge entity variants
        loaded_variants: dict[str, list[str]] = raw.get("entity_variants", {})
        for canonical, variants in loaded_variants.items():
            existing = self._entity_variants.setdefault(canonical, set())
            existing.update(variants)

    # ── Internal helpers ────────────────────────────────────────────

    def _rebuild_maps(self) -> None:
        """Scan all shards to learn existing entity names and relations.

        For each shard, every entity name is cleaned and indexed.  If two
        different surface forms collapse to the same cleaned key the first
        one encountered becomes canonical and the rest become aliases.

        Relations are indexed from each shard's ``relation_color_map`` and
        fed through the synonym table.  Additionally, any triple whose
        relation is ``"alias"`` or ``"is"`` is used to seed the alias map.
        """
        for shard in self._mgr.shards:
            # Index entities
            for raw_name in shard.entity_to_idx:
                cleaned = self._clean_name(raw_name)
                if cleaned not in self._alias_map:
                    # First occurrence becomes the canonical form
                    self._alias_map[cleaned] = cleaned
                    self._entity_variants.setdefault(cleaned, set()).add(cleaned)
                else:
                    canonical = self._alias_map[cleaned]
                    self._entity_variants.setdefault(canonical, set()).add(cleaned)

            # Index relations from the colour map
            for rel_name in shard.relation_color_map:
                canon_rel = self.canonicalize_relation(rel_name)
                cleaned_rel = rel_name.strip().lower().replace(" ", "_")
                if cleaned_rel != canon_rel:
                    self._relation_map[cleaned_rel] = canon_rel

            # Learn aliases from explicit "alias" / "is" relations stored
            # in the shard's relation matrix.
            self._learn_aliases_from_shard(shard)

    def _learn_aliases_from_shard(self, shard: PixelMemUnit) -> None:
        """Extract alias relationships encoded in a shard.

        Looks for relations named ``"alias"`` or ``"is"`` in the shard's
        colour map and, for every pair of entities connected by that
        relation, registers the object as an alias of the subject.
        """
        alias_colors: set[tuple[int, int, int]] = set()
        for rel_name, color in shard.relation_color_map.items():
            low = rel_name.strip().lower()
            if low in ("alias", "is", "same_as", "also_known_as"):
                alias_colors.add(color)

        if not alias_colors:
            return

        n = shard.n
        for i in range(n):
            for j in range(n):
                pixel = tuple(shard.relation[i, j])
                if pixel == BLACK:
                    continue
                if pixel in alias_colors:
                    subj = shard.idx_to_entity.get(i)
                    obj = shard.idx_to_entity.get(j)
                    if subj and obj:
                        canonical = self._clean_name(subj)
                        alias = self._clean_name(obj)
                        self._alias_map[alias] = canonical
                        self._entity_variants.setdefault(canonical, set()).add(alias)

    @staticmethod
    def _clean_name(name: str) -> str:
        """Lowercase, strip, and collapse whitespace to underscores."""
        return re.sub(r"\s+", "_", name.strip().lower())

    # ── Stats ───────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Summary statistics for the canonicalizer state.

        Returns
        -------
        dict
            Keys: ``entities`` (number of canonical entities),
            ``aliases`` (total alias entries), ``relation_synonyms``
            (number of relation mappings), ``variants`` (total variant
            count across all entities).
        """
        total_variants = sum(len(v) for v in self._entity_variants.values())
        return {
            "entities": len(self._entity_variants),
            "aliases": len(self._alias_map),
            "relation_synonyms": len(self._relation_map),
            "variants": total_variants,
        }
