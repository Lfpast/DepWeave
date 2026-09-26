"""Sharded PixelMem Architecture (Section 5 of the paper).

Manages multiple small, dense PixelMem units with a summary index
for query routing and selective loading.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from pixelmem.memory import PixelMemUnit
from pixelmem.encoder import encode_text
from pixelmem.decoder import decode_query, DecodedTriple, triples_to_text
from pixelmem.triple_extractor import Triple


class ShardManager:
    """Manages a collection of sharded PixelMem units.

    Each shard is a self-contained PixelMemUnit with:
      - Its own entity index and color space
      - A target size N (default 64)
      - A density target (default 75%)

    The summary index routes queries to relevant shards based on
    lightweight text matching of entity names.
    """

    def __init__(
        self,
        storage_dir: str | Path,
        shard_size: int = 64,
        density_target: float = 0.75,
    ):
        self.storage_dir = Path(storage_dir)
        self.shard_size = shard_size
        self.density_target = density_target
        self.shards: list[PixelMemUnit] = []
        self._shard_dirs: list[Path] = []
        # Global entity -> shard index for O(1) routing
        self._entity_shard_index: dict[str, list[int]] = {}

    def _rebuild_entity_index(self) -> None:
        """Rebuild the global entity-to-shard index from all shards."""
        self._entity_shard_index.clear()
        for shard_idx, shard in enumerate(self.shards):
            for entity_name in shard.entity_to_idx:
                self._entity_shard_index.setdefault(entity_name, [])
                if shard_idx not in self._entity_shard_index[entity_name]:
                    self._entity_shard_index[entity_name].append(shard_idx)

    def _index_shard_entities(self, shard_idx: int) -> None:
        """Incrementally index entities from a single shard."""
        shard = self.shards[shard_idx]
        for entity_name in shard.entity_to_idx:
            self._entity_shard_index.setdefault(entity_name, [])
            if shard_idx not in self._entity_shard_index[entity_name]:
                self._entity_shard_index[entity_name].append(shard_idx)

    # ── Shard creation and selection ────────────────────────────────

    def _create_shard(self) -> PixelMemUnit:
        """Create a new empty shard."""
        idx = len(self.shards)
        name = f"shard_{idx:04d}"
        unit = PixelMemUnit(n=self.shard_size, name=name)
        self.shards.append(unit)
        shard_dir = self.storage_dir / name
        self._shard_dirs.append(shard_dir)
        return unit

    def _find_best_shard(self, entities: list[str]) -> PixelMemUnit:
        """Find the best shard for a set of entities.

        Strategy:
          1. Prefer shards that already contain some of the entities
             (maximizes density via entity reuse).
          2. Among those, prefer the least-full shard that still has capacity.
          3. If no shard has room or matches, create a new one.
        """
        if not self.shards:
            return self._create_shard()

        best_shard = None
        best_score = -1

        for shard in self.shards:
            # Check capacity
            new_entities = sum(
                1 for e in entities if not shard.has_entity(e)
            )
            if shard.entity_count() + new_entities >= shard.n:
                continue  # Would overflow

            # Check density target
            if shard.density() >= self.density_target:
                continue  # Already full enough, don't add noise

            # Score: number of entities already present (reuse is good)
            overlap = sum(1 for e in entities if shard.has_entity(e))
            # Tie-break: prefer shard with more existing data (denser)
            score = overlap * 1000 + shard.entity_count()

            if score > best_score:
                best_score = score
                best_shard = shard

        if best_shard is None:
            return self._create_shard()

        return best_shard

    # ── Encoding ────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        triples: Optional[list[Triple]] = None,
    ) -> tuple[PixelMemUnit, list[Triple]]:
        """Encode text into the best-matching shard.

        Returns the shard used and the triples encoded.
        """
        from pixelmem.triple_extractor import extract_triples

        if triples is None:
            triples = extract_triples(text)

        if not triples:
            if not self.shards:
                self._create_shard()
            return self.shards[-1], []

        # Collect entities mentioned in these triples
        entities = set()
        for t in triples:
            entities.add(t.subject)
            entities.add(t.object)

        shard = self._find_best_shard(list(entities))
        encoded = encode_text(text, shard, triples=triples)
        # Update entity index
        shard_idx = self.shards.index(shard)
        self._index_shard_entities(shard_idx)
        return shard, encoded

    # ── Topic-aware encoding ──────────────────────────────────────

    # Relation type -> topic category mapping
    _TOPIC_MAP: dict[str, str] = {
        "works_at": "employment",
        "ceo_of": "employment",
        "role": "employment",
        "manages": "management",
        "reports_to": "management",
        "lives_in": "locations",
        "born_in": "locations",
        "located_in": "locations",
        "has_skill": "skills",
        "knows": "skills",
        "founded_by": "organizations",
        "partner_of": "organizations",
    }

    def _get_topic(self, relation: str) -> str:
        """Map a relation name to a topic category."""
        rel_lower = relation.strip().lower()
        return self._TOPIC_MAP.get(rel_lower, "general")

    def _find_topic_shard(self, topic: str, entities: list[str]) -> PixelMemUnit:
        """Find or create a shard for a specific topic."""
        # Look for existing shard with matching topic
        for shard in self.shards:
            if not hasattr(shard, '_topic'):
                continue
            if shard._topic != topic:
                continue
            # Check capacity
            new_ents = sum(1 for e in entities if not shard.has_entity(e))
            if shard.entity_count() + new_ents < shard.n:
                return shard

        # Create new topic shard
        shard = self._create_shard()
        shard._topic = topic
        return shard

    def encode_by_topic(
        self,
        text: str,
        triples: Optional[list[Triple]] = None,
    ) -> list[tuple[PixelMemUnit, list[Triple]]]:
        """Encode triples into topic-specific shards.

        Groups triples by relation type (employment, locations, management,
        etc.) and routes each group to a dedicated shard. Each shard gets
        a rich summary describing what it stores.

        Returns list of (shard, triples_encoded) pairs.
        """
        from pixelmem.triple_extractor import extract_triples

        if triples is None:
            triples = extract_triples(text)

        if not triples:
            return []

        # Group triples by topic
        topic_groups: dict[str, list[Triple]] = {}
        for t in triples:
            topic = self._get_topic(t.relation)
            topic_groups.setdefault(topic, []).append(t)

        results = []
        for topic, group_triples in topic_groups.items():
            entities = set()
            for t in group_triples:
                entities.add(t.subject)
                entities.add(t.object)

            shard = self._find_topic_shard(topic, list(entities))
            encoded = encode_text(text, shard, triples=group_triples)

            # Build rich summary
            relations = sorted(set(t.relation for t in group_triples))
            subjects = sorted(set(t.subject for t in group_triples))[:10]
            shard.summary = (
                f"Topic: {topic}. "
                f"Relations: {', '.join(relations)}. "
                f"Entities: {', '.join(subjects)}"
                + (f"... (+{len(set(t.subject for t in group_triples)) - 10} more)"
                   if len(set(t.subject for t in group_triples)) > 10 else "")
            )
            shard._topic = topic

            shard_idx = self.shards.index(shard)
            self._index_shard_entities(shard_idx)
            results.append((shard, encoded))

        return results

    # ── Query routing ───────────────────────────────────────────────

    # Query keyword -> topic mapping for routing
    _QUERY_TOPIC_HINTS: dict[str, str] = {
        "work": "employment", "works": "employment", "job": "employment",
        "employ": "employment", "company": "employment", "ceo": "employment",
        "manage": "management", "manages": "management", "manager": "management",
        "report": "management", "reports": "management", "boss": "management",
        "live": "locations", "lives": "locations", "born": "locations",
        "city": "locations", "location": "locations", "located": "locations",
        "where": "locations",
        "skill": "skills", "knows": "skills", "expertise": "skills",
        "found": "organizations", "founded": "organizations",
        "partner": "organizations",
    }

    # First-person pronouns that map to "user" entity
    _PRONOUN_MAP = {"i", "my", "me", "mine", "myself", "we", "our"}

    def match_shards(self, query: str) -> list[PixelMemUnit]:
        """Route a query to relevant shards — fuzzy semantic matching.

        Like Claude Code reading MEMORY.md: matches related concepts,
        not just exact entity names. "my commute" matches "user commute_duration".

        Four-phase routing:
          1. Entity index: exact + substring matching
          2. Pronoun resolution: I/my/me → "user"
          3. Relation/keyword matching: query words match relation names and
             entity names via substring overlap
          4. All shards fallback if nothing matched (return all, scored)

        Returns shards sorted by relevance.
        """
        query_lower = query.lower()
        query_terms = set(re.findall(r'\w+', query_lower))

        shard_scores: dict[int, float] = {}

        # Phase 0: Pronoun → "user" resolution
        if query_terms & self._PRONOUN_MAP:
            query_terms.add("user")

        # Phase 1: Entity index — exact match + substring match
        for term in query_terms:
            if len(term) < 3:
                continue
            for entity_name, shard_indices in self._entity_shard_index.items():
                # Exact match
                if term == entity_name:
                    for si in shard_indices:
                        shard_scores[si] = shard_scores.get(si, 0) + 10
                # Substring: "commute" matches "commute_duration", "user" matches "user"
                elif term in entity_name or entity_name in term:
                    for si in shard_indices:
                        shard_scores[si] = shard_scores.get(si, 0) + 5
                # Word overlap: "degree" matches "graduated_with" if both in same shard
                # (check relation names too)

        # Phase 2: Relation name matching — "commute" matches shard with "commutes" relation
        for shard_idx, shard in enumerate(self.shards):
            for rel_name in shard.relation_color_map:
                rel_terms = set(re.findall(r'\w+', rel_name.lower()))
                overlap = query_terms & rel_terms
                if overlap:
                    shard_scores[shard_idx] = shard_scores.get(shard_idx, 0) + 3 * len(overlap)

            # Also check entity names for partial word matches
            for ent_name in shard.entity_to_idx:
                ent_terms = set(re.findall(r'\w+', ent_name.lower()))
                overlap = query_terms & ent_terms
                if overlap:
                    shard_scores[shard_idx] = shard_scores.get(shard_idx, 0) + 2 * len(overlap)

        # Phase 3: Topic hint matching
        query_topics = set()
        for term in query_terms:
            if term in self._QUERY_TOPIC_HINTS:
                query_topics.add(self._QUERY_TOPIC_HINTS[term])
        if query_topics:
            for shard_idx, shard in enumerate(self.shards):
                shard_topic = getattr(shard, '_topic', '')
                if shard_topic in query_topics:
                    shard_scores[shard_idx] = shard_scores.get(shard_idx, 0) + 3

        # Phase 4: If still nothing, score ALL shards by summary overlap
        if not shard_scores:
            for shard_idx, shard in enumerate(self.shards):
                summary_terms = set(re.findall(r'\w+', shard.summary.lower()))
                hits = len(query_terms & summary_terms)
                if hits > 0:
                    shard_scores[shard_idx] = hits

        scored = sorted(shard_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.shards[idx] for idx, _ in scored]

    def query_agent(
        self,
        query: str,
        model: str = "haiku",
        max_hops: int = 1,
    ) -> list[DecodedTriple]:
        """Agent-guided retrieval — like Claude Code reading MEMORY.md.

        Instead of fuzzy string matching, shows the entity/relation index
        to a CLI agent and lets it decide which entities are relevant.

        Steps:
          1. Build index string: all entity names + relation names
          2. Ask CLI agent: "which entities/relations are relevant to this query?"
          3. Agent returns relevant entity names
          4. Decode only those entities' triples

        This is how Claude Code works: it reads MEMORY.md (the index),
        decides which memory files are relevant, then reads those files.
        """
        import subprocess, os, sys
        from pathlib import Path

        # Build the index (like MEMORY.md) — show per-shard summaries + entities + relations
        lines = []
        for i, shard in enumerate(self.shards):
            ents = sorted(shard.entity_to_idx.keys())
            rels = sorted(shard.relation_color_map.keys())
            summary = shard.summary or "no summary"
            topic = getattr(shard, '_topic', '')
            header = f"Shard {shard.name}"
            if topic:
                header += f" [{topic}]"
            lines.append(
                f"{header}: {summary}\n"
                f"  Entities: {', '.join(ents[:30])}\n"
                f"  Relations: {', '.join(rels[:20])}"
            )
        index_text = "\n".join(lines)

        prompt = (
            f"Given this question: \"{query}\"\n\n"
            f"Here is the memory index:\n{index_text}\n\n"
            f"Which entities from the index are most relevant to answering the question? "
            f"Return ONLY a comma-separated list of entity names, nothing else. "
            f"Include entities that might contain the answer, even if the connection is indirect. "
            f"Map pronouns (I, my, me) to 'user' if present in the index."
        )

        env = os.environ.copy()
        if sys.platform == "win32" and "CLAUDE_CODE_GIT_BASH_PATH" not in env:
            for c in [r"D:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\bin\bash.exe"]:
                if os.path.exists(c):
                    env["CLAUDE_CODE_GIT_BASH_PATH"] = c
                    break

        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--max-turns", "1", "--model", model],
                capture_output=True, text=True, timeout=60,
                cwd=str(Path(__file__).resolve().parent.parent), env=env,
            )
            response = result.stdout.strip()
        except Exception:
            # Fallback to fuzzy matching
            return self.query(query, max_hops=max_hops)

        # Parse agent's entity selection
        selected = [e.strip().lower() for e in response.split(",") if e.strip()]
        if not selected:
            return self.query(query, max_hops=max_hops)

        # Decode using selected entities
        return self.query(query, entity_names=selected, max_hops=max_hops)

    def query(
        self,
        query: str,
        entity_names: Optional[list[str]] = None,
        max_shards: int = 3,
        max_hops: int = 1,
        include_chunks: bool = False,
    ) -> list[DecodedTriple]:
        """Query across all relevant shards with multi-hop following.

        Args:
            query: Natural language query text.
            entity_names: Specific entity names to look up. If None,
                extracts entity-like words from the query.
            max_shards: Maximum number of shards to load.
            max_hops: Number of hops to follow (1=direct, 2=neighbors, etc.)
            include_chunks: Whether to reconstruct chunk context.

        Returns:
            Combined decoded triples from all matching shards.
        """
        relevant = self.match_shards(query)[:max_shards]

        if entity_names is None:
            # Simple extraction: multi-word capitalised phrases or single words
            entity_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            if not entity_names:
                entity_names = [
                    w for w in re.findall(r'\w+', query)
                    if len(w) > 2 and w.lower() not in _STOP_WORDS
                ]

        all_results: list[DecodedTriple] = []
        for shard in relevant:
            results = decode_query(
                entity_names, shard,
                include_chunks=include_chunks,
                max_hops=max_hops,
            )
            all_results.extend(results)

        # For multi-hop: discovered entities might live in other shards
        if max_hops > 1 and all_results:
            discovered = set()
            for t in all_results:
                discovered.add(t.subject)
                discovered.add(t.object)
            # Find shards for discovered entities not yet queried
            extra_shards: set[int] = set()
            for name in discovered:
                canonical = name.strip().lower()
                for shard_idx in self._entity_shard_index.get(canonical, []):
                    if self.shards[shard_idx] not in relevant:
                        extra_shards.add(shard_idx)
            for shard_idx in list(extra_shards)[:max_shards]:
                results = decode_query(
                    list(discovered), self.shards[shard_idx],
                    include_chunks=include_chunks,
                    max_hops=1,  # Only 1 hop in cross-shard follow
                )
                all_results.extend(results)

        return all_results

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, sparse: bool = False) -> None:
        """Save all shards and the summary index to disk.

        Args:
            sparse: If True, use compact sparse format:
                    - Global entity/color registry in index.json (shared)
                    - Per-shard: only triples.bin + tapes.bin (no per-shard JSON)
                    If False, use legacy PNG format with per-shard JSON.
        """
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Build global entity/color registry (shared across shards)
        global_entities = {}  # name -> idx
        global_idx2entity = {}  # idx -> name
        global_relations = {}  # name -> [r,g,b]
        global_rel_reverse = {}  # "r,g,b" -> name
        global_conditions = {}
        global_cond_reverse = {}
        max_entity_idx = 1

        for shard in self.shards:
            global_entities.update(shard.entity_to_idx)
            global_idx2entity.update({str(k): v for k, v in shard.idx_to_entity.items()})
            global_relations.update({k: list(v) for k, v in shard.relation_color_map.items()})
            global_rel_reverse.update({
                f"{r},{g},{b}": n for (r, g, b), n in shard.color_to_relation.items()
            })
            global_conditions.update({k: list(v) for k, v in shard.condition_color_map.items()})
            global_cond_reverse.update({
                f"{r},{g},{b}": n for (r, g, b), n in shard.color_to_condition.items()
            })
            max_entity_idx = max(max_entity_idx, shard._next_entity_idx)

        index = []
        for i, shard in enumerate(self.shards):
            shard_dir = self.storage_dir / shard.name
            if sparse:
                # Only save binary data, no per-shard JSON
                shard_dir.mkdir(parents=True, exist_ok=True)
                import struct
                coords = []
                for ii in range(shard.n):
                    for jj in range(shard.n):
                        rgb = tuple(int(x) for x in shard.relation[ii, jj])
                        if rgb == (0, 0, 0):
                            continue
                        crgb = tuple(int(x) for x in shard.condition[ii, jj])
                        coords.append((ii, jj, *rgb, *crgb))
                with open(shard_dir / "triples.bin", "wb") as f:
                    for c in coords:
                        f.write(struct.pack("8B", *c))
                # Tapes
                tape_len = min(len(shard.tape_r), len(shard.tape_c), len(shard.tape_e))
                with open(shard_dir / "tapes.bin", "wb") as f:
                    for k in range(tape_len):
                        r, c, e = shard.tape_r[k], shard.tape_c[k], shard.tape_e[k]
                        f.write(struct.pack("9B", *r, *c, *e))
            else:
                shard.save(shard_dir)

            shard_entry = {
                "name": shard.name,
                "n": shard.n,
                "entity_count": shard.entity_count(),
                "triple_count": shard.triple_count(),
                "density": round(shard.density(), 4),
                "summary": shard.summary,
                "topic": getattr(shard, '_topic', ''),
            }
            if sparse:
                # Per-shard entity mapping (compact, no duplication of colors)
                shard_entry["entities"] = {
                    "e2i": shard.entity_to_idx,
                    "i2e": {str(k): v for k, v in shard.idx_to_entity.items()},
                    "ni": shard._next_entity_idx,
                }
            index.append(shard_entry)

        index_data = {
            "shard_size": self.shard_size,
            "density_target": self.density_target,
            "shard_count": len(self.shards),
            "format": "sparse" if sparse else "png",
            "shards": index,
        }

        if sparse:
            # Embed global registry in index (no per-shard duplication)
            index_data["registry"] = {
                "e2i": global_entities,
                "i2e": global_idx2entity,
                "rc": global_relations,
                "rr": global_rel_reverse,
                "cc": global_conditions,
                "cr": global_cond_reverse,
                "ni": max_entity_idx,
                "nri": max(s._next_relation_color_id for s in self.shards) if self.shards else 1,
                "nci": max(s._next_condition_color_id for s in self.shards) if self.shards else 1,
            }

        with open(self.storage_dir / "index.json", "w") as f:
            json.dump(index_data, f, separators=(",", ":") if sparse else None,
                      indent=None if sparse else 2)

        # Also save single compact binary for whole store
        if sparse:
            self._save_compact_store()

    def _save_compact_store(self) -> None:
        """Save entire store as a single compact binary file.

        One global string table, all triples packed contiguously.
        This achieves the theoretical minimum — smaller than text.
        """
        import struct
        from pixelmem.memory import BLACK

        strings: list[str] = []
        str2idx: dict[str, int] = {}

        def _intern(s: str) -> int:
            if s not in str2idx:
                str2idx[s] = len(strings)
                strings.append(s)
            return str2idx[s]

        # Collect all triples across all shards
        all_entries = []  # (subj_str_idx, rel_str_idx, obj_str_idx, cond_str_idx)
        for shard in self.shards:
            for i in range(shard.n):
                for j in range(shard.n):
                    rgb = tuple(int(x) for x in shard.relation[i, j])
                    if rgb == (0, 0, 0):
                        continue
                    subj = shard.idx_to_entity.get(i, f"entity_{i}")
                    obj = shard.idx_to_entity.get(j, f"entity_{j}")
                    rel = shard.color_to_relation.get(rgb, f"rel_{rgb}")
                    cond_rgb = tuple(int(x) for x in shard.condition[i, j])
                    cond = shard.color_to_condition.get(cond_rgb, "")

                    si = _intern(subj)
                    ri = _intern(rel)
                    oi = _intern(obj)
                    ci = _intern(cond) if cond else 0xFFFF
                    all_entries.append((si, ri, oi, ci))

        # Deduplicate
        seen = set()
        unique_entries = []
        for e in all_entries:
            if e not in seen:
                seen.add(e)
                unique_entries.append(e)

        path = self.storage_dir / "store.pmem"
        with open(path, "wb") as f:
            f.write(b"PMEM")
            f.write(struct.pack("<HI", len(strings), len(unique_entries)))
            for s in strings:
                encoded = s.encode("utf-8")
                f.write(struct.pack("B", len(encoded)))
                f.write(encoded)
            for si, ri, oi, ci in unique_entries:
                f.write(struct.pack("<4H", si, ri, oi, ci))

    @classmethod
    def load(cls, storage_dir: str | Path) -> "ShardManager":
        """Load a ShardManager and all its shards from disk."""
        d = Path(storage_dir)
        with open(d / "index.json") as f:
            idx = json.load(f)

        mgr = cls(
            storage_dir=d,
            shard_size=idx["shard_size"],
            density_target=idx["density_target"],
        )

        fmt = idx.get("format", "png")
        registry = idx.get("registry", None)

        for shard_info in idx["shards"]:
            shard_dir = d / shard_info["name"]
            if fmt == "sparse" and registry:
                # Load from binary + global registry
                import struct
                n = shard_info.get("n", idx["shard_size"])
                unit = PixelMemUnit(n=n, name=shard_info["name"])

                # Apply global color registry
                unit.relation_color_map = {k: tuple(v) for k, v in registry["rc"].items()}
                unit.color_to_relation = {
                    tuple(int(x) for x in k.split(",")): v for k, v in registry["rr"].items()
                }
                unit.condition_color_map = {k: tuple(v) for k, v in registry["cc"].items()}
                unit.color_to_condition = {
                    tuple(int(x) for x in k.split(",")): v for k, v in registry["cr"].items()
                }
                unit._next_relation_color_id = registry["nri"]
                unit._next_condition_color_id = registry["nci"]
                unit.summary = shard_info.get("summary", "")

                # Per-shard entity mapping from shard_entities in index
                shard_ents = shard_info.get("entities", {})
                unit.entity_to_idx = shard_ents.get("e2i", {})
                unit.idx_to_entity = {int(k): v for k, v in shard_ents.get("i2e", {}).items()}
                unit._next_entity_idx = shard_ents.get("ni", 1)

                # Load binary triples
                triples_path = shard_dir / "triples.bin"
                if triples_path.exists():
                    data = triples_path.read_bytes()
                    for off in range(0, len(data), 8):
                        i, j, r, g, b, cr, cg, cb = struct.unpack("8B", data[off:off+8])
                        unit.relation[i, j] = (r, g, b)
                        unit.condition[i, j] = (cr, cg, cb)

                tapes_path = shard_dir / "tapes.bin"
                if tapes_path.exists():
                    data = tapes_path.read_bytes()
                    for off in range(0, len(data), 9):
                        vals = struct.unpack("9B", data[off:off+9])
                        unit.tape_r.append((vals[0], vals[1], vals[2]))
                        unit.tape_c.append((vals[3], vals[4], vals[5]))
                        unit.tape_e.append((vals[6], vals[7], vals[8]))
            elif fmt == "sparse":
                unit = PixelMemUnit.load_sparse(shard_dir)
            else:
                unit = PixelMemUnit.load(shard_dir)
            unit._topic = shard_info.get("topic", "")
            mgr.shards.append(unit)
            mgr._shard_dirs.append(shard_dir)

        mgr._rebuild_entity_index()
        return mgr

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return statistics about the shard collection."""
        total_triples = sum(s.triple_count() for s in self.shards)
        total_entities = sum(s.entity_count() for s in self.shards)
        return {
            "shard_count": len(self.shards),
            "total_triples": total_triples,
            "total_entities": total_entities,
            "shards": [
                {
                    "name": s.name,
                    "entities": s.entity_count(),
                    "triples": s.triple_count(),
                    "density": round(s.density(), 4),
                }
                for s in self.shards
            ],
        }

    def token_estimate(self, loaded_shards: int = 2) -> dict:
        """Estimate token costs for image vs text encoding."""
        n = self.shard_size
        total_triples = sum(s.triple_count() for s in self.shards)
        avg_tape_len = (
            sum(len(s.tape_r) for s in self.shards) / max(len(self.shards), 1)
        )

        # Per-shard image tokens: 2 matrices + 3 tapes
        matrix_tokens = 2 * (n * n) / 750
        tape_tokens = 3 * avg_tape_len / 750
        json_tokens_est = 50  # rough estimate for sidecars
        per_shard_tokens = matrix_tokens + tape_tokens + json_tokens_est

        image_tokens_per_query = per_shard_tokens * min(
            loaded_shards, len(self.shards)
        )
        text_tokens = 14 * total_triples  # ~14 tokens per triple as text

        return {
            "total_triples": total_triples,
            "image_tokens_per_query": round(image_tokens_per_query),
            "text_tokens_all": round(text_tokens),
            "compression_ratio": (
                round(text_tokens / image_tokens_per_query, 1)
                if image_tokens_per_query > 0 else 0
            ),
            "shards_loaded_per_query": min(loaded_shards, len(self.shards)),
        }


_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "about", "above",
    "after", "again", "all", "also", "and", "any", "because", "before",
    "between", "both", "but", "by", "came", "come", "could", "each",
    "for", "from", "get", "got", "had", "has", "have", "her", "here",
    "him", "his", "how", "if", "in", "into", "its", "just", "let",
    "like", "make", "many", "me", "might", "more", "most", "much",
    "my", "never", "no", "nor", "not", "now", "of", "on", "only",
    "or", "other", "our", "out", "over", "own", "said", "same", "she",
    "so", "some", "still", "such", "take", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "up", "very", "want", "was",
    "way", "we", "well", "were", "what", "when", "where", "which",
    "while", "who", "why", "with", "would", "you", "your",
}
