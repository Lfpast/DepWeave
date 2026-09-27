"""Core PixelMem data structures: relation/condition matrices and chunk tapes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# Entity index 0 is reserved (stop signal uses (0,0,0) on all tapes).
# Real entities start at index 1.
RESERVED_INDEX = 0
BLACK = (0, 0, 0)


class PixelMemUnit:
    """A single PixelMem memory unit.

    Contains:
      - relation matrix (N x N x 3, uint8) — RGB encodes relation type
      - condition matrix (N x N x 3, uint8) — RGB encodes metadata
      - tape_r, tape_c, tape_e — sequential pixel streams for chunk ordering
      - entities: bidirectional name <-> index mapping
      - colors: bidirectional RGB <-> relation/condition name mapping
    """

    def __init__(self, n: int = 64, name: str = "default"):
        self.name = name
        self.n = n
        # Knowledge store matrices — initialised to black (no relation)
        self.relation = np.zeros((n, n, 3), dtype=np.uint8)
        self.condition = np.zeros((n, n, 3), dtype=np.uint8)
        # Chunk tapes — lists of (R, G, B) tuples
        self.tape_r: list[tuple[int, int, int]] = []
        self.tape_c: list[tuple[int, int, int]] = []
        self.tape_e: list[tuple[int, int, int]] = []
        # Entity mapping: name -> index, index -> name
        self.entity_to_idx: dict[str, int] = {}
        self.idx_to_entity: dict[int, str] = {}
        self._next_entity_idx = RESERVED_INDEX + 1  # start at 1
        # Color mapping: relation_name -> RGB, RGB -> relation_name
        self.relation_color_map: dict[str, tuple[int, int, int]] = {}
        self.color_to_relation: dict[tuple[int, int, int], str] = {}
        self.condition_color_map: dict[str, tuple[int, int, int]] = {}
        self.color_to_condition: dict[tuple[int, int, int], str] = {}
        self._next_relation_color_id = 1
        self._next_condition_color_id = 1
        # Summary text for shard routing
        self.summary: str = ""

    # ── Entity resolution ───────────────────────────────────────────

    def resolve_entity(self, name: str) -> int:
        """Map an entity name to a matrix index, creating a new one if needed."""
        canonical = self._canonicalize(name)
        if canonical in self.entity_to_idx:
            return self.entity_to_idx[canonical]
        idx = self._next_entity_idx
        if idx >= self.n:
            raise RuntimeError(
                f"Entity capacity exhausted (N={self.n}). "
                "Create a new shard or increase N."
            )
        self.entity_to_idx[canonical] = idx
        self.idx_to_entity[idx] = canonical
        self._next_entity_idx += 1
        return idx

    def has_entity(self, name: str) -> bool:
        return self._canonicalize(name) in self.entity_to_idx

    def entity_count(self) -> int:
        return len(self.entity_to_idx)

    @staticmethod
    def _canonicalize(name: str) -> str:
        """Aggressive entity canonicalisation — lowercase, strip whitespace."""
        return name.strip().lower()

    # ── Color resolution ────────────────────────────────────────────

    def resolve_relation_color(self, relation: str) -> tuple[int, int, int]:
        """Map a relation name to an RGB color, creating a new one if needed."""
        key = relation.strip().lower()
        if key in self.relation_color_map:
            return self.relation_color_map[key]
        rgb = self._id_to_rgb(self._next_relation_color_id)
        self._next_relation_color_id += 1
        self.relation_color_map[key] = rgb
        self.color_to_relation[rgb] = key
        return rgb

    def resolve_condition_color(self, condition: str) -> tuple[int, int, int]:
        """Map a condition name to an RGB color, creating a new one if needed."""
        key = condition.strip().lower()
        if not key:
            return BLACK
        if key in self.condition_color_map:
            return self.condition_color_map[key]
        rgb = self._id_to_rgb(self._next_condition_color_id)
        self._next_condition_color_id += 1
        self.condition_color_map[key] = rgb
        self.color_to_condition[rgb] = key
        return rgb

    # Maximally distinct colors for the first 24 relation/condition types.
    # Hand-picked to be visually distinguishable even at small pixel sizes.
    _DISTINCT_COLORS: list[tuple[int, int, int]] = [
        (255, 0, 0),       # red
        (0, 255, 0),       # green
        (0, 0, 255),       # blue
        (255, 255, 0),     # yellow
        (255, 0, 255),     # magenta
        (0, 255, 255),     # cyan
        (255, 128, 0),     # orange
        (128, 0, 255),     # purple
        (0, 255, 128),     # spring green
        (255, 0, 128),     # rose
        (0, 128, 255),     # azure
        (128, 255, 0),     # chartreuse
        (128, 128, 0),     # olive
        (128, 0, 128),     # dark purple
        (0, 128, 128),     # teal
        (255, 128, 128),   # salmon
        (128, 255, 128),   # light green
        (128, 128, 255),   # light blue
        (255, 255, 128),   # light yellow
        (255, 128, 255),   # pink
        (128, 255, 255),   # light cyan
        (192, 64, 0),      # brown
        (64, 192, 0),      # dark green
        (0, 64, 192),      # navy
    ]

    @staticmethod
    def _id_to_rgb(color_id: int) -> tuple[int, int, int]:
        """Deterministically convert a sequential ID to an RGB tuple.

        Uses a hand-picked palette of maximally distinct colors for the
        first 24 IDs, then falls back to hash-based generation.
        Skips (0,0,0) which is reserved as the no-relation / stop signal.
        """
        if color_id <= 0:
            raise ValueError("color_id must be positive (0 is reserved)")
        if color_id <= len(PixelMemUnit._DISTINCT_COLORS):
            return PixelMemUnit._DISTINCT_COLORS[color_id - 1]
        # Fallback: large prime hash for IDs beyond the palette
        spread = (color_id * 5592413) % (256**3)
        if spread == 0:
            spread = 1
        r = (spread >> 16) & 0xFF
        g = (spread >> 8) & 0xFF
        b = spread & 0xFF
        if (r, g, b) == (0, 0, 0):
            b = 1
        return (r, g, b)

    # ── Matrix density ──────────────────────────────────────────────

    def density(self) -> float:
        """Fraction of matrix cells that contain a non-black relation."""
        total = self.n * self.n
        filled = np.count_nonzero(np.any(self.relation != 0, axis=-1))
        return filled / total

    def triple_count(self) -> int:
        """Number of stored triples (non-black pixels in relation matrix)."""
        return int(np.count_nonzero(np.any(self.relation != 0, axis=-1)))

    # ── Save / Load ─────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Persist the memory unit to disk as PNGs + JSON sidecars."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        # Matrices
        Image.fromarray(self.relation).save(d / "relation.png")
        Image.fromarray(self.condition).save(d / "condition.png")

        # Tapes — reshape to 2D images (width up to 256, height as needed)
        for tape, fname in [
            (self.tape_r, "order_r.png"),
            (self.tape_c, "order_c.png"),
            (self.tape_e, "order_e.png"),
        ]:
            self._save_tape(tape, d / fname)

        # Sidecar JSON
        entities_data = {
            "entity_to_idx": self.entity_to_idx,
            "idx_to_entity": {str(k): v for k, v in self.idx_to_entity.items()},
            "next_idx": self._next_entity_idx,
        }
        with open(d / "entities.json", "w") as f:
            json.dump(entities_data, f, indent=2)

        colors_data = {
            "relation": {k: list(v) for k, v in self.relation_color_map.items()},
            "relation_reverse": {
                f"{r},{g},{b}": name
                for (r, g, b), name in self.color_to_relation.items()
            },
            "condition": {k: list(v) for k, v in self.condition_color_map.items()},
            "condition_reverse": {
                f"{r},{g},{b}": name
                for (r, g, b), name in self.color_to_condition.items()
            },
            "next_relation_id": self._next_relation_color_id,
            "next_condition_id": self._next_condition_color_id,
        }
        with open(d / "colors.json", "w") as f:
            json.dump(colors_data, f, indent=2)

        # Summary
        with open(d / "summary.txt", "w") as f:
            f.write(self.summary)

    @classmethod
    def load(cls, directory: str | Path) -> "PixelMemUnit":
        """Load a memory unit from disk."""
        d = Path(directory)

        # Load matrices
        relation = np.array(Image.open(d / "relation.png").convert("RGB"))
        condition = np.array(Image.open(d / "condition.png").convert("RGB"))
        n = relation.shape[0]

        unit = cls(n=n, name=d.name)
        unit.relation = relation
        unit.condition = condition

        # Load tapes
        unit.tape_r = cls._load_tape(d / "order_r.png")
        unit.tape_c = cls._load_tape(d / "order_c.png")
        unit.tape_e = cls._load_tape(d / "order_e.png")

        # Load entities
        with open(d / "entities.json") as f:
            ent = json.load(f)
        unit.entity_to_idx = ent["entity_to_idx"]
        unit.idx_to_entity = {int(k): v for k, v in ent["idx_to_entity"].items()}
        unit._next_entity_idx = ent["next_idx"]

        # Load colors
        with open(d / "colors.json") as f:
            col = json.load(f)
        unit.relation_color_map = {k: tuple(v) for k, v in col["relation"].items()}
        unit.color_to_relation = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in col["relation_reverse"].items()
        }
        unit.condition_color_map = {k: tuple(v) for k, v in col["condition"].items()}
        unit.color_to_condition = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in col["condition_reverse"].items()
        }
        unit._next_relation_color_id = col["next_relation_id"]
        unit._next_condition_color_id = col["next_condition_id"]

        # Summary
        summary_path = d / "summary.txt"
        if summary_path.exists():
            unit.summary = summary_path.read_text()

        return unit

    # ── Sparse storage ──────────────────────────────────────────────

    def save_sparse(self, directory: str | Path) -> None:
        """Persist as sparse coordinate lists — much smaller than full PNGs.

        Format:
          - triples.bin: packed (i, j, r, g, b, cr, cg, cb) per non-zero pixel
            8 bytes per triple (uint8 each)
          - tapes.bin: packed tape entries (same as before but binary)
          - meta.json: entity/color mappings + dimensions (replaces
            separate entities.json + colors.json)
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        # Sparse triples: only non-zero pixels
        coords = []
        for i in range(self.n):
            for j in range(self.n):
                rgb = tuple(int(x) for x in self.relation[i, j])
                if rgb == (0, 0, 0):
                    continue
                crgb = tuple(int(x) for x in self.condition[i, j])
                coords.append((i, j, *rgb, *crgb))

        # Pack as binary: 8 bytes per triple (i, j, r, g, b, cr, cg, cb)
        import struct
        with open(d / "triples.bin", "wb") as f:
            for c in coords:
                f.write(struct.pack("8B", *c))

        # Tapes: binary
        tape_data = []
        tape_len = min(len(self.tape_r), len(self.tape_c), len(self.tape_e))
        for k in range(tape_len):
            r, c, e = self.tape_r[k], self.tape_c[k], self.tape_e[k]
            tape_data.append((*r, *c, *e))
        with open(d / "tapes.bin", "wb") as f:
            for t in tape_data:
                f.write(struct.pack("9B", *t))

        # Single compact meta.json (no indent, minimal keys)
        meta = {
            "n": self.n,
            "name": self.name,
            "e2i": self.entity_to_idx,
            "i2e": {str(k): v for k, v in self.idx_to_entity.items()},
            "ni": self._next_entity_idx,
            "rc": {k: list(v) for k, v in self.relation_color_map.items()},
            "rr": {f"{r},{g},{b}": n for (r, g, b), n in self.color_to_relation.items()},
            "cc": {k: list(v) for k, v in self.condition_color_map.items()},
            "cr": {f"{r},{g},{b}": n for (r, g, b), n in self.color_to_condition.items()},
            "nri": self._next_relation_color_id,
            "nci": self._next_condition_color_id,
            "s": self.summary,
        }
        with open(d / "meta.json", "w") as f:
            json.dump(meta, f, separators=(",", ":"))

    @classmethod
    def load_sparse(cls, directory: str | Path) -> "PixelMemUnit":
        """Load from sparse format."""
        import struct
        d = Path(directory)

        with open(d / "meta.json") as f:
            meta = json.load(f)

        n = meta["n"]
        unit = cls(n=n, name=meta.get("name", d.name))
        unit.entity_to_idx = meta["e2i"]
        unit.idx_to_entity = {int(k): v for k, v in meta["i2e"].items()}
        unit._next_entity_idx = meta["ni"]
        unit.relation_color_map = {k: tuple(v) for k, v in meta["rc"].items()}
        unit.color_to_relation = {
            tuple(int(x) for x in k.split(",")): v for k, v in meta["rr"].items()
        }
        unit.condition_color_map = {k: tuple(v) for k, v in meta["cc"].items()}
        unit.color_to_condition = {
            tuple(int(x) for x in k.split(",")): v for k, v in meta["cr"].items()
        }
        unit._next_relation_color_id = meta["nri"]
        unit._next_condition_color_id = meta["nci"]
        unit.summary = meta.get("s", "")

        # Load sparse triples into matrix
        triples_path = d / "triples.bin"
        if triples_path.exists():
            data = triples_path.read_bytes()
            for off in range(0, len(data), 8):
                i, j, r, g, b, cr, cg, cb = struct.unpack("8B", data[off:off+8])
                unit.relation[i, j] = (r, g, b)
                unit.condition[i, j] = (cr, cg, cb)

        # Load tapes
        tapes_path = d / "tapes.bin"
        if tapes_path.exists():
            data = tapes_path.read_bytes()
            for off in range(0, len(data), 9):
                vals = struct.unpack("9B", data[off:off+9])
                unit.tape_r.append((vals[0], vals[1], vals[2]))
                unit.tape_c.append((vals[3], vals[4], vals[5]))
                unit.tape_e.append((vals[6], vals[7], vals[8]))

        return unit

    # ── Compact binary format ──────────────────────────────────────

    def save_compact(self, path: str | Path) -> None:
        """Save as a single compact binary file with string table.

        Format:
          [4B] magic "PMEM"
          [2B] n (matrix size)
          [2B] n_strings (string table entries)
          [2B] n_triples
          [2B] n_tapes
          --- string table ---
          For each string: [1B length] [N bytes UTF-8]
          --- triple entries ---
          For each: [2B subj_idx] [2B rel_idx] [2B obj_idx] [2B cond_idx]
            (cond_idx = 0xFFFF if no condition)
          --- tape entries ---
          For each: [2B subj_idx] [2B rel_idx] [2B obj_idx] [2B cond_idx] [1B is_stop]
          --- summary ---
          Remaining bytes = UTF-8 summary text

        Total per triple: 8 bytes + amortized string table.
        """
        import struct
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Build string table: all unique entity + relation + condition names
        strings: list[str] = []
        str2idx: dict[str, int] = {}

        def _intern(s: str) -> int:
            if s not in str2idx:
                str2idx[s] = len(strings)
                strings.append(s)
            return str2idx[s]

        # Intern all names
        for name in self.entity_to_idx:
            _intern(name)
        for name in self.relation_color_map:
            _intern(name)
        for name in self.condition_color_map:
            _intern(name)

        # Collect triples as (subj_name, rel_name, obj_name, cond_name)
        triple_entries = []
        for i in range(self.n):
            for j in range(self.n):
                rgb = tuple(int(x) for x in self.relation[i, j])
                if rgb == (0, 0, 0):
                    continue
                subj = self.idx_to_entity.get(i, f"entity_{i}")
                obj = self.idx_to_entity.get(j, f"entity_{j}")
                rel = self.color_to_relation.get(rgb, f"rel_{rgb}")
                cond_rgb = tuple(int(x) for x in self.condition[i, j])
                cond = self.color_to_condition.get(cond_rgb, "")

                si = _intern(subj)
                ri = _intern(rel)
                oi = _intern(obj)
                ci = _intern(cond) if cond else 0xFFFF
                triple_entries.append((si, ri, oi, ci))

        # Tape entries
        tape_entries = []
        tape_len = min(len(self.tape_r), len(self.tape_c), len(self.tape_e))
        from pixelmem.memory import BLACK
        for k in range(tape_len):
            r_px, c_px, e_px = self.tape_r[k], self.tape_c[k], self.tape_e[k]
            if r_px == BLACK and c_px == BLACK and e_px == BLACK:
                tape_entries.append((0, 0, 0, 0, 1))  # stop marker
                continue
            i, j = e_px[0], e_px[1]
            subj = self.idx_to_entity.get(i, f"entity_{i}")
            obj = self.idx_to_entity.get(j, f"entity_{j}")
            rel = self.color_to_relation.get(tuple(r_px), "")
            cond = self.color_to_condition.get(tuple(c_px), "")
            si = _intern(subj)
            ri = _intern(rel)
            oi = _intern(obj)
            ci = _intern(cond) if cond else 0xFFFF
            tape_entries.append((si, ri, oi, ci, 0))

        # Write binary
        with open(p, "wb") as f:
            f.write(b"PMEM")
            f.write(struct.pack("<4H", self.n, len(strings), len(triple_entries), len(tape_entries)))
            # String table
            for s in strings:
                encoded = s.encode("utf-8")
                f.write(struct.pack("B", len(encoded)))
                f.write(encoded)
            # Triples
            for si, ri, oi, ci in triple_entries:
                f.write(struct.pack("<4H", si, ri, oi, ci))
            # Tapes
            for si, ri, oi, ci, stop in tape_entries:
                f.write(struct.pack("<4HB", si, ri, oi, ci, stop))
            # Summary
            f.write(self.summary.encode("utf-8"))

    @classmethod
    def load_compact(cls, path: str | Path) -> "PixelMemUnit":
        """Load from compact binary format."""
        import struct
        p = Path(path)
        data = p.read_bytes()
        off = 0

        # Header
        magic = data[off:off+4]
        off += 4
        assert magic == b"PMEM", f"Invalid magic: {magic}"
        n, n_strings, n_triples, n_tapes = struct.unpack_from("<4H", data, off)
        off += 8

        # String table
        strings = []
        for _ in range(n_strings):
            slen = data[off]
            off += 1
            s = data[off:off+slen].decode("utf-8")
            off += slen
            strings.append(s)

        unit = cls(n=n, name=p.stem)

        # Triples: rebuild entity/color mappings and matrix
        color_id = 1
        for _ in range(n_triples):
            si, ri, oi, ci = struct.unpack_from("<4H", data, off)
            off += 8
            subj, rel, obj = strings[si], strings[ri], strings[oi]
            cond = strings[ci] if ci != 0xFFFF else ""

            subj_idx = unit.resolve_entity(subj)
            obj_idx = unit.resolve_entity(obj)
            rgb = unit.resolve_relation_color(rel)
            unit.relation[subj_idx, obj_idx] = rgb

            if cond:
                crgb = unit.resolve_condition_color(cond)
                unit.condition[subj_idx, obj_idx] = crgb

        # Tapes
        for _ in range(n_tapes):
            si, ri, oi, ci, stop = struct.unpack_from("<4HB", data, off)
            off += 9
            if stop:
                unit.tape_r.append(BLACK)
                unit.tape_c.append(BLACK)
                unit.tape_e.append(BLACK)
            else:
                subj, rel, obj = strings[si], strings[ri], strings[oi]
                cond = strings[ci] if ci != 0xFFFF else ""
                subj_idx = unit.entity_to_idx.get(subj, 0)
                obj_idx = unit.entity_to_idx.get(obj, 0)
                unit.tape_r.append(unit.relation_color_map.get(rel, BLACK))
                unit.tape_c.append(unit.condition_color_map.get(cond, BLACK) if cond else BLACK)
                unit.tape_e.append((subj_idx, obj_idx, 0))

        # Summary
        if off < len(data):
            unit.summary = data[off:].decode("utf-8")

        return unit

    # ── Tape helpers ────────────────────────────────────────────────

    @staticmethod
    def _save_tape(tape: list[tuple[int, int, int]], path: Path) -> None:
        if not tape:
            # Save a 1x1 black image as placeholder
            Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(path)
            return
        width = min(256, len(tape))
        height = (len(tape) + width - 1) // width
        # Pad to fill the rectangle
        padded = list(tape) + [(0, 0, 0)] * (width * height - len(tape))
        arr = np.array(padded, dtype=np.uint8).reshape(height, width, 3)
        Image.fromarray(arr).save(path)

    @staticmethod
    def _load_tape(path: Path) -> list[tuple[int, int, int]]:
        img = np.array(Image.open(path).convert("RGB"))
        flat = img.reshape(-1, 3)
        return [tuple(int(x) for x in px) for px in flat]
