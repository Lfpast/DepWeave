"""Dual Condition Layer — two condition matrices for richer metadata.

Extends PixelMem with a second condition matrix stored alongside
the existing one. No modification to core PixelMemUnit needed.

Condition 1 (existing): stores the primary condition (import line, date, etc.)
Condition 2 (new): stores disambiguation context (package path, source info)

For DependEval __init__.py disambiguation:
  Relation:    (__init__.py, imports, core.py)
  Condition 1: "from .core import X"           ← what it imports
  Condition 2: "pkg=criteria parent=lightseq"   ← which __init__ this is

The second condition is stored as additional triples in the same shard:
  (file, _pkg_context, file, "pkg=criteria parent=lightseq")

This uses PixelMem's existing relation matrix — no new PNG needed.
The _pkg_context relation is just another color in the relation matrix.
"""

from __future__ import annotations

import re
from typing import Optional

from pixelmem.triple_extractor import Triple


def build_context_triples(
    files: list[str],
    file_contents: dict[str, str],
) -> list[Triple]:
    """Build second-layer context triples for disambiguation.

    For each file, stores:
      - Package path (which directory it's in)
      - Parent package name
      - Unique identifiers (first defined name, unique imports)
      - File size category (small/medium/large)

    These help the LLM distinguish files with the same basename.
    """
    triples = []
    basenames = [f.split("/")[-1] for f in files]
    has_duplicates = len(set(basenames)) < len(basenames)

    for fpath in files:
        code = file_contents.get(fpath, "")
        bn = fpath.split("/")[-1]
        parts = fpath.split("/")

        # Package context
        parent_dir = parts[-2] if len(parts) >= 2 else ""
        pkg_path = "/".join(parts[:-1]) if len(parts) >= 2 else ""

        # First defined name (unique identifier)
        first_def = ""
        for line in code.split("\n"):
            m = re.match(r'^\s*(?:def|class)\s+(\w+)', line.strip())
            if m:
                first_def = m.group(1)
                break

        # Count imports and definitions
        n_imports = sum(1 for l in code.split("\n") if re.match(r'^\s*(?:from|import)\s', l.strip()))
        n_defs = sum(1 for l in code.split("\n") if re.match(r'^\s*(?:def|class)\s', l.strip()))
        n_lines = len(code.split("\n"))

        # Build context string
        ctx_parts = []
        if pkg_path:
            ctx_parts.append(f"pkg={pkg_path}")
        if parent_dir:
            ctx_parts.append(f"parent={parent_dir}")
        if first_def:
            ctx_parts.append(f"first_def={first_def}")
        ctx_parts.append(f"imports={n_imports}")
        ctx_parts.append(f"defs={n_defs}")
        ctx_parts.append(f"lines={n_lines}")

        context = " ".join(ctx_parts)

        # Store as a _pkg_context relation
        triples.append(Triple(fpath, "_pkg_context", fpath, context))

        # For duplicate basenames, also store a unique tag
        if has_duplicates and basenames.count(bn) > 1:
            unique_tag = parent_dir or f"idx{files.index(fpath)}"
            triples.append(Triple(fpath, "_unique_tag", bn, unique_tag))

    return triples


def get_file_context(
    fpath: str,
    mgr: "ShardManager",
) -> dict:
    """Read second-layer context from PixelMem for a file.

    Returns dict with: pkg, parent, first_def, n_imports, n_defs, n_lines, unique_tag
    """
    from pixelmem.memory import BLACK

    canonical = fpath.strip().lower()
    context = {}

    for shard in mgr.shards:
        if canonical not in shard.entity_to_idx:
            continue
        idx = shard.entity_to_idx[canonical]

        for j in range(shard.n):
            rgb = tuple(int(x) for x in shard.relation[idx, j])
            if rgb == (0, 0, 0):
                continue
            rel = shard.color_to_relation.get(rgb, "?")

            if rel == "_pkg_context":
                crgb = tuple(int(x) for x in shard.condition[idx, j])
                cond = shard.color_to_condition.get(crgb, "")
                # Parse "pkg=X parent=Y first_def=Z imports=N defs=M lines=L"
                for part in cond.split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        context[k] = v

            elif rel == "_unique_tag":
                crgb = tuple(int(x) for x in shard.condition[idx, j])
                cond = shard.color_to_condition.get(crgb, "")
                context["unique_tag"] = cond

    return context


def format_with_context(
    files: list[str],
    summaries: dict[str, dict],
    mgr: "ShardManager",
) -> str:
    """Format file overview with second-layer context for disambiguation.

    Only adds context when files have duplicate basenames.
    Regular files get simple format (which works best for LLM).
    """
    basenames = [f.split("/")[-1] for f in files]
    has_duplicates = len(set(basenames)) < len(basenames)

    lines = []
    for fpath in files:
        s = summaries.get(fpath, {})
        bn = s.get("filename", fpath.split("/")[-1])

        if has_duplicates and basenames.count(bn) > 1:
            # Duplicate — add context from second condition layer
            ctx = get_file_context(fpath, mgr)
            parent = ctx.get("parent", "")
            if parent:
                label = f"{bn} (in {parent}/)"
            else:
                label = fpath  # fallback to full path
        else:
            label = bn

        lines.append(f"=== {label} ===")
        if s.get("imports"):
            lines.append("  imports: " + "; ".join(s["imports"][:10]))
        if s.get("definitions"):
            lines.append("  defines: " + ", ".join(s["definitions"][:10]))

    return "\n".join(lines)
