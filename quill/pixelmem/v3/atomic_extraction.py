"""Atomic Import Extraction — decompose every import into maximal triples.

Instead of:
  (views.py, imports, classify.py)  ← 1 triple, lossy

Decompose to:
  (views.py, imports_from, popnews.classify)
  (views.py, uses_symbol, combine_text_and_image)
  (popnews.classify, provides, combine_text_and_image)
  (popnews, contains_module, classify)
  (views.py, depends_on, classify.py)                  ← resolved

This preserves ALL information from the original import line.
PixelMem's PNG compression handles the extra volume — a 64×64 matrix
stores 4096 cells, more than enough for 3-5 files with ~50 imports each.

The LLM can then trace dependency chains through the pixel store.
"""

from __future__ import annotations

import re
from typing import Optional

from pixelmem.triple_extractor import Triple


def decompose_import(
    source_file: str,
    import_line: str,
    target_file: Optional[str] = None,
) -> list[Triple]:
    """Decompose one import line into atomic triples.

    Args:
        source_file: File containing the import (full path)
        import_line: The raw import statement
        target_file: Resolved target file (if known), or None for external

    Returns:
        List of atomic triples capturing all information.
    """
    triples = []
    src = source_file
    src_bn = source_file.split("/")[-1]
    line = import_line.strip()

    # Parse: from X import Y, Z
    m = re.match(r'^from\s+(\.{0,3}[\w.]*)\s+import\s+([\w, *]+)', line)
    if m:
        from_module = m.group(1).lstrip(".")
        names = [n.strip() for n in m.group(2).split(",") if n.strip()]

        # Module reference
        if from_module:
            triples.append(Triple(src, "imports_from", from_module, line))

            # Package structure: a.b.c → a contains b, b contains c
            parts = from_module.split(".")
            for i in range(len(parts) - 1):
                parent = ".".join(parts[:i+1])
                child = ".".join(parts[:i+2])
                triples.append(Triple(parent, "contains_module", child, "package_structure"))

        # Specific symbols imported
        for name in names:
            if name == "*":
                triples.append(Triple(src, "imports_all_from", from_module, line))
            else:
                triples.append(Triple(src, "uses_symbol", name, f"from {from_module}"))
                if from_module:
                    triples.append(Triple(from_module, "provides", name, "export"))

        # Resolved file dependency
        if target_file and target_file != source_file:
            triples.append(Triple(src, "depends_on", target_file, line))

        return triples

    # Parse: import X, Y, Z
    m = re.match(r'^import\s+([\w., ]+)', line)
    if m:
        modules = [mod.strip() for mod in m.group(1).split(",") if mod.strip()]
        for module in modules:
            if not re.match(r'^[\w.]+$', module):
                continue
            triples.append(Triple(src, "imports_module", module, line))

            # Package structure
            parts = module.split(".")
            for i in range(len(parts) - 1):
                parent = ".".join(parts[:i+1])
                child = ".".join(parts[:i+2])
                triples.append(Triple(parent, "contains_module", child, "package_structure"))

            if target_file and target_file != source_file:
                triples.append(Triple(src, "depends_on", target_file, line))

        return triples

    return triples


def decompose_definition(
    source_file: str,
    def_line: str,
) -> list[Triple]:
    """Decompose a definition line into triples."""
    triples = []
    stripped = def_line.strip()

    m = re.match(r'^def\s+(\w+)\s*\(([^)]*)\)', stripped)
    if m:
        name = m.group(1)
        params = m.group(2).strip()
        triples.append(Triple(source_file, "defines_function", name, f"def {name}({params[:30]})"))
        return triples

    m = re.match(r'^class\s+(\w+)\s*(?:\(([^)]*)\))?', stripped)
    if m:
        name = m.group(1)
        bases = m.group(2) or ""
        triples.append(Triple(source_file, "defines_class", name, f"class {name}({bases[:30]})"))
        if bases:
            for base in bases.split(","):
                base = base.strip()
                if base and base not in ("object",):
                    triples.append(Triple(name, "extends", base.split(".")[-1], "inheritance"))
        return triples

    return triples


def atomic_extract(
    files: list[str],
    file_contents: dict[str, str],
) -> tuple[list[Triple], dict]:
    """Full atomic extraction: decompose ALL imports and definitions.

    Returns all atomic triples + stats.
    PixelMem's PNG compression handles the volume.
    """
    # Build module→file lookup for resolving targets
    module_to_file: dict[str, str] = {}
    for fpath in files:
        basename = fpath.split("/")[-1].replace(".py", "")
        full_module = fpath.replace("/", ".").replace(".py", "")
        parts = full_module.split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            module_to_file[suffix] = fpath
        module_to_file[basename] = fpath

    # Also build basename→files for ambiguity detection
    basename_to_files: dict[str, list[str]] = {}
    for fpath in files:
        bn = fpath.split("/")[-1]
        basename_to_files.setdefault(bn, []).append(fpath)

    all_triples = []
    n_imports = 0
    n_defs = 0
    n_resolved = 0

    for fpath, code in file_contents.items():
        for line in code.split("\n"):
            stripped = line.strip()

            # Import lines
            if re.match(r'^\s*(?:from|import)\s', stripped):
                n_imports += 1

                # Try to resolve target file
                target = _resolve_target(stripped, fpath, files, module_to_file)
                if target:
                    n_resolved += 1

                import_triples = decompose_import(fpath, stripped, target)
                all_triples.extend(import_triples)
                continue

            # Definition lines
            if re.match(r'^\s*(?:def|class)\s+\w+', stripped):
                n_defs += 1
                def_triples = decompose_definition(fpath, stripped)
                all_triples.extend(def_triples)

    stats = {
        "n_files": len(files),
        "n_imports": n_imports,
        "n_defs": n_defs,
        "n_resolved": n_resolved,
        "n_triples": len(all_triples),
        "avg_triples_per_file": len(all_triples) / max(1, len(files)),
    }

    return all_triples, stats


def _resolve_target(
    import_line: str,
    source_file: str,
    files: list[str],
    module_to_file: dict[str, str],
) -> Optional[str]:
    """Try to resolve an import line to a target file."""
    m = re.match(r'^from\s+(\.{0,3}[\w.]*)\s+import\s+([\w, *]+)', import_line)
    if m:
        from_mod = m.group(1).lstrip(".")
        names = [n.strip() for n in m.group(2).split(",")]
        for name in names:
            name = name.strip()
            if name == "*":
                candidates = [from_mod]
            else:
                candidates = [f"{from_mod}.{name}", from_mod, name]
            for c in candidates:
                t = module_to_file.get(c)
                if t and t != source_file:
                    return t
        # Try basename of from_module parts
        if from_mod:
            for part in from_mod.split("."):
                t = module_to_file.get(part)
                if t and t != source_file:
                    return t
        return None

    m = re.match(r'^import\s+([\w., ]+)', import_line)
    if m:
        for mod in m.group(1).split(","):
            mod = mod.strip()
            if mod:
                t = module_to_file.get(mod)
                if t and t != source_file:
                    return t
                for part in mod.split("."):
                    t = module_to_file.get(part)
                    if t and t != source_file:
                        return t
    return None
