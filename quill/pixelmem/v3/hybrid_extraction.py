"""Hybrid Extraction: deterministic first, LLM only for unresolved imports.

Stage 1: AST/regex extracts all import statements (free, instant)
Stage 2: Deterministic resolver maps imports to known files (free)
Stage 3: ONLY unresolved imports go to LLM with minimal context:
         "File X has 'from foo.bar import baz'. Which of [a.py, b.py, c.py]
          does this import from?"
         (~50 tokens per unresolved import, not 40K for full code)

This closes the extraction gap while keeping token cost <500 per question.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from pixelmem.triple_extractor import Triple


def extract_all_imports(file_contents: dict[str, str]) -> dict[str, list[str]]:
    """Stage 1: Extract all import lines from each file (deterministic)."""
    imports: dict[str, list[str]] = {}
    for fpath, code in file_contents.items():
        file_imports = []
        for line in code.split("\n"):
            line = line.strip()
            if re.match(r'^(?:from|import)\s', line):
                file_imports.append(line)
        imports[fpath] = file_imports
    return imports


def resolve_deterministic(
    imports_by_file: dict[str, list[str]],
    files: list[str],
) -> tuple[list[Triple], list[tuple[str, str]]]:
    """Stage 2: Resolve imports deterministically. Return (resolved, unresolved).

    resolved: list of Triple(source_file, "imports", target_file, condition)
    unresolved: list of (source_file, import_line) that couldn't be mapped
    """
    # Build module→file lookup
    module_to_file: dict[str, str] = {}
    basename_to_files: dict[str, list[str]] = {}  # basename → [full paths] (for ambiguity detection)

    for fpath in files:
        basename = fpath.split("/")[-1].replace(".py", "")
        full_module = fpath.replace("/", ".").replace(".py", "")

        # Register all suffixes
        parts = full_module.split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            module_to_file[suffix] = fpath

        basename_to_files.setdefault(basename, []).append(fpath)
        module_to_file[basename] = fpath

    # Track parent packages
    file_packages: dict[str, set[str]] = {}
    for fpath in files:
        parts = fpath.replace("/", ".").replace(".py", "").split(".")
        file_packages[fpath] = set()
        for i in range(len(parts)):
            file_packages[fpath].add(".".join(parts[:i + 1]))
            file_packages[fpath].add(parts[i])

    resolved: list[Triple] = []
    unresolved: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()

    for source_file, import_lines in imports_by_file.items():
        for line in import_lines:
            target = _try_resolve(line, source_file, files, module_to_file, file_packages)
            if target and target != source_file:
                edge = (source_file, target)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    resolved.append(Triple(
                        source_file, "imports", target,
                        f"{source_file.split('/')[-1]}→{target.split('/')[-1]}",
                    ))
            elif _references_internal(line, file_packages, source_file):
                # Looks like it references our files but couldn't resolve
                unresolved.append((source_file, line))

    return resolved, unresolved


def _try_resolve(
    import_line: str,
    source_file: str,
    files: list[str],
    module_to_file: dict[str, str],
    file_packages: dict[str, set[str]],
) -> Optional[str]:
    """Try to resolve a single import line to a target file."""
    # from X import Y
    m = re.match(r'^from\s+(\.{0,3}[\w.]*)\s+import\s+([\w, *]+)', import_line)
    if m:
        from_mod = m.group(1)
        names = [n.strip() for n in m.group(2).split(",")]
        clean = from_mod.lstrip(".")

        # Try: from_module.name, from_module, name
        for name in names:
            name = name.strip()
            if name == "*":
                candidates = [clean]
            else:
                candidates = [f"{clean}.{name}", clean, name]

            for c in candidates:
                target = module_to_file.get(c)
                if target and target != source_file:
                    return target

        # Relative imports
        if from_mod.startswith("."):
            dots = len(from_mod) - len(from_mod.lstrip("."))
            source_parts = source_file.split("/")
            if dots < len(source_parts):
                base = source_parts[:len(source_parts) - dots]
                if clean:
                    for part in clean.split("."):
                        target = module_to_file.get(part)
                        if target and target != source_file:
                            return target

        return None

    # import X or import X, Y, Z
    m = re.match(r'^import\s+([\w., ]+)', import_line)
    if m:
        for module in m.group(1).split(","):
            module = module.strip()
            if module and re.match(r'^[\w.]+$', module):
                target = module_to_file.get(module)
                if target and target != source_file:
                    return target

    return None


def _references_internal(
    import_line: str,
    file_packages: dict[str, set[str]],
    source_file: str,
) -> bool:
    """Check if an import line likely references our file set."""
    # Extract module name
    m = re.match(r'^(?:from\s+([\w.]+)|import\s+([\w.]+))', import_line.lstrip("."))
    if not m:
        return False
    module = (m.group(1) or m.group(2) or "").split(".")[0]
    if not module or len(module) < 3:
        return False

    # Check if any file's package tree contains this root
    for fpath, packages in file_packages.items():
        if fpath == source_file:
            continue
        if module in packages:
            return True
    return False


def resolve_with_llm(
    unresolved: list[tuple[str, str]],
    files: list[str],
    ask_fn,  # function(prompt) -> answer
    max_workers: int = 10,
) -> list[Triple]:
    """Stage 3: Use LLM to resolve ambiguous imports.

    Sends ONLY the import line + file list to the LLM (~50 tokens each).
    NOT the full source code.
    """
    if not unresolved:
        return []

    # Group unresolved by source file to batch
    by_source: dict[str, list[str]] = {}
    for source, line in unresolved:
        by_source.setdefault(source, []).append(line)

    file_basenames = [f.split("/")[-1] for f in files]
    file_list = ", ".join(file_basenames)

    resolved: list[Triple] = []
    seen: set[tuple[str, str]] = set()

    def resolve_batch(source_file: str, import_lines: list[str]):
        """Ask LLM to resolve a batch of imports from one file."""
        source_bn = source_file.split("/")[-1]
        imports_text = "\n".join(f"  {line}" for line in import_lines[:5])

        prompt = (
            f"File '{source_bn}' has these import statements:\n{imports_text}\n\n"
            f"Available files: {file_list}\n\n"
            f"Which of these files does '{source_bn}' import from? "
            f"Return ONLY a JSON array of filenames that '{source_bn}' depends on. "
            f"Exclude '{source_bn}' itself and external packages."
        )

        answer = ask_fn(prompt)
        if isinstance(answer, tuple):
            answer = answer[0]  # (text, in_tok, out_tok)

        # Parse response
        batch_triples = []
        try:
            m = re.search(r'\[.*?\]', answer, re.DOTALL)
            if m:
                import json
                targets = json.loads(m.group(0))
                for target_bn in targets:
                    target_bn = target_bn.strip("'\" ")
                    # Map basename back to full path
                    for fpath in files:
                        if fpath.split("/")[-1] == target_bn and fpath != source_file:
                            edge = (source_file, fpath)
                            if edge not in seen:
                                seen.add(edge)
                                batch_triples.append(Triple(
                                    source_file, "imports", fpath,
                                    f"{source_bn}→{target_bn} [llm_resolved]",
                                ))
                            break
        except Exception:
            pass
        return batch_triples

    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(resolve_batch, src, lines): src
            for src, lines in by_source.items()
        }
        for f in as_completed(futures):
            try:
                resolved.extend(f.result())
            except Exception:
                pass

    return resolved


def hybrid_extract(
    files: list[str],
    file_contents: dict[str, str],
    ask_fn=None,
    max_workers: int = 10,
) -> tuple[list[Triple], dict]:
    """Full hybrid extraction pipeline.

    Returns (triples, stats) where stats has extraction metrics.
    """
    # Stage 1: Get all import lines
    imports_by_file = extract_all_imports(file_contents)
    total_imports = sum(len(v) for v in imports_by_file.values())

    # Stage 2: Deterministic resolution
    resolved, unresolved = resolve_deterministic(imports_by_file, files)

    stats = {
        "total_imports": total_imports,
        "resolved_deterministic": len(resolved),
        "unresolved": len(unresolved),
        "resolved_llm": 0,
        "total_edges": len(resolved),
        "method": "deterministic",
    }

    # Stage 3: LLM resolution (only if ask_fn provided and there are unresolved)
    if ask_fn and unresolved:
        llm_resolved = resolve_with_llm(unresolved, files, ask_fn, max_workers)
        resolved.extend(llm_resolved)
        stats["resolved_llm"] = len(llm_resolved)
        stats["total_edges"] = len(resolved)
        stats["method"] = "hybrid"

    return resolved, stats
