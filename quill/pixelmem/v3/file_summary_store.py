"""File Summary Store — store raw per-file summaries in PixelMem.

Simple approach: one triple per file containing ALL its import lines
and definitions in the condition field. This preserves the EXACT format
that the LLM reasons best from.

  (file_path, file_summary, file_path, "imports: from X import Y; from Z import W\ndefines: func1, func2")

The condition field holds the raw summary. The pixel matrix compresses it.
On retrieval, we read the condition back — identical to raw file scanning.
"""

from __future__ import annotations

import json
import re
import tempfile
from typing import Optional

from pixelmem.shard_manager import ShardManager
from pixelmem.triple_extractor import Triple


def build_file_summary_triples(
    files: list[str],
    file_contents: dict[str, str],
    ask_fn=None,
) -> list[Triple]:
    """Build one summary triple per file containing imports + definitions.

    Uses lang_extraction to detect language and build/select the right
    extraction tool. Works for ANY language, not just Python.

    Pipeline:
      1. Detect language from file extensions
      2. Get/build extractor for that language (cached per language)
      3. Extract import lines using the tool
      4. Extract definitions using language-appropriate patterns
      5. Store as file_summary triples with raw lines in condition
      6. Also store inter-file dependency edges

    Args:
        files: List of file paths
        file_contents: Dict of file path → source code
        ask_fn: Optional LLM function for building extractors for unknown languages
    """
    from pixelmem.v3.tool_registry import get_registry

    registry = get_registry()

    # Step 1: Get/build tool from persistent registry (cached per language)
    sample_code = next(iter(file_contents.values()), "")
    tool = registry.get_tool(files=files, sample_code=sample_code, ask_fn=ask_fn)
    language = tool.language
    def_patterns = tool.def_patterns

    triples = []

    for fpath in files:
        code = file_contents.get(fpath, "")
        fname = fpath.split("/")[-1]

        # Extract import LINES (raw, for summary) using cached tool
        import_lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            if tool.import_line_checker(stripped):
                import_lines.append(stripped)

        # Extract definitions
        defs = []
        for line in code.split("\n"):
            for pattern in def_patterns:
                m = re.match(pattern, line.strip())
                if m:
                    defs.append(m.group(1))
                    break

        # Build compact summary — include package path for duplicate disambiguation
        parts = []
        # For duplicate basenames (e.g. __init__.py), add the directory path
        dirname = "/".join(fpath.split("/")[:-1]) if "/" in fpath else ""
        if dirname:
            parts.append(f"path: {dirname}/")
        if import_lines:
            parts.append("imports: " + "; ".join(import_lines[:15]))
        if defs:
            parts.append("defines: " + ", ".join(defs[:15]))

        summary = "\n".join(parts) if parts else "empty file"

        # Store summary triple
        triples.append(Triple(fpath, "file_summary", fpath, summary))

        # Store language metadata
        triples.append(Triple(fpath, "language", language, fname))

    # Also store inter-file dependency edges
    dep_triples = _extract_dependency_edges(files, file_contents, language)
    triples.extend(dep_triples)

    return triples


def _get_def_patterns(language: str) -> list[str]:
    """Get regex patterns for extracting definitions per language."""
    patterns = {
        "python": [r'^\s*(?:def|class)\s+(\w+)'],
        "java": [
            r'(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface|enum)\s+(\w+)',
            r'(?:public|private|protected)\s+[\w<>\[\]]+\s+(\w+)\s*\(',
        ],
        "javascript": [r'(?:export\s+)?(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)'],
        "typescript": [r'(?:export\s+)?(?:default\s+)?(?:function|class|const|let|var|interface|type)\s+(\w+)'],
        "c": [r'(?:[\w*]+\s+)+(\w+)\s*\(', r'typedef\s+\w+\s+(\w+)'],
        "cpp": [r'(?:[\w:*&]+\s+)+(\w+)\s*\(', r'class\s+(\w+)'],
        "csharp": [r'(?:public|private|internal)?\s*(?:static\s+)?(?:class|struct|interface)\s+(\w+)'],
        "php": [
            r'(?:public|private|protected)?\s*(?:static\s+)?function\s+(\w+)',
            r'class\s+(\w+)',
        ],
    }
    # Default: try common patterns across languages
    return patterns.get(language, [
        r'^\s*(?:def|function|class|struct|interface|type|enum)\s+(\w+)',
        r'(?:public|private|protected)\s+[\w<>\[\]]+\s+(\w+)\s*\(',
    ])


def _extract_dependency_edges(
    files: list[str],
    file_contents: dict[str, str],
    language: str = "python",
) -> list[Triple]:
    """Extract inter-file import edges (for graph queries).

    Uses lang_extraction module map builder for any language.
    """
    from pixelmem.v3.lang_extraction import _build_module_map, get_extractor

    module_to_file = _build_module_map(files, language)

    triples = []
    seen = set()

    for fpath, code in file_contents.items():
        for line in code.split("\n"):
            line = line.strip()

            # from X import Y
            m = re.match(r'^from\s+(\.{0,3}[\w.]*)\s+import\s+([\w, *]+)', line)
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
                        target = module_to_file.get(c)
                        if target and target != fpath:
                            edge = (fpath, target)
                            if edge not in seen:
                                seen.add(edge)
                                triples.append(Triple(fpath, "depends_on", target, line))
                            break
                    else:
                        continue
                    break
                continue

            # import X, Y, Z
            m = re.match(r'^import\s+([\w., ]+)', line)
            if m:
                for mod in m.group(1).split(","):
                    mod = mod.strip()
                    if mod and re.match(r'^[\w.]+$', mod):
                        target = module_to_file.get(mod)
                        if target and target != fpath:
                            edge = (fpath, target)
                            if edge not in seen:
                                seen.add(edge)
                                triples.append(Triple(fpath, "depends_on", target, line))

    return triples


def _topo_sort_from_pixelmem(
    files: list[str],
    mgr: ShardManager,
) -> list[str]:
    """Topologically sort files using depends_on edges from pixel matrix.

    Returns ordered list of basenames (base files first, dependent last).
    Returns partial list if not all files can be ordered.
    Returns empty list if no dependency edges found.
    """
    from collections import defaultdict, deque

    file_set = set(f.strip().lower() for f in files)
    basename_to_full = {}
    for f in files:
        bn = f.split("/")[-1]
        basename_to_full[f.strip().lower()] = bn

    # Collect depends_on edges from pixel matrix
    graph: dict[str, set[str]] = defaultdict(set)  # file → set of files it depends on
    in_degree: dict[str, int] = {basename_to_full.get(f.strip().lower(), f.split("/")[-1]): 0 for f in files}
    all_nodes = set(in_degree.keys())

    for shard in mgr.shards:
        for i in range(shard.n):
            subj = shard.idx_to_entity.get(i, "")
            if subj not in file_set:
                continue
            for j in range(shard.n):
                rgb = tuple(int(x) for x in shard.relation[i, j])
                if rgb == (0, 0, 0):
                    continue
                rel = shard.color_to_relation.get(rgb, "?")
                if rel != "depends_on":
                    continue
                obj = shard.idx_to_entity.get(j, "")
                if obj not in file_set:
                    continue
                src_bn = basename_to_full.get(subj, subj.split("/")[-1])
                tgt_bn = basename_to_full.get(obj, obj.split("/")[-1])
                if src_bn != tgt_bn:
                    graph[src_bn].add(tgt_bn)

    # Compute in-degrees
    for node, deps in graph.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[node] = in_degree.get(node, 0)  # ensure exists

    # Recalculate in-degree from graph
    for node in all_nodes:
        in_degree[node] = 0
    for node, deps in graph.items():
        for dep in deps:
            if node in in_degree:
                in_degree[node] += 1  # node depends on dep → node has higher in-degree

    # Wait — in-degree should count how many things depend on this node
    # Actually for topo sort: in_degree[X] = number of things X depends on
    # We want: nodes with 0 dependencies come first
    # graph[X] = set of nodes X depends on
    # So in_degree[X] = len(graph[X])

    in_deg = {}
    reverse_graph: dict[str, set[str]] = defaultdict(set)  # dep → set of dependents
    for node in all_nodes:
        in_deg[node] = len(graph.get(node, set()))
    for node, deps in graph.items():
        for dep in deps:
            reverse_graph[dep].add(node)

    # Kahn's algorithm
    queue = deque(n for n in all_nodes if in_deg[n] == 0)
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for dependent in reverse_graph.get(node, set()):
            in_deg[dependent] -= 1
            if in_deg[dependent] == 0:
                queue.append(dependent)

    # If not all nodes sorted, return partial
    if len(result) < len(all_nodes):
        # Add remaining unsorted nodes at the end
        remaining = [n for n in all_nodes if n not in result]
        result.extend(remaining)

    return result


def read_file_summaries_from_pixelmem(
    files: list[str],
    mgr: ShardManager,
) -> dict[str, dict]:
    """Read per-file summaries back from PixelMem.

    Looks for file_summary triples and parses the condition field
    back into structured summaries.
    """
    summaries = {}

    for fpath in files:
        canonical = fpath.strip().lower()
        fname = fpath.split("/")[-1]
        imports = []
        defs = []
        deps = []

        for shard in mgr.shards:
            if canonical not in shard.entity_to_idx:
                continue
            idx = shard.entity_to_idx[canonical]

            # Scan row for this file
            for j in range(shard.n):
                rgb = tuple(int(x) for x in shard.relation[idx, j])
                if rgb == (0, 0, 0):
                    continue
                rel = shard.color_to_relation.get(rgb, "?")
                crgb = tuple(int(x) for x in shard.condition[idx, j])
                cond = shard.color_to_condition.get(crgb, "")

                if rel == "file_summary":
                    # Parse the summary condition back into imports + defs
                    for line in cond.split("\n"):
                        if line.startswith("imports: "):
                            imports.extend(line[9:].split("; "))
                        elif line.startswith("defines: "):
                            defs.extend(line[9:].split(", "))

                elif rel == "depends_on":
                    obj = shard.idx_to_entity.get(j, "")
                    target_bn = obj.split("/")[-1] if "/" in obj else obj
                    deps.append(target_bn)

        summaries[fpath] = {
            "filename": fname,
            "imports": imports,
            "definitions": defs,
            "depends_on": deps,
            "source": "pixelmem",
        }

    return summaries


def index_and_query(
    files: list[str],
    file_contents: dict[str, str],
    query: str,
    ask_fn,
    shard_size: int = 128,
) -> tuple[str, dict]:
    """Complete pipeline: extract → store in PixelMem → retrieve → LLM answer.

    Returns (answer, stats).
    """
    import tempfile, shutil

    # Step 1: Build tools + extract + store
    triples = build_file_summary_triples(files, file_contents, ask_fn=ask_fn)

    # Step 1b: Add second-layer context triples for disambiguation
    from pixelmem.v3.dual_condition import build_context_triples
    context_triples = build_context_triples(files, file_contents)
    triples.extend(context_triples)

    # Step 2: Store in PixelMem
    td = tempfile.mkdtemp()
    try:
        mgr = ShardManager(td, shard_size=shard_size)
        for i in range(0, len(triples), 10):
            mgr.encode("", triples=triples[i:i+10])
        mgr.save()
        mgr._rebuild_entity_index()

        # Step 3: Read summaries from PixelMem
        summaries = read_file_summaries_from_pixelmem(files, mgr)

        # Step 4: Bayesian graph ordering (probabilistic, handles ambiguity)
        # Pass file_contents for indirect evidence scanning
        from pixelmem.v3.bayesian_order import order_files
        topo_order = order_files(files, mgr, file_contents=file_contents)

        # Step 5: Check for duplicate basenames
        basenames = [f.split("/")[-1] for f in files]
        has_duplicates = len(set(basenames)) < len(basenames)

        # Step 6: Format overview — use dual condition for dup disambiguation
        from pixelmem.v3.dual_condition import format_with_context
        overview = format_with_context(files, summaries, mgr)

        # File list: use parent-disambiguated names for duplicates
        basenames = [f.split("/")[-1] for f in files]
        has_duplicates = len(set(basenames)) < len(basenames)
        if has_duplicates:
            # Build disambiguated names from second condition layer
            from pixelmem.v3.dual_condition import get_file_context
            name_parts = []
            for f in files:
                bn = f.split("/")[-1]
                if basenames.count(bn) > 1:
                    ctx = get_file_context(f, mgr)
                    parent = ctx.get("parent", "")
                    name_parts.append(f"{bn} (in {parent}/)" if parent else f)
                else:
                    name_parts.append(bn)
            file_list = ", ".join(name_parts)
            path_note = ""
        else:
            file_list = ", ".join(f.split("/")[-1] for f in files)
            path_note = ""

        # Step 7: LLM answers — always consult LLM but give topo sort as suggestion
        if topo_order and len(topo_order) == len(files):
            # Partial confidence — LLM verifies
            topo_list = json.dumps(topo_order)
            prompt = (
                f"Based on these file dependencies:\n\n{overview}\n\n"
                f"Files: {file_list}{path_note}\n"
                f"Computed dependency order (base first, dependent last): {topo_list}\n\n"
                f"Is this correct? If yes, return it. If not, fix and return the corrected order.\n"
                f"Return ONLY a JSON array of filenames, preserving exact case."
            )
        elif topo_order:
            topo_list = json.dumps(topo_order)
            prompt = (
                f"Based on these file dependencies:\n\n{overview}\n\n"
                f"Files: {file_list}{path_note}\n"
                f"Partial dependency order so far: {topo_list}\n"
                f"Complete the ordering (base first, dependent last).\n"
                f"Return ONLY a JSON array of ALL filenames, preserving exact case."
            )
        else:
            prompt = (
                f"Analyze these files and order by dependency "
                f"(base files first, files that depend on others last).\n\n"
                f"{overview}\n\n"
                f"Files: {file_list}{path_note}\n"
                f"Return ONLY a JSON array of filenames, preserving exact case."
            )
        answer = ask_fn(prompt)
        if isinstance(answer, tuple):
            answer, in_tok, out_tok = answer
        else:
            in_tok = out_tok = 0

        stats = {
            "n_triples_stored": len(triples),
            "n_shards": len(mgr.shards),
            "overview_tokens": len(overview) // 4,
            "llm_in_tokens": in_tok,
            "llm_out_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        }

        return answer, stats

    finally:
        shutil.rmtree(td, ignore_errors=True)
