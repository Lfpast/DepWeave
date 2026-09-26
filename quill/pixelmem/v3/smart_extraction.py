"""Smart Extraction: import edges + function signatures + call sites.

Instead of 40K tokens (full code) or 330 tokens (import edges only),
extract ~2K tokens of structured code signals:

  1. Import statements (free, AST)
  2. Function/class definitions with signatures (AST, ~50 tokens/file)
  3. Call sites: which functions call which names (AST, ~100 tokens/file)
  4. Variable references to other modules (regex, ~30 tokens/file)

Then let LLM resolve the remaining edges from this compact summary.
Total: ~2K tokens (20x less than full code, 6x more than import-only).
"""

from __future__ import annotations

import ast
import re
import json
from typing import Optional

from pixelmem.triple_extractor import Triple


def extract_code_signals(file_contents: dict[str, str]) -> dict[str, dict]:
    """Extract structured code signals from each file via AST.

    Returns per-file:
      - imports: list of import lines
      - definitions: list of "def func(args)" / "class Name(bases)"
      - calls: list of function/method calls made
      - references: list of module-level name references (e.g. views.handler)
    """
    signals: dict[str, dict] = {}

    for fpath, code in file_contents.items():
        fname = fpath.split("/")[-1]
        sig = {
            "imports": [],
            "definitions": [],
            "calls": set(),
            "references": set(),
        }

        # Extract imports (always works even if AST fails)
        for line in code.split("\n"):
            line = line.strip()
            if re.match(r'^(?:from|import)\s', line):
                sig["imports"].append(line)

        # Try AST parsing
        try:
            tree = ast.parse(code, filename=fname)

            # Definitions
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    params = [a.arg for a in node.args.args if a.arg != "self"]
                    sig["definitions"].append(f"def {node.name}({', '.join(params[:5])})")
                elif isinstance(node, ast.ClassDef):
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            bases.append(f"{_get_name(base)}")
                    sig["definitions"].append(f"class {node.name}({', '.join(bases)})")
                    # Also get methods
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            params = [a.arg for a in item.args.args if a.arg != "self"]
                            sig["definitions"].append(f"  def {node.name}.{item.name}({', '.join(params[:3])})")

            # Calls and references
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = _get_name(node.func)
                    if name and len(name) > 1:
                        sig["calls"].add(name)
                elif isinstance(node, ast.Attribute):
                    name = _get_name(node)
                    if name and "." in name and len(name) > 3:
                        sig["references"].add(name)

        except (SyntaxError, ValueError):
            # Fallback: regex-based extraction
            for line in code.split("\n"):
                m = re.match(r'^\s*def\s+(\w+)\s*\(([^)]*)\)', line)
                if m:
                    sig["definitions"].append(f"def {m.group(1)}({m.group(2)[:30]})")
                m = re.match(r'^\s*class\s+(\w+)\s*(?:\(([^)]*)\))?', line)
                if m:
                    sig["definitions"].append(f"class {m.group(1)}({m.group(2) or ''})")

            # Regex for calls: word.word( or word(
            for m in re.finditer(r'\b([\w.]+)\s*\(', code):
                name = m.group(1)
                if len(name) > 2 and not name.startswith("__"):
                    sig["calls"].add(name)

        sig["calls"] = sorted(sig["calls"])[:30]
        sig["references"] = sorted(sig["references"])[:20]

        signals[fpath] = sig

    return signals


def _get_name(node) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        val = _get_name(node.value)
        if val:
            return f"{val}.{node.attr}"
        return node.attr
    return None


def signals_to_compact_text(
    signals: dict[str, dict],
    files: list[str],
) -> str:
    """Format code signals as compact text for LLM (~2K tokens total)."""
    lines = []
    for fpath in files:
        fname = fpath.split("/")[-1]
        sig = signals.get(fpath, {})
        if not sig:
            continue

        lines.append(f"=== {fname} ({fpath}) ===")

        # Imports (all)
        if sig.get("imports"):
            for imp in sig["imports"]:
                lines.append(f"  {imp}")

        # Definitions (all)
        if sig.get("definitions"):
            for d in sig["definitions"]:
                lines.append(f"  {d}")

        # Calls (top 15 — most important for cross-file deps)
        if sig.get("calls"):
            # Filter to calls that reference other files' names
            other_names = set()
            for other in files:
                if other != fpath:
                    on = other.split("/")[-1].replace(".py", "")
                    other_names.add(on)
                    # Also add function names from other files
                    other_sig = signals.get(other, {})
                    for d in other_sig.get("definitions", []):
                        m = re.match(r'(?:def|class)\s+([\w.]+)', d)
                        if m:
                            other_names.add(m.group(1).split(".")[-1])

            relevant_calls = [c for c in sig["calls"] if any(n in c for n in other_names)]
            if relevant_calls:
                lines.append(f"  # calls: {', '.join(relevant_calls[:10])}")

        # References
        if sig.get("references"):
            other_basenames = set(f.split("/")[-1].replace(".py", "") for f in files if f != fpath)
            relevant_refs = [r for r in sig["references"] if any(n in r for n in other_basenames)]
            if relevant_refs:
                lines.append(f"  # refs: {', '.join(relevant_refs[:10])}")

        lines.append("")

    return "\n".join(lines)


def inspect_extraction(
    files: list[str],
    file_contents: dict[str, str],
) -> dict:
    """Inspect what each extraction stage can and cannot resolve.

    Returns detailed breakdown of:
      - AST success/failure per file
      - Which imports resolved deterministically
      - Which imports remain unresolved and why
      - Code signals extracted per file
    """
    from pixelmem.v3.hybrid_extraction import resolve_deterministic, extract_all_imports

    report = {"files": {}, "summary": {}}

    # Per-file AST check
    for fpath, code in file_contents.items():
        fname = fpath.split("/")[-1]
        file_report = {"ast_ok": False, "n_imports": 0, "n_defs": 0, "n_calls": 0}

        try:
            tree = ast.parse(code, filename=fname)
            file_report["ast_ok"] = True
            file_report["n_defs"] = sum(1 for n in ast.iter_child_nodes(tree)
                                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
            file_report["n_calls"] = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call))
        except SyntaxError as e:
            file_report["ast_error"] = str(e)[:60]

        file_report["n_imports"] = sum(1 for l in code.split("\n") if re.match(r'^\s*(?:from|import)\s', l.strip()))
        report["files"][fpath] = file_report

    # Import resolution
    imports_by_file = extract_all_imports(file_contents)
    resolved, unresolved = resolve_deterministic(imports_by_file, files)

    report["summary"] = {
        "total_files": len(files),
        "ast_success": sum(1 for f in report["files"].values() if f["ast_ok"]),
        "total_imports": sum(len(v) for v in imports_by_file.values()),
        "resolved_deterministic": len(resolved),
        "unresolved": len(unresolved),
        "unresolved_lines": [(src.split("/")[-1], line) for src, line in unresolved[:10]],
        "edges_found": [(t.subject.split("/")[-1], t.object.split("/")[-1]) for t in resolved],
    }

    return report


def smart_extract(
    files: list[str],
    file_contents: dict[str, str],
    ask_fn=None,
    max_workers: int = 10,
    inspect: bool = False,
) -> tuple[list[Triple], dict]:
    """Full smart extraction pipeline with cascade fallback.

    Stage 1: AST import resolution (free, ~60% of edges)
    Stage 2: Regex call-site matching (free, catches runtime refs)
    Stage 3: Code signals + LLM (1 call, ~2K tokens, catches remaining)
    Stage 4: If still insufficient, LLM gets slightly more context

    Returns (triples, stats).
    """
    from pixelmem.v3.hybrid_extraction import resolve_deterministic, extract_all_imports

    # Stage 1: Deterministic import resolution
    imports_by_file = extract_all_imports(file_contents)
    det_triples, unresolved = resolve_deterministic(imports_by_file, files)

    # Stage 2: Regex call-site cross-reference (free)
    # Look for function/class names from other files being called
    regex_triples = _resolve_by_callsite(files, file_contents, det_triples)
    det_triples.extend(regex_triples)

    # Extract code signals for potential LLM use
    signals = extract_code_signals(file_contents)
    compact_text = signals_to_compact_text(signals, files)
    signal_tokens = len(compact_text) // 4

    stats = {
        "deterministic_edges": len(det_triples) - len(regex_triples),
        "regex_edges": len(regex_triples),
        "unresolved_imports": len(unresolved),
        "signal_tokens": signal_tokens,
        "llm_edges": 0,
        "total_edges": len(det_triples),
        "method": "deterministic+regex",
    }

    if inspect:
        stats["inspection"] = inspect_extraction(files, file_contents)

    n_needed = len(files) - 1
    have_enough = len(det_triples) >= n_needed

    if not ask_fn or have_enough:
        return det_triples, stats

    # Stage 3: LLM resolves from compact signals
    file_list = "\n".join(f"  {f}" for f in files)
    prompt = (
        f"Analyze these Python file summaries and determine ALL dependency edges.\n"
        f"A depends on B if A imports from B, calls functions defined in B, "
        f"or references objects from B.\n\n"
        f"{compact_text}\n"
        f"Files:\n{file_list}\n\n"
        f"Return a JSON array of [source, target] pairs where source depends on target.\n"
        f"Use FULL file paths. Only include edges between the listed files.\n"
        f'Example: [["path/a.py", "path/b.py"], ["path/c.py", "path/a.py"]]'
    )

    answer = ask_fn(prompt)
    if isinstance(answer, tuple):
        answer, in_tok, out_tok = answer
        stats["llm_input_tokens"] = in_tok
        stats["llm_output_tokens"] = out_tok
    else:
        in_tok = out_tok = 0

    llm_triples = _parse_llm_edges(answer, files, det_triples)
    all_triples = det_triples + llm_triples

    stats["llm_edges"] = len(llm_triples)
    stats["total_edges"] = len(all_triples)
    stats["method"] = "smart"

    # Stage 4: If STILL not enough edges, try with more context
    if len(all_triples) < n_needed and ask_fn:
        # Give LLM the first 100 lines of each file (not full code, but more than signals)
        extra_context = []
        for fpath in files:
            code = file_contents.get(fpath, "")
            lines = code.split("\n")[:80]
            extra_context.append(f"=== {fpath} (first {len(lines)} lines) ===\n" + "\n".join(lines))
        extra_text = "\n\n".join(extra_context)
        # Cap at 4K tokens
        extra_text = extra_text[:16000]

        prompt2 = (
            f"These Python files have dependencies between them. "
            f"Determine the dependency order.\n\n"
            f"{extra_text}\n\n"
            f"Return a JSON array of [source, target] pairs where source depends on target.\n"
            f"Use FULL file paths. Only edges between listed files.\n"
            f'Example: [["a.py", "b.py"]]'
        )
        answer2 = ask_fn(prompt2)
        if isinstance(answer2, tuple):
            answer2, in2, out2 = answer2
            stats["llm_input_tokens"] = stats.get("llm_input_tokens", 0) + in2
            stats["llm_output_tokens"] = stats.get("llm_output_tokens", 0) + out2

        extra_triples = _parse_llm_edges(answer2, files, all_triples)
        all_triples.extend(extra_triples)
        stats["llm_edges"] += len(extra_triples)
        stats["total_edges"] = len(all_triples)
        stats["method"] = "smart+fallback"

    return all_triples, stats


def _resolve_by_callsite(
    files: list[str],
    file_contents: dict[str, str],
    existing_triples: list[Triple],
) -> list[Triple]:
    """Stage 2: Find cross-file dependencies via call-site matching.

    If file A calls a function defined in file B (even without importing),
    A depends on B. This catches runtime references like:
      urls.py: urlpatterns = [path('/api', views.handle_request)]
    """
    # Collect all defined names per file
    defined_names: dict[str, set[str]] = {}  # file → {name, name, ...}
    for fpath, code in file_contents.items():
        names = set()
        try:
            tree = ast.parse(code)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
        except SyntaxError:
            for m in re.finditer(r'(?:def|class)\s+(\w+)', code):
                names.add(m.group(1))
        defined_names[fpath] = names

    # Find cross-file references
    existing = {(t.subject, t.object) for t in existing_triples}
    triples = []

    for fpath, code in file_contents.items():
        fname_base = fpath.split("/")[-1].replace(".py", "")
        for other_fpath, other_names in defined_names.items():
            if other_fpath == fpath:
                continue
            other_base = other_fpath.split("/")[-1].replace(".py", "")

            # Check if this file references the other file's module name + defined names
            # e.g. "views.handle_request" in urls.py → urls depends on views
            for name in other_names:
                # Pattern: other_module.name (e.g. views.handle_request)
                pattern = f"{other_base}.{name}"
                if pattern in code:
                    edge = (fpath, other_fpath)
                    if edge not in existing:
                        existing.add(edge)
                        triples.append(Triple(
                            fpath, "imports", other_fpath,
                            f"{fpath.split('/')[-1]}→{other_fpath.split('/')[-1]} [callsite]",
                        ))
                    break

    return triples


def _parse_llm_edges(
    answer: str,
    files: list[str],
    existing_triples: list[Triple],
) -> list[Triple]:
    """Parse LLM response into validated Triple edges."""
    seen = {(t.subject, t.object) for t in existing_triples}
    triples = []
    try:
        m = re.search(r'\[.*\]', answer, re.DOTALL)
        if m:
            edges = json.loads(m.group(0))
            for edge in edges:
                if isinstance(edge, list) and len(edge) == 2:
                    src = edge[0].strip("'\" ")
                    tgt = edge[1].strip("'\" ")
                    src_match = _match_file(src, files)
                    tgt_match = _match_file(tgt, files)
                    if src_match and tgt_match and src_match != tgt_match:
                        if (src_match, tgt_match) not in seen:
                            seen.add((src_match, tgt_match))
                            triples.append(Triple(
                                src_match, "imports", tgt_match,
                                f"{src_match.split('/')[-1]}→{tgt_match.split('/')[-1]} [llm]",
                            ))
    except Exception:
        pass
    return triples


def _match_file(name: str, files: list[str]) -> Optional[str]:
    """Match a name (full path or basename) to a file in our list."""
    name = name.strip("'\" ")
    # Exact match
    if name in files:
        return name
    # Basename match
    bn = name.split("/")[-1]
    matches = [f for f in files if f.split("/")[-1] == bn]
    if len(matches) == 1:
        return matches[0]
    # Suffix match
    for f in files:
        if f.endswith(name) or name.endswith(f.split("/")[-1]):
            return f
    return None
