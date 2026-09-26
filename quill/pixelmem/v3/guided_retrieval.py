"""LLM-Guided Retrieval: LLM plans, tools execute, LLM answers.

The LLM doesn't start from nothing — we give it:
  1. What entities/files exist (compact list)
  2. What tools are available (scan imports, scan definitions, etc.)
  3. Pre-computed suggestions based on query type

The LLM then either:
  A. Accepts our suggestion (fast path — 1 API call)
  B. Requests different/additional info (2 API calls)

Tools available to the LLM:
  - scan_imports(file): return all import lines for a file
  - scan_definitions(file): return all function/class names
  - scan_calls(file): return all function calls made
  - find_dependency(file_a, file_b): check if a depends on b
  - get_file_summary(file): imports + definitions + calls compact

For known query types (dependency ordering, structure lookup),
we pre-compute the optimal retrieval and skip Turn 1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class RetrievalRequest:
    """What the LLM asked for."""
    tool: str
    args: dict
    reason: str = ""


@dataclass
class RetrievalResult:
    """What the tool returned."""
    request: RetrievalRequest
    data: str
    token_cost: int = 0


@dataclass
class GuidedSession:
    """A guided retrieval session."""
    query: str
    files: list[str]
    available_info: dict[str, str]  # pre-computed per-file summaries
    requests: list[RetrievalRequest] = field(default_factory=list)
    results: list[RetrievalResult] = field(default_factory=list)
    answer: str = ""
    total_tokens: int = 0


def build_file_summaries(
    files: list[str],
    file_contents: dict[str, str],
) -> dict[str, dict]:
    """Pre-compute per-file summaries from raw code (deterministic, free).

    Use this when you have raw file contents (e.g., first-time indexing).
    """
    summaries = {}
    for fpath in files:
        code = file_contents.get(fpath, "")
        fname = fpath.split("/")[-1]

        imports = []
        defs = []
        calls = set()

        for line in code.split("\n"):
            stripped = line.strip()
            if re.match(r'^\s*(?:from|import)\s', stripped):
                imports.append(stripped)
            m = re.match(r'^\s*(?:def|class)\s+(\w+)', stripped)
            if m:
                defs.append(m.group(1))

        for m in re.finditer(r'\b(\w+)\s*\(', code):
            name = m.group(1)
            if name not in ('if', 'for', 'while', 'with', 'print', 'len', 'range',
                           'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
                           'isinstance', 'hasattr', 'getattr', 'setattr', 'type',
                           'super', 'open', 'sorted', 'enumerate', 'zip', 'map',
                           'filter', 'any', 'all', 'min', 'max', 'sum', 'abs'):
                calls.add(name)

        summaries[fpath] = {
            "filename": fname,
            "imports": imports[:15],
            "definitions": defs[:15],
            "calls": sorted(calls)[:20],
            "n_lines": len(code.split("\n")),
        }

    return summaries


def build_file_summaries_from_pixelmem(
    files: list[str],
    mgr: "ShardManager",
) -> dict[str, dict]:
    """Build per-file summaries by reading from PixelMem pixel matrices.

    This is the production path: data was already indexed into PixelMem,
    now we retrieve it via scan_entity (reads exact RGB from PNG matrix).

    No raw file contents needed — everything comes from the pixel store.
    """
    from pixelmem.memory import BLACK

    summaries = {}
    for fpath in files:
        fname = fpath.split("/")[-1]

        # EXACT scan — read directly from matrix by entity index
        # No fuzzy matching — we know the exact entity name
        facts = []
        for si, shard in enumerate(mgr.shards):
            canonical = fpath.strip().lower()
            if canonical not in shard.entity_to_idx:
                continue
            idx = shard.entity_to_idx[canonical]
            # Scan row (file as subject)
            for j in range(shard.n):
                rgb = tuple(int(x) for x in shard.relation[idx, j])
                if rgb == (0, 0, 0):
                    continue
                obj = shard.idx_to_entity.get(j, "")
                rel = shard.color_to_relation.get(rgb, "?")
                crgb = tuple(int(x) for x in shard.condition[idx, j])
                cond = shard.color_to_condition.get(crgb, "")
                from pixelmem.v3.retrieval_algebra import Fact
                facts.append(Fact(canonical, rel, obj, cond, si))
            # Scan column (file as object)
            for i in range(shard.n):
                if i == idx:
                    continue
                rgb = tuple(int(x) for x in shard.relation[i, idx])
                if rgb == (0, 0, 0):
                    continue
                subj = shard.idx_to_entity.get(i, "")
                rel = shard.color_to_relation.get(rgb, "?")
                crgb = tuple(int(x) for x in shard.condition[i, idx])
                cond = shard.color_to_condition.get(crgb, "")
                from pixelmem.v3.retrieval_algebra import Fact
                facts.append(Fact(subj, rel, canonical, cond, si))

        imports = []
        imports_external = []
        defs = []
        calls = []

        for f in facts:
            rel = f.relation
            is_subject = (f.subject.lower() == fpath.lower())

            # Import-related relations (internal)
            if rel in ("imports", "depends_on", "imports_from", "imports_module") and is_subject:
                if f.condition and ("import" in f.condition or "from" in f.condition):
                    imports.append(f.condition)  # original import line
                else:
                    target_name = f.object.split("/")[-1] if "/" in f.object else f.object
                    imports.append(f"imports {target_name}")

            # Symbol usage
            elif rel == "uses_symbol" and is_subject:
                source_mod = f.condition.replace("from ", "") if f.condition else ""
                imports.append(f"uses {f.object} (from {source_mod})" if source_mod else f"uses {f.object}")

            # External imports
            elif rel in ("imports_external",) and is_subject:
                if f.condition and ("import" in f.condition or "from" in f.condition):
                    imports_external.append(f.condition)
                else:
                    imports_external.append(f"imports {f.object} (external)")

            # Definitions
            elif rel in ("defines", "defines_function", "defines_class") and is_subject:
                defs.append(f.object)

            # Calls
            elif rel in ("calls",) and is_subject:
                target_name = f.object.split("/")[-1] if "/" in f.object else f.object
                calls.append(target_name)

            elif rel in ("contains_function", "contains_class", "contains_method") and is_subject:
                defs.append(f.object)

        # Combine internal + external imports for LLM context
        all_imports = imports + imports_external

        summaries[fpath] = {
            "filename": fname,
            "imports": all_imports[:15],
            "definitions": defs[:15],
            "calls": calls[:20],
            "source": "pixelmem",  # flag that this came from pixel store
        }

    return summaries


def format_overview(
    files: list[str],
    summaries: dict[str, dict],
    query_type: str = "general",
) -> str:
    """Format a compact overview for the LLM.

    For known query types, include the most relevant info.
    For unknown types, include everything compact.
    """
    lines = []

    for fpath in files:
        s = summaries.get(fpath, {})
        fname = s.get("filename", fpath.split("/")[-1])

        if query_type == "dependency_order":
            # For dependency: show imports and definitions
            lines.append(f"=== {fname} ===")
            if s.get("imports"):
                lines.append("  imports: " + "; ".join(s["imports"][:10]))
            if s.get("definitions"):
                lines.append("  defines: " + ", ".join(s["definitions"][:10]))

        elif query_type == "structure_lookup":
            # For structure: show definitions and line count
            lines.append(f"=== {fname} ({s.get('n_lines', '?')} lines) ===")
            if s.get("definitions"):
                lines.append("  defines: " + ", ".join(s["definitions"]))
            if s.get("imports"):
                lines.append("  imports: " + "; ".join(s["imports"][:5]))

        else:
            # General: everything compact
            lines.append(f"=== {fname} ===")
            if s.get("imports"):
                lines.append("  imports: " + "; ".join(s["imports"][:8]))
            if s.get("definitions"):
                lines.append("  defines: " + ", ".join(s["definitions"][:8]))
            if s.get("calls"):
                # Only show calls that reference other files
                other_names = set()
                for other in files:
                    if other != fpath:
                        other_s = summaries.get(other, {})
                        other_names.update(other_s.get("definitions", []))
                        other_names.add(other.split("/")[-1].replace(".py", ""))
                relevant_calls = [c for c in s["calls"] if c in other_names]
                if relevant_calls:
                    lines.append(f"  calls from other files: {', '.join(relevant_calls[:8])}")

    return "\n".join(lines)


def guided_query(
    query: str,
    files: list[str],
    ask_fn: Callable,
    file_contents: Optional[dict[str, str]] = None,
    mgr: Optional["ShardManager"] = None,
    query_type: str = "auto",
) -> GuidedSession:
    """Execute a guided retrieval session.

    Two modes:
      A. file_contents provided → build summaries from raw code (first-time)
      B. mgr (ShardManager) provided → read summaries from PixelMem pixel store

    For known query types, uses pre-computed optimal retrieval (1 API call).
    For unknown types, asks LLM what to retrieve first (2 API calls).
    """
    session = GuidedSession(query=query, files=files, available_info={})

    # Build summaries from either raw files or PixelMem
    if mgr is not None:
        summaries = build_file_summaries_from_pixelmem(files, mgr)
    elif file_contents is not None:
        summaries = build_file_summaries(files, file_contents)
    else:
        raise ValueError("Provide either file_contents or mgr (ShardManager)")
    session.available_info = {f: json.dumps(s, default=str)[:200] for f, s in summaries.items()}

    # Auto-detect query type
    if query_type == "auto":
        q_lower = query.lower()
        if any(w in q_lower for w in ["order", "dependency", "depends", "import order"]):
            query_type = "dependency_order"
        elif any(w in q_lower for w in ["contains", "defines", "what is in", "structure"]):
            query_type = "structure_lookup"
        else:
            query_type = "general"

    # Build overview for LLM
    overview = format_overview(files, summaries, query_type)
    file_list = ", ".join(f.split("/")[-1] for f in files)

    if query_type == "dependency_order":
        # FAST PATH: we know what the LLM needs — just give imports + ask
        prompt = (
            f"Analyze these Python files and order by dependency "
            f"(base files first, files that depend on others last).\n\n"
            f"{overview}\n\n"
            f"Files: {file_list}\n"
            f"Return ONLY a JSON array of filenames, preserving exact case."
        )
        answer = ask_fn(prompt)
        if isinstance(answer, tuple):
            answer, in_tok, out_tok = answer
            session.total_tokens = in_tok + out_tok
        else:
            session.total_tokens = len(prompt) // 4
        session.answer = answer

    elif query_type == "structure_lookup":
        # FAST PATH: give structure overview + ask
        prompt = (
            f"Using this file structure information, answer the question.\n\n"
            f"{overview}\n\n"
            f"Question: {query}\n"
            f"Answer concisely."
        )
        answer = ask_fn(prompt)
        if isinstance(answer, tuple):
            answer, in_tok, out_tok = answer
            session.total_tokens = in_tok + out_tok
        else:
            session.total_tokens = len(prompt) // 4
        session.answer = answer

    else:
        # GENERAL: 2-turn — ask LLM what it needs, then provide it
        # Turn 1: LLM sees file list + available tools
        turn1_prompt = (
            f"I have information about these files: {file_list}\n"
            f"Available info per file: imports, definitions (functions/classes), calls.\n\n"
            f"Question: {query}\n\n"
            f"What information do you need to answer this? "
            f"Reply with a JSON object: {{\"files_to_inspect\": [\"file1.py\"], \"info_needed\": [\"imports\", \"definitions\"]}}"
        )
        turn1_answer = ask_fn(turn1_prompt)
        if isinstance(turn1_answer, tuple):
            turn1_answer, t1_in, t1_out = turn1_answer
        else:
            t1_in = t1_out = 0

        # Parse what LLM wants
        files_to_inspect = files  # default: all
        info_types = ["imports", "definitions"]  # default
        try:
            m = re.search(r'\{.*\}', turn1_answer, re.DOTALL)
            if m:
                req = json.loads(m.group(0))
                if "files_to_inspect" in req:
                    requested = req["files_to_inspect"]
                    files_to_inspect = [f for f in files if f.split("/")[-1] in requested]
                    if not files_to_inspect:
                        files_to_inspect = files
                if "info_needed" in req:
                    info_types = req["info_needed"]
        except:
            pass

        # Turn 2: provide requested info + ask for answer
        info_lines = []
        for fpath in files_to_inspect:
            s = summaries.get(fpath, {})
            fname = s.get("filename", fpath.split("/")[-1])
            info_lines.append(f"=== {fname} ===")
            if "imports" in info_types and s.get("imports"):
                info_lines.append("  imports: " + "; ".join(s["imports"][:10]))
            if "definitions" in info_types and s.get("definitions"):
                info_lines.append("  defines: " + ", ".join(s["definitions"][:10]))
            if "calls" in info_types and s.get("calls"):
                info_lines.append("  calls: " + ", ".join(s["calls"][:10]))

        info_text = "\n".join(info_lines)
        turn2_prompt = (
            f"Here is the requested information:\n\n{info_text}\n\n"
            f"Question: {query}\n"
            f"Answer concisely."
        )
        answer = ask_fn(turn2_prompt)
        if isinstance(answer, tuple):
            answer, t2_in, t2_out = answer
            session.total_tokens = t1_in + t1_out + t2_in + t2_out
        else:
            session.total_tokens = (len(turn1_prompt) + len(turn2_prompt)) // 4
        session.answer = answer

    return session
