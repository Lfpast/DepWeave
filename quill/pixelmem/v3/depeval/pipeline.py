"""DependEval v3 pipeline -- end-to-end file dependency ordering.

Drop-in replacement for ``file_summary_store.index_and_query``.  Wires
together every depeval module and routes data through PixelMem storage:

    resolve imports  -->  encode triples in pixel matrix
                          -->  read back from pixel matrix
                               -->  build evidence graph
                                    -->  partial order
                                         -->  pairwise ranking
                                              -->  decode order
                                                   -->  LLM verify
                                                        -->  result

The round-trip through PixelMem (encode then decode) is the core of the
experiment: it proves the pixel matrix is a lossless (or near-lossless)
transport for structured dependency data.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from typing import Callable, Optional

from pixelmem.shard_manager import ShardManager
from pixelmem.triple_extractor import Triple

from .file_id_mapper import FileIDMapper
from .partial_order import PartialOrder, STRONG_THRESHOLD, AMBIGUOUS_CEILING
from .pairwise_ranker import rank_ambiguous_pairs
from .analyzer import PipelineResult, analyze_result


# ---------------------------------------------------------------------------
# Import resolution (inline -- lightweight, no separate module needed yet)
# ---------------------------------------------------------------------------


def _extract_imports_and_defs(code: str) -> tuple[list[str], list[str]]:
    """Extract raw import lines and definition names from source code.

    Returns:
        (import_lines, definition_names)
    """
    imports: list[str] = []
    defs: list[str] = []
    for line in code.split("\n"):
        stripped = line.strip()
        if re.match(r"^\s*(?:from|import)\s", stripped):
            imports.append(stripped)
        m = re.match(r"^\s*(?:def|class)\s+(\w+)", stripped)
        if m:
            defs.append(m.group(1))
    return imports, defs


def _build_module_map(files: list[str]) -> dict[str, str]:
    """Map module names / basenames to full file paths.

    Handles Python conventions: ``foo.bar.baz`` from path components,
    plus plain basename (with and without extension).
    """
    mod_map: dict[str, str] = {}
    for fpath in files:
        basename = fpath.rsplit("/", 1)[-1] if "/" in fpath else fpath
        name_no_ext = basename.rsplit(".", 1)[0]

        mod_map[basename] = fpath
        mod_map[name_no_ext] = fpath

        # Python module path: a/b/c.py -> a.b.c, b.c, c
        full_mod = fpath.replace("/", ".").replace(".py", "")
        parts = full_mod.split(".")
        for i in range(len(parts)):
            mod_map[".".join(parts[i:])] = fpath
    return mod_map


def _resolve_import_targets(
    fpath: str,
    code: str,
    module_map: dict[str, str],
) -> list[tuple[str, str, float]]:
    """Resolve import lines to target files within the project.

    Returns:
        List of ``(target_path, raw_import_line, confidence)`` triples.
        Confidence: 1.0 for relative imports, 0.85 for resolved absolute,
        0.5 for name-only matches.
    """
    results: list[tuple[str, str, float]] = []
    seen_targets: set[str] = set()

    for line in code.split("\n"):
        stripped = line.strip()

        # from X import Y  /  from .X import Y
        m = re.match(r"^from\s+(\.{0,3}[\w.]*)\s+import\s+([\w, *]+)", stripped)
        if m:
            from_mod = m.group(1).lstrip(".")
            is_relative = m.group(1).startswith(".")
            names = [n.strip() for n in m.group(2).split(",")]
            conf = 1.0 if is_relative else 0.85

            for name in names:
                name = name.strip()
                if name == "*":
                    candidates = [from_mod]
                else:
                    candidates = [f"{from_mod}.{name}", from_mod, name]
                for c in candidates:
                    target = module_map.get(c)
                    if target and target != fpath and target not in seen_targets:
                        seen_targets.add(target)
                        results.append((target, stripped, conf))
                        break
            continue

        # import X, Y, Z
        m = re.match(r"^import\s+([\w., ]+)", stripped)
        if m:
            for mod in m.group(1).split(","):
                mod = mod.strip()
                if mod and re.match(r"^[\w.]+$", mod):
                    target = module_map.get(mod)
                    if target and target != fpath and target not in seen_targets:
                        seen_targets.add(target)
                        results.append((target, stripped, 0.85))

    return results


# ---------------------------------------------------------------------------
# Evidence graph construction (reads from PixelMem)
# ---------------------------------------------------------------------------


def _build_evidence_from_pixelmem(
    files: list[str],
    mapper: FileIDMapper,
    mgr: ShardManager,
    file_contents: dict[str, str],
) -> dict[str, object]:
    """Read dependency triples back from the pixel matrix and build edge stats.

    This is the PixelMem read-back step: we encoded triples earlier,
    now we decode them to verify the round-trip and build confidence scores.

    Returns:
        Dict with ``constraints`` (list of (before_id, after_id, conf))
        and ``n_candidates`` (total resolved candidates).
    """
    file_set = {f.strip().lower() for f in files}
    constraints: list[tuple[str, str, float]] = []
    n_candidates = 0

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
                obj = shard.idx_to_entity.get(j, "")

                if rel == "depends_on" and obj in file_set and obj != subj:
                    n_candidates += 1
                    # source depends on target => target comes BEFORE source
                    source_path = _find_path(subj, files)
                    target_path = _find_path(obj, files)
                    if source_path and target_path:
                        src_id = mapper.id_for(source_path)
                        tgt_id = mapper.id_for(target_path)

                        # Read condition for confidence calibration
                        crgb = tuple(int(x) for x in shard.condition[i, j])
                        cond = shard.color_to_condition.get(crgb, "")
                        if cond.startswith("from .") or cond.startswith("from .."):
                            conf = 1.0
                        elif "import" in cond:
                            conf = 0.85
                        else:
                            conf = 0.7

                        # target before source (target is the dependency)
                        constraints.append((tgt_id, src_id, conf))

    # Add heuristic edges from naming patterns
    heuristic_constraints = _infer_heuristic_edges(files, mapper, file_contents)
    constraints.extend(heuristic_constraints)

    return {
        "constraints": constraints,
        "n_candidates": n_candidates,
    }


def _find_path(canonical: str, files: list[str]) -> Optional[str]:
    """Find the original file path matching a canonicalised entity name."""
    for f in files:
        if f.strip().lower() == canonical:
            return f
    return None


def _infer_heuristic_edges(
    files: list[str],
    mapper: FileIDMapper,
    file_contents: dict[str, str],
) -> list[tuple[str, str, float]]:
    """Infer ordering hints from naming conventions and code patterns.

    - ``test_*`` files depend on the module they test.
    - ``__init__.py`` typically aggregates (goes late).
    - ``conftest.py`` is a test fixture (goes late).
    - Files with no imports and many definitions are likely base files.
    """
    constraints: list[tuple[str, str, float]] = []
    ids = mapper.all_ids()
    bn_to_ids: dict[str, list[str]] = {}
    for fid in ids:
        bn = mapper.basename_for(fid)
        bn_to_ids.setdefault(bn, []).append(fid)

    for fid in ids:
        bn = mapper.basename_for(fid)
        path = mapper.path_for(fid)

        # test_ files go after the module they test
        if bn.startswith("test_"):
            tested = bn[5:]  # test_foo.py -> foo.py
            for candidate_id in bn_to_ids.get(tested, []):
                constraints.append((candidate_id, fid, 0.5))

        # __init__.py is an aggregator -- weak "goes late" signal
        if bn == "__init__.py":
            for other_id in ids:
                if other_id == fid:
                    continue
                other_bn = mapper.basename_for(other_id)
                if other_bn.startswith("test_") or other_bn == "__init__.py":
                    continue
                constraints.append((other_id, fid, 0.15))

        # conftest.py goes near the end
        if bn == "conftest.py":
            for other_id in ids:
                if other_id == fid:
                    continue
                other_bn = mapper.basename_for(other_id)
                if other_bn.startswith("test_"):
                    continue
                constraints.append((other_id, fid, 0.25))

    return constraints


# ---------------------------------------------------------------------------
# Triple construction for PixelMem storage
# ---------------------------------------------------------------------------


def _build_triples(
    files: list[str],
    file_contents: dict[str, str],
    module_map: dict[str, str],
) -> list[Triple]:
    """Build PixelMem triples: file summaries + depends_on edges.

    Two kinds of triples are stored:
      1. ``(path, file_summary, path, condition)`` where condition holds
         the raw imports + definitions.
      2. ``(source, depends_on, target, raw_import_line)`` for every
         resolved import edge.

    Both go into the pixel matrix and are read back by the evidence builder.
    """
    triples: list[Triple] = []

    for fpath in files:
        code = file_contents.get(fpath, "")
        imports, defs = _extract_imports_and_defs(code)

        # File summary triple
        parts: list[str] = []
        dirname = "/".join(fpath.split("/")[:-1]) if "/" in fpath else ""
        if dirname:
            parts.append(f"path: {dirname}/")
        if imports:
            parts.append("imports: " + "; ".join(imports[:15]))
        if defs:
            parts.append("defines: " + ", ".join(defs[:15]))
        summary = "\n".join(parts) if parts else "empty file"
        triples.append(Triple(fpath, "file_summary", fpath, summary))

        # Dependency edge triples
        resolved = _resolve_import_targets(fpath, code, module_map)
        for target, raw_line, _conf in resolved:
            triples.append(Triple(fpath, "depends_on", target, raw_line))

    return triples


# ---------------------------------------------------------------------------
# Order decoding with pairwise integration
# ---------------------------------------------------------------------------


def _decode_order(
    partial: PartialOrder,
    pairwise_scores: dict[tuple[str, str], float],
    mapper: FileIDMapper,
) -> list[str]:
    """Decode a total order from the partial order + pairwise scores.

    Strategy:
      1. Get topological layers from strong constraints.
      2. Within each layer, sort by pairwise scores (if available),
         then by out-degree heuristic.
      3. Flatten layers into a single list.
    """
    layers = partial.topological_layers()

    def _intra_layer_key(fid: str) -> tuple[float, str]:
        """Sort key within a layer: lower score = comes first."""
        # Sum of pairwise scores: if fid is "before" in many pairs, it
        # gets a lower (more negative) total, so it comes earlier.
        score = 0.0
        for (fi, fj), prob in pairwise_scores.items():
            if fi == fid:
                score -= prob       # prob > 0.5 means fi before fj
            elif fj == fid:
                score -= (1 - prob)  # 1-prob means fj before fi
        # Tie-break: test files and __init__ go later
        bn = mapper.basename_for(fid)
        if bn.startswith("test_"):
            score += 100
        if bn == "__init__.py":
            score += 50
        if bn == "conftest.py":
            score += 80
        return (score, fid)

    result: list[str] = []
    for layer in layers:
        sorted_layer = sorted(layer, key=_intra_layer_key)
        result.extend(sorted_layer)

    return result


# ---------------------------------------------------------------------------
# Hard constraint enforcement
# ---------------------------------------------------------------------------


def _enforce_hard_constraints(
    order: list[str],
    partial: PartialOrder,
) -> list[str]:
    """Post-process: fix any strong-constraint violations by local swaps.

    Iterates until no more strong violations exist (or a max iteration
    limit is reached to handle degenerate cases).
    """
    result = list(order)
    strong = partial.get_strong_constraints()
    if not strong:
        return result

    for _ in range(len(result) * 2):
        pos = {fid: i for i, fid in enumerate(result)}
        swapped = False
        for before, after in strong:
            if before in pos and after in pos and pos[before] > pos[after]:
                # Swap them
                i, j = pos[before], pos[after]
                result[i], result[j] = result[j], result[i]
                swapped = True
                break  # restart scan after swap
        if not swapped:
            break

    return result


# ---------------------------------------------------------------------------
# LLM verification prompt
# ---------------------------------------------------------------------------


def _build_verify_prompt(
    mapper: FileIDMapper,
    computed_order: list[str],
    file_contents: dict[str, str],
) -> str:
    """Build the final LLM verification prompt.

    Shows the file table (ID -> basename), compact summaries, and the
    computed order, then asks the LLM to verify or correct.
    """
    file_table = mapper.format_file_table()

    # Compact per-file summaries using IDs
    summaries: list[str] = []
    for fid in mapper.all_ids():
        path = mapper.path_for(fid)
        code = file_contents.get(path, "")
        imports, defs = _extract_imports_and_defs(code)
        parts = [f"{fid}:"]
        if imports:
            parts.append(f"  imports: {'; '.join(imports[:8])}")
        if defs:
            parts.append(f"  defines: {', '.join(defs[:8])}")
        summaries.append("\n".join(parts))

    summary_block = "\n".join(summaries)
    order_json = json.dumps(computed_order)

    return (
        f"File table:\n{file_table}\n\n"
        f"File summaries:\n{summary_block}\n\n"
        f"Computed dependency order (base first, dependent last): {order_json}\n\n"
        "Verify this ordering is correct. If wrong, fix it.\n"
        "Return ONLY a JSON array of file IDs (e.g. [\"F0\", \"F2\", \"F1\"])."
    )


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_dependeval(
    files: list[str],
    file_contents: dict[str, str],
    ask_fn: Callable[[str], str],
    shard_size: int = 128,
    enable_pairwise: bool = True,
) -> tuple[str, PipelineResult]:
    """Run the full DependEval v3 pipeline.

    Drop-in replacement for ``file_summary_store.index_and_query``.

    Pipeline stages:
      1. Build FileIDMapper for stable short IDs.
      2. Resolve imports and build module map.
      3. Encode file_summary + depends_on triples into PixelMem.
      4. Read back from pixel matrix to build evidence graph.
      5. Construct PartialOrder from evidence.
      6. (Optional) Rank ambiguous pairs with pairwise LLM calls.
      7. Decode total order from partial order + pairwise scores.
      8. Enforce hard constraints.
      9. LLM verification/correction.
      10. Parse response, enforce constraints on final answer.
      11. Convert to basenames for evaluation.

    Args:
        files: List of full file paths.
        file_contents: ``{path: source_code}`` dict.
        ask_fn: ``fn(prompt) -> response_text`` (may return tuple with tokens).
        shard_size: PixelMem shard matrix dimension.
        enable_pairwise: Whether to use pairwise LLM ranking for ambiguous pairs.

    Returns:
        ``(json_answer, PipelineResult)`` where json_answer is a JSON array
        of basenames (matching the format expected by the evaluator).
    """
    # -- Stage 1: FileIDMapper -----------------------------------------------
    mapper = FileIDMapper(files)

    # -- Stage 2: Module map + import resolution -----------------------------
    module_map = _build_module_map(files)

    # -- Stage 3: Build triples and store in PixelMem ------------------------
    triples = _build_triples(files, file_contents, module_map)

    # Also add dual-condition context triples for disambiguation
    from pixelmem.v3.dual_condition import build_context_triples
    context_triples = build_context_triples(files, file_contents)
    triples.extend(context_triples)

    td = tempfile.mkdtemp(prefix="depeval_")
    tokens_used = 0
    try:
        mgr = ShardManager(td, shard_size=shard_size)
        # Encode in batches (like file_summary_store)
        for i in range(0, len(triples), 10):
            mgr.encode("", triples=triples[i : i + 10])
        mgr.save()
        mgr._rebuild_entity_index()

        # -- Stage 4: Read back from pixel matrix ----------------------------
        evidence = _build_evidence_from_pixelmem(
            files, mapper, mgr, file_contents
        )
        constraints = evidence["constraints"]
        n_candidates = evidence["n_candidates"]

        # -- Stage 5: Build PartialOrder -------------------------------------
        partial = PartialOrder(mapper.all_ids())
        for before_id, after_id, conf in constraints:
            partial.add_constraint(before_id, after_id, conf)

        # -- Stage 6: Local uncertainty refinement ----------------------------
        # Principled policy: refine ONLY the unresolved frontier, not all pairs.
        #
        # Step 1: Identify unresolved groups (sets of files with no strong
        #         ordering between them) from the topological layers.
        # Step 2: If any unresolved group has exactly 2 files, one pairwise
        #         LLM call can resolve it cheaply.
        # Step 3: Groups of 3+ files are better handled by the global
        #         constrained decoder (no pairwise — too many conflicts).
        #
        # This is driven by GRAPH STRUCTURE, not file count.
        # A 10-file case with 9 strong edges has 0 unresolved pairs → no LLM.
        # A 3-file case with 0 edges has 3 ambiguous pairs → still no pairwise
        #   (because the group size is 3, not 2).

        ambiguous_pairs = partial.get_ambiguous_pairs()
        pairwise_scores: dict[tuple[str, str], float] = {}
        n_pairwise = 0

        if enable_pairwise and ambiguous_pairs:
            # Find unresolved groups from topological layers
            layers = partial.topological_layers()
            refinement_pairs: list[tuple[str, str]] = []

            ambiguous_set = set()
            for a, b in ambiguous_pairs:
                ambiguous_set.add((a, b))
                ambiguous_set.add((b, a))

            for layer in layers:
                if len(layer) == 2:
                    # Exactly 2 files tied — one pairwise call resolves it
                    a, b = layer[0], layer[1]
                    if (a, b) in ambiguous_set:
                        refinement_pairs.append((a, b))
                # Groups of 1: no refinement needed
                # Groups of 3+: global decoder handles (pairwise would conflict)

            if refinement_pairs:
                pairwise_scores = rank_ambiguous_pairs(
                    refinement_pairs,
                    mapper,
                    file_contents,
                    ask_fn,
                    max_pairs=len(refinement_pairs),  # bounded by graph structure
                )
                n_pairwise = len(pairwise_scores)

                # Integrate pairwise results as medium-confidence constraints
                for (fi, fj), prob in pairwise_scores.items():
                    if prob > 0.6:
                        partial.add_constraint(fi, fj, min(prob, 0.65))
                    elif prob < 0.4:
                        partial.add_constraint(fj, fi, min(1.0 - prob, 0.65))

        # -- Stage 7: Decode total order -------------------------------------
        order_ids = _decode_order(partial, pairwise_scores, mapper)

        # -- Stage 8: Enforce hard constraints -------------------------------
        order_ids = _enforce_hard_constraints(order_ids, partial)

        # -- Stage 9: LLM verification --------------------------------------
        verify_prompt = _build_verify_prompt(mapper, order_ids, file_contents)
        raw_answer = ask_fn(verify_prompt)
        if isinstance(raw_answer, tuple):
            answer_text, in_tok, out_tok = raw_answer
            tokens_used = in_tok + out_tok
        else:
            answer_text = raw_answer

        # -- Stage 10: Parse LLM response ------------------------------------
        parsed_ids = mapper.parse_id_response(answer_text)

        # Validate: must have all IDs exactly once
        all_ids_set = set(mapper.all_ids())
        if set(parsed_ids) == all_ids_set and len(parsed_ids) == len(all_ids_set):
            final_ids = parsed_ids
        else:
            # LLM response was malformed -- fall back to computed order
            final_ids = order_ids

        # Enforce hard constraints on the final answer too
        final_ids = _enforce_hard_constraints(final_ids, partial)

        # -- Stage 11: Convert to basenames ----------------------------------
        final_basenames = mapper.ids_to_basenames(final_ids)

        # Build the JSON answer (array of basenames)
        json_answer = json.dumps(final_basenames)

        # -- Build PipelineResult --------------------------------------------
        # Pipeline does NOT have ground truth. Return a partial result.
        # The caller (experiment script) must call analyze_result() with
        # the real expected_bn to get proper error classification.
        result = PipelineResult(
            file_ids=final_ids,
            basenames=final_basenames,
            expected=[],  # caller fills this in
            exact=False,  # caller evaluates
            n_files=len(files),
            n_candidates=n_candidates,
            n_strong_edges=len(partial.get_strong_constraints()),
            n_medium_edges=sum(1 for (_, _), c in partial._constraints.items() if 0.3 < c <= 0.7),
            n_weak_edges=sum(1 for (_, _), c in partial._constraints.items() if c <= 0.3),
            n_ambiguous_pairs=len(partial.get_ambiguous_pairs()),
            n_pairwise_queries=n_pairwise,
            has_duplicates=len(set(f.split("/")[-1] for f in files)) < len(files),
            has_cycle=partial.has_cycle(),
            n_constraint_violations=sum(1 for _, _, c in partial.check_violations(final_ids) if c > 0.7),
            tokens_used=tokens_used,
            error_type="pending",  # caller must classify
        )

        return json_answer, result

    finally:
        shutil.rmtree(td, ignore_errors=True)
