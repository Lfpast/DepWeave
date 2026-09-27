"""Pairwise LLM ranking for ambiguous file pairs.

When the evidence graph cannot determine ordering between two files
(no imports between them, no transitive path), we ask the LLM a
focused pairwise question: "Which file should come first?"

This is MUCH cheaper than asking the LLM about all N files at once.
Each pairwise query is a tiny prompt (~200 tokens) with just two files'
imports and definitions.  We run up to ``max_pairs`` queries in parallel
via ThreadPoolExecutor.

The LLM returns which file is the base dependency, and we convert that
to a probability P(F_i before F_j).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .file_id_mapper import FileIDMapper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rank_ambiguous_pairs(
    pairs: list[tuple[str, str]],
    mapper: FileIDMapper,
    file_contents: dict[str, str],
    ask_fn: Callable[[str], str],
    max_pairs: int = 5,
) -> dict[tuple[str, str], float]:
    """Ask the LLM to rank ambiguous file pairs in parallel.

    For each pair ``(Fi, Fj)`` in *pairs* (up to *max_pairs*), we build
    a compact prompt showing only those two files' imports and definitions,
    then ask the LLM which should come first.

    Args:
        pairs: List of ``(file_id_i, file_id_j)`` pairs to rank.
        mapper: The FileIDMapper for resolving IDs to paths/basenames.
        file_contents: ``{full_path: source_code}`` for building summaries.
        ask_fn: ``fn(prompt) -> response_text`` (or returns tuple with tokens).
        max_pairs: Maximum number of LLM calls (budget cap).

    Returns:
        ``{(Fi, Fj): probability}`` where probability is P(Fi before Fj).
        0.5 means undetermined; >0.5 means Fi likely comes first.
    """
    to_rank = pairs[:max_pairs]
    if not to_rank:
        return {}

    results: dict[tuple[str, str], float] = {}

    def _rank_one(pair: tuple[str, str]) -> tuple[tuple[str, str], float]:
        fi, fj = pair
        prompt = build_pairwise_prompt(fi, fj, mapper, file_contents)
        raw = ask_fn(prompt)
        # ask_fn may return (text, in_tok, out_tok)
        if isinstance(raw, tuple):
            raw = raw[0]
        return pair, parse_pairwise_response(raw, fi, fj)

    workers = min(len(to_rank), 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_rank_one, p) for p in to_rank]
        for fut in as_completed(futures):
            try:
                pair, prob = fut.result()
                results[pair] = prob
            except Exception:
                # On failure, treat as undetermined
                pass

    return results


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_pairwise_prompt(
    fi: str,
    fj: str,
    mapper: FileIDMapper,
    file_contents: dict[str, str],
) -> str:
    """Build a compact pairwise ranking prompt for two files.

    The prompt shows each file's basename, imports, and top definitions,
    then asks the LLM which file is the base dependency.

    Args:
        fi: First file ID (e.g. ``"F0"``).
        fj: Second file ID (e.g. ``"F1"``).
        mapper: FileIDMapper for resolving IDs.
        file_contents: Source code dict keyed by full path.

    Returns:
        Prompt string ready for the LLM.
    """
    def _summarise(fid: str) -> str:
        path = mapper.path_for(fid)
        bn = mapper.basename_for(fid)
        code = file_contents.get(path, "")

        imports: list[str] = []
        defs: list[str] = []
        for line in code.split("\n"):
            stripped = line.strip()
            if re.match(r"^\s*(?:from|import)\s", stripped):
                imports.append(stripped)
            m = re.match(r"^\s*(?:def|class)\s+(\w+)", stripped)
            if m:
                defs.append(m.group(1))

        parts = [f"{fid} = {bn}"]
        if imports:
            parts.append(f"  imports: {'; '.join(imports[:8])}")
        if defs:
            parts.append(f"  defines: {', '.join(defs[:8])}")
        if not imports and not defs:
            parts.append("  (empty or no recognisable code)")
        return "\n".join(parts)

    summary_i = _summarise(fi)
    summary_j = _summarise(fj)

    return (
        "Which file is the base dependency (should come first in load order)?\n\n"
        f"{summary_i}\n\n"
        f"{summary_j}\n\n"
        f"Answer with ONLY the file ID ({fi} or {fj}). "
        "If you cannot determine, answer UNKNOWN."
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_pairwise_response(response: str, fi: str, fj: str) -> float:
    """Parse the LLM's pairwise response into P(fi before fj).

    Args:
        response: Raw LLM response text.
        fi: First file ID.
        fj: Second file ID.

    Returns:
        Probability that fi comes before fj:
        - ~0.9 if the LLM clearly says fi is the base.
        - ~0.1 if the LLM clearly says fj is the base.
        - 0.5 if undetermined.
    """
    text = response.strip().upper()

    # Look for explicit file ID mentions
    fi_upper = fi.upper()
    fj_upper = fj.upper()

    # Check for exact mentions
    has_fi = bool(re.search(rf"\b{re.escape(fi_upper)}\b", text))
    has_fj = bool(re.search(rf"\b{re.escape(fj_upper)}\b", text))

    if "UNKNOWN" in text or "CANNOT" in text or "UNDETERMINED" in text:
        return 0.5

    if has_fi and not has_fj:
        return 0.9  # LLM says fi is the base
    if has_fj and not has_fi:
        return 0.1  # LLM says fj is the base

    # Both mentioned — check which comes first or look for context clues
    if has_fi and has_fj:
        fi_pos = text.find(fi_upper)
        fj_pos = text.find(fj_upper)
        # The first mentioned is usually the answer
        if fi_pos < fj_pos:
            return 0.7
        else:
            return 0.3

    # Neither found — undetermined
    return 0.5
