"""Workload-reducing tool: surface symbol candidates, let the LLM pick.

The premise: on specialized tasks (code completion, clinical QA, legal
look-ups), the tool often can't produce the final answer directly —
but it CAN hand the LLM a short, ranked list of candidate symbols
extracted from whatever context is available. This shrinks the LLM's
output space from "the universe of possible Python identifiers" to
"one of these 10 names, or flag that none fits."

Two pieces:

1. :class:`SymbolExtractor` — pulls function / class / import / variable
   symbol NAMES (not full lines) from documents, so the candidate list is
   compact.

2. :class:`CandidateAwarePrompt` — parses the completion site (what KIND of
   token is expected?), ranks extracted symbols by relevance, surfaces the
   top-K in the prompt, and instructs the LLM to:
     (a) use the exact candidate name when one fits, or
     (b) acknowledge that no candidate fits and emit a best-effort guess
         flagged as uncertain.

Designed for Python line-completion (CCEval) but the interface is
language-agnostic; swap the regex constants for another language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from quill.plugins import Extractor, PromptTemplate
from quill.types import EvidenceBundle, Primitive, TaskSpec


# ---------------------------------------------------------------------------
# Symbol extraction (names only — not full lines)
# ---------------------------------------------------------------------------


_PY_FUNC_DEF = re.compile(r"^\s*#?\s*(?:async\s+)?def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_PY_CLASS_DEF = re.compile(r"^\s*#?\s*class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_PY_IMPORT_FROM = re.compile(r"^\s*#?\s*from\s+[\w.]+\s+import\s+(?P<names>[^#\n]+)", re.MULTILINE)
_PY_IMPORT_PLAIN = re.compile(r"^\s*#?\s*import\s+(?P<name>[\w.]+)(?:\s+as\s+(?P<alias>\w+))?", re.MULTILINE)
_PY_ASSIGN = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE)
# Usage-site patterns: capture symbols that are CALLED somewhere, even in
# comments. These are the best signal when the defining file isn't in context.
_PY_METHOD_CALL = re.compile(r"\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PY_FUNC_CALL = re.compile(r"(?<![\.\w])(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
# Receiver-aware method call: capture the dotted-chain receiver before the
# method name. `self.rule.get_header_value(` → receiver="self.rule", method="get_header_value"
_PY_RECEIVER_METHOD = re.compile(
    r"(?P<recv>(?:self|cls|[A-Za-z_][A-Za-z0-9_]*)(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
)

# Python keywords / builtins we never want to emit as candidates.
_NOISE_NAMES = frozenset({
    "if", "elif", "else", "for", "while", "with", "try", "except", "finally",
    "return", "yield", "raise", "pass", "break", "continue", "import", "from",
    "as", "in", "is", "not", "and", "or", "lambda", "def", "class",
    "True", "False", "None", "self", "cls",
    "print", "len", "range", "list", "tuple", "dict", "set", "str", "int",
    "float", "bool", "type", "isinstance", "hasattr", "getattr", "setattr",
    "super", "open", "map", "filter", "sorted", "zip", "enumerate", "iter",
    "next", "all", "any", "sum", "min", "max", "abs", "round",
})


class SymbolExtractor(Extractor):
    """Extract bare symbol names (not full lines) from Python documents.

    Relations:
      - ``defines_function``   (subject = doc_id, object = function name)
      - ``defines_class``      (subject = doc_id, object = class name)
      - ``imports_symbol``     (subject = doc_id, object = imported name)
      - ``imports_module``     (subject = doc_id, object = module dotted path)
      - ``assigns_local``      (subject = doc_id, object = top-level variable)
    """

    def extract(self, documents: dict[str, str], **kwargs: Any) -> list[Primitive]:
        out: list[Primitive] = []
        for doc_id, text in documents.items():
            if not isinstance(text, str):
                continue
            seen_def: set[str] = set()
            for m in _PY_FUNC_DEF.finditer(text):
                n = m.group("name")
                if n in _NOISE_NAMES or n in seen_def:
                    continue
                seen_def.add(n)
                out.append(Primitive(doc_id, "defines_function", n,
                                     "symbol", {"doc_id": doc_id}))
            for m in _PY_CLASS_DEF.finditer(text):
                n = m.group("name")
                if n in _NOISE_NAMES:
                    continue
                out.append(Primitive(doc_id, "defines_class", n,
                                     "symbol", {"doc_id": doc_id}))
            for m in _PY_IMPORT_FROM.finditer(text):
                names = m.group("names")
                for piece in names.split(","):
                    piece = piece.strip()
                    if not piece:
                        continue
                    if " as " in piece:
                        piece = piece.split(" as ")[1].strip()
                    piece = piece.rstrip("\\").strip()
                    if (piece and re.match(r"[A-Za-z_][A-Za-z0-9_]*$", piece)
                            and piece not in _NOISE_NAMES):
                        out.append(Primitive(doc_id, "imports_symbol", piece,
                                             "external", {"doc_id": doc_id}))
            for m in _PY_IMPORT_PLAIN.finditer(text):
                name = m.group("alias") or m.group("name").split(".")[-1]
                if name in _NOISE_NAMES:
                    continue
                out.append(Primitive(doc_id, "imports_module", name,
                                     "external", {"doc_id": doc_id}))
            # Top-level variables
            seen_vars: set[str] = set()
            for m in _PY_ASSIGN.finditer(text):
                name = m.group("name")
                if name in _NOISE_NAMES or name in seen_vars:
                    continue
                seen_vars.add(name)
                out.append(Primitive(doc_id, "assigns_local", name,
                                     "local", {"doc_id": doc_id}))
            # Usage sites — capture methods and functions CALLED anywhere,
            # including in commented-out example code. Count occurrences
            # so the ranker can use frequency as a signal.
            method_counts: dict[str, int] = {}
            # Track which receivers each method is called on: method -> {receiver: count}.
            # Used by the ranker to boost methods whose receiver matches the
            # completion site.
            method_receivers: dict[str, dict[str, int]] = {}
            for m in _PY_RECEIVER_METHOD.finditer(text):
                n = m.group("name")
                recv = m.group("recv")
                if n in _NOISE_NAMES:
                    continue
                method_counts[n] = method_counts.get(n, 0) + 1
                recv_tail = recv.split(".")[-1] if "." in recv else recv
                method_receivers.setdefault(n, {})
                method_receivers[n][recv] = method_receivers[n].get(recv, 0) + 1
                # Also record the receiver's last segment for fuzzy matching.
                method_receivers[n][f".{recv_tail}"] = method_receivers[n].get(
                    f".{recv_tail}", 0) + 1
            # Catch calls with non-dotted receivers too (fallback)
            for m in _PY_METHOD_CALL.finditer(text):
                n = m.group("name")
                if n in _NOISE_NAMES or n in method_counts:
                    continue
                method_counts[n] = method_counts.get(n, 0) + 1
            for name, count in method_counts.items():
                out.append(Primitive(
                    doc_id, "used_as_method", name,
                    f"count={count}",
                    {"doc_id": doc_id, "count": count,
                     "receivers": method_receivers.get(name, {})},
                ))
            func_counts: dict[str, int] = {}
            for m in _PY_FUNC_CALL.finditer(text):
                n = m.group("name")
                if n in _NOISE_NAMES:
                    continue
                func_counts[n] = func_counts.get(n, 0) + 1
            for name, count in func_counts.items():
                out.append(Primitive(doc_id, "used_as_function", name,
                                     f"count={count}", {"doc_id": doc_id,
                                                        "count": count}))
        return out


# ---------------------------------------------------------------------------
# Completion-site analyzer
# ---------------------------------------------------------------------------


@dataclass
class CompletionSite:
    """What kind of thing is the next line expected to be?"""
    kind: str            # "method_call", "attribute", "bare", "arg_position", "unknown"
    receiver: Optional[str] = None   # the expression before `.`, if any
    partial: Optional[str] = None    # the partial identifier being completed, if any
    surrounding_line: str = ""


def _analyze_completion_site(focal_tail: str) -> CompletionSite:
    """Heuristic parse of the last non-blank line.

    Covers the common CCEval patterns: trailing `.`, trailing `(`, trailing
    identifier chars after `.`, assignment LHS.
    """
    lines = [ln for ln in focal_tail.rstrip().splitlines() if ln.strip()]
    if not lines:
        return CompletionSite(kind="unknown")
    line = lines[-1].rstrip()

    # Trailing `.` → method or attribute call on preceding receiver.
    m = re.search(r"([A-Za-z_][A-Za-z0-9_\.\[\]]*)\.$", line)
    if m:
        return CompletionSite(kind="method_call", receiver=m.group(1),
                              surrounding_line=line)
    # Partial identifier after `.` — e.g. `.foo`
    m = re.search(r"([A-Za-z_][A-Za-z0-9_\.\[\]]*)\.([A-Za-z_][A-Za-z0-9_]*)$", line)
    if m:
        return CompletionSite(kind="attribute",
                              receiver=m.group(1), partial=m.group(2),
                              surrounding_line=line)
    # Trailing `(` → start of arg list
    if line.endswith("(") or line.endswith(","):
        return CompletionSite(kind="arg_position", surrounding_line=line)
    # Default
    return CompletionSite(kind="bare", surrounding_line=line)


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


def _rank_candidates(
    primitives: list[Primitive],
    site: CompletionSite,
    focal_tail: str,
    top_k: int = 12,
) -> list[tuple[str, str, int]]:
    """Return [(symbol_name, kind, score)] ranked descending.

    ``kind`` is one of ``function``, ``class``, ``import``, ``module``, ``var``.
    Score is a simple integer: base + completion-site match + focal-recency.
    """
    # Deduplicate by (symbol, kind) — multiple docs can define the same name.
    scores: dict[tuple[str, str], int] = {}
    # Merged receiver info across primitives for used_methods.
    receiver_info: dict[str, dict[str, int]] = {}
    for p in primitives:
        kind = {
            "defines_function": "function",
            "defines_class":    "class",
            "imports_symbol":   "import",
            "imports_module":   "module",
            "assigns_local":    "var",
            "used_as_method":   "used_method",
            "used_as_function": "used_func",
        }.get(p.relation)
        if kind is None:
            continue
        key = (p.object, kind)
        base = scores.get(key, 10)
        # Usage-count bonus: frequently-referenced symbols rank higher.
        if p.provenance and isinstance(p.provenance.get("count"), int):
            base = max(base, 10 + min(20, p.provenance["count"] * 3))
        scores[key] = base
        # Merge receiver info for used_methods
        if kind == "used_method" and p.provenance:
            recvs = p.provenance.get("receivers", {}) or {}
            tgt = receiver_info.setdefault(p.object, {})
            for r, c in recvs.items():
                tgt[r] = tgt.get(r, 0) + c

    # Underscore-prefix penalty: names starting with `_` are usually
    # private helpers. When a public alternative ranks nearby, the public
    # one is almost always the intended completion.
    for (sym, kind), _ in list(scores.items()):
        if sym.startswith("_") and not sym.startswith("__"):
            scores[(sym, kind)] -= 15

    # Completion-site match bonus
    for (sym, kind), _ in list(scores.items()):
        if site.kind == "method_call":
            if kind in ("function", "import", "used_method"):
                scores[(sym, kind)] += 30
            if kind == "class":
                scores[(sym, kind)] += 5
            # Receiver-match bonus: for method_call sites, HEAVILY favor
            # methods whose recorded receivers match the completion site's
            # receiver. Also PENALIZE methods called on unrelated receivers
            # (e.g. test-framework assertions on `self` when the site is
            # `self.rule.`).
            if kind == "used_method" and site.receiver:
                recvs = receiver_info.get(sym, {})
                site_recv = site.receiver
                site_recv_tail = site_recv.split(".")[-1]
                # Exact match: 'self.rule' — big bonus
                if site_recv in recvs:
                    scores[(sym, kind)] += 60
                # Same tail segment: '.rule' — medium bonus
                elif f".{site_recv_tail}" in recvs:
                    scores[(sym, kind)] += 30
                elif recvs:
                    # Method is called on OTHER receivers; de-prioritize vs
                    # a method that has no receiver info (could match).
                    max_other = max(recvs.values())
                    if max_other >= 3:
                        scores[(sym, kind)] -= 25
        elif site.kind == "attribute":
            partial = (site.partial or "").lower()
            if partial and sym.lower().startswith(partial):
                scores[(sym, kind)] += 50
            if kind in ("function", "import", "used_method"):
                scores[(sym, kind)] += 10
        elif site.kind == "arg_position":
            if kind in ("var", "import", "used_func"):
                scores[(sym, kind)] += 20
        elif site.kind == "bare":
            if kind in ("function", "class", "used_func"):
                scores[(sym, kind)] += 5

    # Focal-file recency: if the symbol appears in the focal tail it's
    # more likely to be used again nearby.
    for (sym, kind), _ in list(scores.items()):
        if re.search(rf"\b{re.escape(sym)}\b", focal_tail):
            scores[(sym, kind)] += 8

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    ranked = [(sym, kind, score) for (sym, kind), score in ranked[:top_k]]
    return ranked


# ---------------------------------------------------------------------------
# Candidate-aware prompt
# ---------------------------------------------------------------------------


class CandidateAwarePrompt(PromptTemplate):
    """Prompt that surfaces ranked candidates and asks the LLM to pick.

    Design:
      - If at least one strong-scoring candidate exists, tell the LLM to
        use the exact candidate name.
      - If no candidate clearly fits (max score below threshold), tell the
        LLM to emit its best guess but flag the uncertainty.
      - Include a site analysis so the LLM knows what SHAPE the answer
        should take (method call, arg list, attribute access, ...).
    """

    _FIT_THRESHOLD = 30  # minimum candidate score to count as "information present"

    def build(self, task: TaskSpec, query_input: dict, evidence: EvidenceBundle) -> str:
        focal = query_input.get("focal_prompt", "") or ""
        focal_tail = "\n".join(focal.splitlines()[-40:])
        site = _analyze_completion_site(focal_tail)

        ranked = _rank_candidates(
            evidence.raw_primitives, site, focal_tail, top_k=12,
        )
        info_present = bool(ranked and ranked[0][2] >= self._FIT_THRESHOLD)

        cand_block_lines = [
            f"  {sym} ({kind}, score={score})"
            for sym, kind, score in ranked
        ]
        cand_block = "\n".join(cand_block_lines) if cand_block_lines else "  (no candidates extracted)"

        site_str = (
            f"kind={site.kind}"
            + (f" receiver={site.receiver}" if site.receiver else "")
            + (f" partial={site.partial}" if site.partial else "")
        )

        if info_present:
            picking_rule = (
                "The candidate symbols above are the most likely identifiers "
                "for the next line. WRITE A SYNTACTICALLY COMPLETE CONTINUATION "
                "(not just a bare symbol). If one of the top candidates fits, "
                "use its EXACT name from the candidates list inside the full "
                "completed expression. Do not invent similar-but-different "
                "names like FOO_from_response when FOO itself is a candidate. "
                "Close any open parens/brackets as needed. Your output is "
                "ONE source line."
            )
        else:
            picking_rule = (
                "None of the candidates match strongly. The correct symbol "
                "may be defined in a file NOT included in this prompt. "
                "Write a syntactically complete continuation using the "
                "focal file's existing patterns (similar call shapes, "
                "argument styles). Keep the guess short. Your output is "
                "ONE source line."
            )

        xfile = ""
        for doc_id, text in (query_input.get("documents") or {}).items():
            if doc_id == "__crossfile_context__":
                xfile = str(text)[:1500]
                break

        return (
            "You are completing the NEXT LINE of a Python file. Return ONLY "
            "the next line — no explanation, no code fences.\n\n"
            f"Completion site: {site_str}\n\n"
            f"Candidate symbols (from extracted primitives, ranked):\n{cand_block}\n\n"
            f"Cross-file context (truncated):\n{xfile}\n\n"
            f"Focal file tail:\n{focal_tail}\n\n"
            f"INSTRUCTIONS: {picking_rule}"
        )

    def parse(self, completion: str, task: TaskSpec) -> str:
        txt = completion.strip()
        if txt.startswith("```"):
            txt = "\n".join(l for l in txt.splitlines() if not l.startswith("```")).strip()
        for line in txt.splitlines():
            if line.strip():
                return line.rstrip("\n")
        raise ValueError("no non-empty line in completion")


__all__ = [
    "SymbolExtractor",
    "CompletionSite",
    "CandidateAwarePrompt",
]
