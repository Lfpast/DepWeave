"""RepoQA function extraction and candidate card."""

from __future__ import annotations

import ast as _pyast
import re
from typing import Any

from core.plugins import Extractor, PromptTemplate
from core.types import EvidenceBundle, Primitive, TaskSpec

REPOQA_JSON = "/home/jackson/python/DepWeave/data/RepoQA/repoqa-2024-06-23.json"

# ---------------------------------------------------------------------------
# RepoQA extractor: one primitive per function definition in the repo
# ---------------------------------------------------------------------------


class RepoQAFunctionExtractor(Extractor):
    """Parses every Python file in the repo and emits a Primitive per top-level
    function/method. Provenance stores docstring, name, and a body snippet.
    """

    def extract(self, documents: dict[str, str], **kwargs: Any) -> list[Primitive]:
        out: list[Primitive] = []
        for path, code in documents.items():
            if not isinstance(code, str) or not path.endswith(".py"):
                continue
            try:
                tree = _pyast.parse(code)
            except SyntaxError:
                continue
            for node in _pyast.walk(tree):
                if isinstance(node, (_pyast.FunctionDef, _pyast.AsyncFunctionDef)):
                    docstring = _pyast.get_docstring(node) or ""
                    body_src = self._body_snippet(code, node)
                    out.append(Primitive(
                        path,
                        "defines_function",
                        node.name,
                        f"lineno={node.lineno}",
                        {
                            "path": path,
                            "lineno": node.lineno,
                            "docstring": docstring,
                            "snippet": body_src,
                        },
                    ))
        return out

    def _body_snippet(self, code: str, node, max_chars: int = 600) -> str:
        # Pull the function's source by line numbers (approximate).
        lines = code.splitlines()
        start = node.lineno - 1
        end = min(start + 20, len(lines))
        return "\n".join(lines[start:end])[:max_chars]


# ---------------------------------------------------------------------------
# RepoQA prompt — candidate-aware, ranks by overlap with NL description
# ---------------------------------------------------------------------------


_STOP = frozenset({
    "the", "a", "an", "is", "and", "or", "of", "to", "in", "on", "for",
    "with", "by", "from", "at", "as", "that", "this", "it", "its",
    "be", "are", "was", "were", "will", "can", "should", "would",
    "if", "else", "when", "where", "what", "how", "which", "who",
    "not", "no", "but", "also", "these", "those", "any", "all",
    "function", "method", "returns", "return", "takes", "given",
    "value", "values", "object", "objects", "code",
})


def _tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_]+", (s or "").lower())
            if w not in _STOP and len(w) > 2]


def _score_function_against_desc(func_name: str, docstring: str,
                                 snippet: str, desc_tokens: set[str]) -> int:
    """Count overlapping tokens (description word count that appear in function context)."""
    cand_tokens = set(_tokens(func_name)) | set(_tokens(docstring)) | set(_tokens(snippet))
    return len(cand_tokens & desc_tokens)


class RepoQASearchPrompt(PromptTemplate):
    def build(self, task: TaskSpec, query_input: dict, evidence: EvidenceBundle) -> str:
        description = query_input.get("description", "") or ""
        desc_tokens = set(_tokens(description))

        # Score all function primitives and pick top-K.
        scored = []
        for p in evidence.raw_primitives:
            if p.relation != "defines_function":
                continue
            prov = p.provenance or {}
            s = _score_function_against_desc(
                p.object, prov.get("docstring", ""),
                prov.get("snippet", ""), desc_tokens,
            )
            scored.append((s, p))

        scored.sort(key=lambda x: -x[0])
        top_k = 12
        top = scored[:top_k]

        cand_lines = []
        for score, p in top:
            prov = p.provenance or {}
            doc1 = (prov.get("docstring") or "").splitlines()[0] if prov.get("docstring") else ""
            cand_lines.append(
                f"  {p.object}  (score={score})"
                + (f"\n     docstring: {doc1[:120]}" if doc1 else "")
                + f"\n     path: {prov.get('path')}:{prov.get('lineno')}"
            )
        cand_block = "\n".join(cand_lines) if cand_lines else "  (no candidates extracted)"

        return (
            "Task: pick the Python function from the candidate list below whose "
            "behavior best matches the natural-language description. The "
            "function name itself has been obfuscated in the description.\n\n"
            f"DESCRIPTION:\n{description[:2000]}\n\n"
            f"TOP-{len(top)} CANDIDATE FUNCTIONS (ranked by word overlap):\n"
            f"{cand_block}\n\n"
            "Return ONLY the exact function name (the identifier shown at the "
            "start of each candidate row). No labels like 'C0', no 'score=', "
            "no explanation. Just the identifier."
        )

    def parse(self, completion: str, task: TaskSpec) -> str:
        txt = completion.strip()
        if txt.startswith("```"):
            txt = "\n".join(l for l in txt.splitlines() if not l.startswith("```")).strip()
        # First identifier-like token
        m = re.search(r"[A-Za-z_][A-Za-z0-9_]*", txt)
        if not m:
            raise ValueError("no identifier in completion")
        return m.group(0)

