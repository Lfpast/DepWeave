"""Tolerant JSON parsing for LLM-produced synthesis output.

LLMs frequently emit JSON with trailing commas, JS-style comments, single
quotes, or markdown code fences. Strict ``json.loads`` fails on all of
these. This module extracts the first JSON object/array from a raw response
and normalizes common LLM tics before parsing.

Use :func:`parse_json_object` for ``{...}`` responses and
:func:`parse_json_array` for ``[...]`` responses.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional


_FENCE_RE = re.compile(r"```(?:json|javascript|js)?\s*(.*?)```", re.DOTALL)
_COMMENT_RE = re.compile(r"(?m)^\s*//.*$")
_TRAILING_COMMA_RE = re.compile(r",(\s*[\]}])")
# Python raw-string literal masquerading as JSON: r"..." or r'...'
_RAW_STRING_RE = re.compile(r"\br(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')")


def _strip_fences(s: str) -> str:
    m = _FENCE_RE.search(s)
    return m.group(1) if m else s


def _normalize(s: str) -> str:
    """Remove LLM-y tics that break ``json.loads``."""
    s = _strip_fences(s)
    s = _COMMENT_RE.sub("", s)  # drop // comments
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    # Python raw-string literals (r"...") are not valid JSON; strip the r prefix.
    s = _RAW_STRING_RE.sub(lambda m: m.group(1), s)
    return s


def _extract_span(s: str, open_ch: str, close_ch: str) -> Optional[str]:
    """Return the substring from the first open_ch to its matching close_ch.

    Counts nesting; ignores braces/brackets inside quoted strings.
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    quote_ch = ""
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == quote_ch:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote_ch = ch
            continue
        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                return s[start:i + 1]
    return None


def _best_effort_parse(raw: str) -> Optional[Any]:
    """Try json → json after normalize → python literal_eval. Return None on failure."""
    candidates = [raw, _normalize(raw)]
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            pass
    # Last resort: python literal_eval tolerates single quotes and trailing commas differently.
    try:
        return ast.literal_eval(_normalize(raw))
    except Exception:
        return None


def parse_json_object(raw: str) -> dict:
    """Extract and parse the first ``{...}`` in ``raw``. Raises ValueError on failure."""
    normalized = _normalize(raw)
    span = _extract_span(normalized, "{", "}")
    if span is None:
        raise ValueError(f"no JSON object found in response: {raw[:300]!r}")
    parsed = _best_effort_parse(span)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object, got {type(parsed).__name__}: {raw[:300]!r}")
    return parsed


def parse_json_array(raw: str) -> list:
    """Extract and parse the first ``[...]`` in ``raw``. Raises ValueError on failure."""
    normalized = _normalize(raw)
    span = _extract_span(normalized, "[", "]")
    if span is None:
        raise ValueError(f"no JSON array found in response: {raw[:300]!r}")
    parsed = _best_effort_parse(span)
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array, got {type(parsed).__name__}: {raw[:300]!r}")
    return parsed
