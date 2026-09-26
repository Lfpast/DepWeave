"""Tool Registry — build once, reuse forever.

Maintains a persistent cache of extraction tools per language.
First question with Python files builds the Python tool.
All subsequent Python questions reuse it — zero cost.

Also tracks tool build stats (how many times built vs reused).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from pixelmem.v3.lang_extraction import (
    detect_language, get_extractor, _LANG_EXTRACTORS, _LEARNED_PATTERNS,
    _is_import_line,
)
from pixelmem.v3.file_summary_store import _get_def_patterns


@dataclass
class ToolEntry:
    """A cached extraction tool for one language."""
    language: str
    extractor: Callable  # function(code) → list[import_targets]
    import_line_checker: Callable  # function(line) → bool
    def_patterns: list[str]
    build_method: str  # "builtin", "learned", "fallback"
    times_used: int = 0
    times_built: int = 1


class ToolRegistry:
    """Persistent registry of extraction tools per language.

    Usage:
        registry = ToolRegistry()

        # First Python question — builds tool
        tool = registry.get_tool("python", sample_code, ask_fn)

        # Second Python question — reuses cached tool (free)
        tool = registry.get_tool("python")

        # First Java question — builds Java tool
        tool = registry.get_tool("java", sample_code, ask_fn)

        # Stats
        print(registry.stats())
    """

    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}

    def get_tool(
        self,
        language: Optional[str] = None,
        files: Optional[list[str]] = None,
        sample_code: str = "",
        ask_fn: Optional[Callable] = None,
    ) -> ToolEntry:
        """Get or build an extraction tool for a language.

        Args:
            language: Language name. If None, detected from files.
            files: File paths for language detection.
            sample_code: Sample code for building extractors (needed first time only).
            ask_fn: LLM function for building extractors for unknown languages.

        Returns:
            ToolEntry with extractor, import checker, and def patterns.
        """
        if language is None and files:
            language = detect_language(files)
        if language is None:
            language = "unknown"

        # Cache hit — reuse existing tool
        if language in self._tools:
            entry = self._tools[language]
            entry.times_used += 1
            return entry

        # Cache miss — build new tool
        extractor = get_extractor(language, sample_code, ask_fn)
        def_patterns = _get_def_patterns(language)

        # Determine build method
        if language in _LANG_EXTRACTORS:
            build_method = "builtin"
        elif language in _LEARNED_PATTERNS:
            build_method = "learned"
        else:
            build_method = "fallback"

        def import_checker(line: str) -> bool:
            return _is_import_line(line, language)

        entry = ToolEntry(
            language=language,
            extractor=extractor,
            import_line_checker=import_checker,
            def_patterns=def_patterns,
            build_method=build_method,
        )

        self._tools[language] = entry
        return entry

    def has_tool(self, language: str) -> bool:
        return language in self._tools

    @property
    def languages(self) -> list[str]:
        return list(self._tools.keys())

    def stats(self) -> dict:
        """Tool build/reuse statistics."""
        total_built = sum(e.times_built for e in self._tools.values())
        total_reused = sum(e.times_used for e in self._tools.values())
        return {
            "languages": len(self._tools),
            "total_built": total_built,
            "total_reused": total_reused,
            "tools": {
                lang: {
                    "build_method": e.build_method,
                    "times_used": e.times_used,
                }
                for lang, e in self._tools.items()
            },
        }

    def save(self, path: str | Path) -> None:
        """Save learned patterns (builtin tools don't need saving)."""
        data = {}
        for lang, entry in self._tools.items():
            if entry.build_method == "learned" and lang in _LEARNED_PATTERNS:
                data[lang] = _LEARNED_PATTERNS[lang]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load previously learned patterns."""
        try:
            with open(path) as f:
                data = json.load(f)
            for lang, patterns in data.items():
                _LEARNED_PATTERNS[lang] = patterns
                # Pre-build the tool entry
                self.get_tool(language=lang)
        except (FileNotFoundError, json.JSONDecodeError):
            pass


# Global singleton
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def prebuild_tools(
    questions: list[dict],
    file_key: str = "files",
    content_key: str = "content",
    ask_fn: Optional[Callable] = None,
) -> ToolRegistry:
    """Pre-build all needed tools before parallel execution.

    Scans all questions to detect which languages are needed,
    builds one tool per language, caches in the global registry.

    Call this ONCE before running questions in parallel batches.

    Args:
        questions: List of question dicts containing file lists.
        file_key: Key in each dict containing file paths.
        content_key: Key containing file content (for sample code).
        ask_fn: LLM function for building unknown language extractors.

    Returns:
        The populated registry.
    """
    registry = get_registry()

    # Scan all questions to find unique languages
    languages_needed: dict[str, str] = {}  # language → sample_code

    for q in questions:
        files = q.get(file_key, [])
        if not files:
            continue
        # Clean file names
        clean_files = [f.strip("'\"") for f in files]
        lang = detect_language(clean_files)

        if lang not in languages_needed:
            # Get sample code from content if available
            content = q.get(content_key, "")
            if content:
                # Take first 500 chars as sample
                languages_needed[lang] = content[:500]
            else:
                languages_needed[lang] = ""

    # Build all tools upfront (sequential — fast for builtin, 1 LLM call per unknown)
    print(f"Pre-building tools for {len(languages_needed)} languages: {list(languages_needed.keys())}")
    for lang, sample in languages_needed.items():
        if not registry.has_tool(lang):
            tool = registry.get_tool(language=lang, sample_code=sample, ask_fn=ask_fn)
            print(f"  Built: {lang} ({tool.build_method})")
        else:
            print(f"  Cached: {lang}")

    return registry
