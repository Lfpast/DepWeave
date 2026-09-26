"""Confidence-scored import resolution.

Every candidate gets a calibrated score in [0, 1] — no hard-coded
if/else heuristics.  Ambiguity is tracked explicitly so downstream
consumers can decide how much to trust each edge.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from pixelmem.v3.depeval.file_id_mapper import FileIDMapper

# ── Known stdlib module names (first-party / built-in) ─────────────

STDLIB_MODULES: frozenset[str] = frozenset({
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
    "binhex", "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb",
    "cmath", "cmd", "code", "codecs", "codeop", "collections",
    "colorsys", "compileall", "concurrent", "configparser",
    "contextlib", "contextvars", "copy", "copyreg", "cProfile",
    "crypt", "csv", "ctypes", "curses", "dataclasses", "datetime",
    "dbm", "decimal", "difflib", "dis", "distutils", "doctest",
    "email", "encodings", "enum", "errno", "faulthandler", "fcntl",
    "filecmp", "fileinput", "fnmatch", "formatter", "fractions",
    "ftplib", "functools", "gc", "getopt", "getpass", "gettext",
    "glob", "grp", "gzip", "hashlib", "heapq", "hmac", "html",
    "http", "idlelib", "imaplib", "imghdr", "imp", "importlib",
    "inspect", "io", "ipaddress", "itertools", "json", "keyword",
    "lib2to3", "linecache", "locale", "logging", "lzma", "mailbox",
    "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder",
    "multiprocessing", "netrc", "nis", "nntplib", "numbers",
    "operator", "optparse", "os", "ossaudiodev", "pathlib", "pdb",
    "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint",
    "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr",
    "pydoc", "queue", "quopri", "random", "re", "readline",
    "reprlib", "resource", "rlcompleter", "runpy", "sched",
    "secrets", "select", "selectors", "shelve", "shlex", "shutil",
    "signal", "site", "smtpd", "smtplib", "sndhdr", "socket",
    "socketserver", "sqlite3", "ssl", "stat", "statistics",
    "string", "stringprep", "struct", "subprocess", "sunau",
    "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test",
    "textwrap", "threading", "time", "timeit", "tkinter", "token",
    "tokenize", "trace", "traceback", "tracemalloc", "tty",
    "turtle", "turtledemo", "types", "typing", "unicodedata",
    "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave",
    "weakref", "webbrowser", "winreg", "winsound", "wsgiref",
    "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile", "zipimport",
    "zlib", "_thread",
})

# ── Data classes ───────────────────────────────────────────────────


@dataclass
class ImportCandidate:
    """A single candidate resolution for an import statement."""
    source_file: str       # full path of the importing file
    target_file: str       # full path of the candidate target
    score: float           # calibrated score in [0, 1]
    reason: str            # e.g. "exact_package_path", "basename_match"
    ambiguity: int         # how many candidates scored > 0.3
    raw_import: str        # original import line text


# ── Feature weights ────────────────────────────────────────────────

_WEIGHTS: dict[str, float] = {
    "relative":   0.35,
    "package":    0.30,
    "basename":   0.15,
    "ambiguity":  0.10,
    "same_pkg":   0.05,
    "stdlib":    -0.10,
}


def calibrated_score(features: dict[str, float]) -> float:
    """Weighted combination of feature signals, clipped to [0, 1].

    Args:
        features: Keys matching ``_WEIGHTS``, values in [0, 1] (or
            negative for penalty signals).

    Returns:
        Final calibrated score.
    """
    total = sum(_WEIGHTS.get(k, 0.0) * v for k, v in features.items())
    return max(0.0, min(1.0, total))


# ── Import line parsing ───────────────────────────────────────────

# Matches: "from foo.bar import baz", "import foo.bar", "from .rel import x"
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(\.{0,3}[\w.]*)\s+import\s+[\w, *]+|import\s+([\w.]+))",
)


def _parse_import_line(line: str) -> tuple[str | None, bool]:
    """Parse an import line into (module_path, is_relative).

    Returns ``(None, False)`` when the line is not a recognisable import.
    """
    m = _IMPORT_RE.match(line)
    if not m:
        return None, False
    raw = m.group(1) or m.group(2)
    if raw is None:
        return None, False
    is_relative = raw.startswith(".")
    return raw, is_relative


def _module_path_segments(raw: str) -> list[str]:
    """Split ``'.foo.bar'`` or ``'foo.bar'`` into ``['foo', 'bar']``,
    stripping leading dots."""
    cleaned = raw.lstrip(".")
    if not cleaned:
        return []
    return cleaned.split(".")


def _count_leading_dots(raw: str) -> int:
    """Number of leading dots in a relative import path."""
    count = 0
    for ch in raw:
        if ch == ".":
            count += 1
        else:
            break
    return count


# ── Single-import resolution ──────────────────────────────────────

def _file_to_module_segments(path: str) -> list[str]:
    """Convert ``'/a/b/pkg/core/utils.py'`` to ``['pkg', 'core', 'utils']``."""
    parts = os.path.normpath(path).replace("\\", "/").split("/")
    # Drop the .py extension from the last segment
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    # Trim __init__ — the package is the parent dir
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def resolve_import(
    import_line: str,
    source_file: str,
    all_files: list[str],
    mapper: FileIDMapper,
) -> list[ImportCandidate]:
    """Resolve one import line against all candidate files.

    Returns all candidates ranked by score (highest first).
    """
    raw, is_relative = _parse_import_line(import_line)
    if raw is None:
        return []

    segments = _module_path_segments(raw)
    if not segments:
        # Bare relative import like ``from . import x`` — segment is just ``x``
        # Try extracting the imported name
        m = re.match(r"^\s*from\s+\.+\s+import\s+(\w+)", import_line)
        if m:
            segments = [m.group(1)]
        else:
            return []

    top_module = segments[0]

    # Quick check: if the top-level module is a known stdlib name and
    # the import is absolute, set a stdlib flag for scoring.
    is_stdlib_name = (not is_relative) and (top_module in STDLIB_MODULES)

    source_dir = os.path.dirname(os.path.normpath(source_file))
    source_segments = _file_to_module_segments(source_file)

    candidates: list[ImportCandidate] = []

    for candidate_path in all_files:
        normed = os.path.normpath(candidate_path)
        if normed == os.path.normpath(source_file):
            continue  # skip self

        cand_segments = _file_to_module_segments(normed)
        cand_dir = os.path.dirname(normed)

        features: dict[str, float] = {}

        # -- relative import signal --
        if is_relative:
            dots = _count_leading_dots(raw)
            # Walk up `dots` directories from source
            expected_base = source_dir
            for _ in range(dots - 1):
                expected_base = os.path.dirname(expected_base)
            # Build expected path from the relative root
            expected_tail = os.path.join(expected_base, *segments)
            # The candidate matches if its module path ends with the
            # expected tail segments.
            cand_tail = os.path.join(cand_dir, os.path.basename(normed).replace(".py", ""))
            if cand_tail.replace("\\", "/").endswith(expected_tail.replace("\\", "/")):
                features["relative"] = 1.0
            else:
                # Partial: check if last segment matches
                if segments and cand_segments and cand_segments[-1] == segments[-1]:
                    features["relative"] = 0.5
                else:
                    features["relative"] = 0.0
        else:
            features["relative"] = 0.0

        # -- package path match (fraction of segments that line up) --
        if segments and cand_segments:
            # Align from the right
            match_count = 0
            for s_seg, c_seg in zip(reversed(segments), reversed(cand_segments)):
                if s_seg == c_seg:
                    match_count += 1
                else:
                    break
            features["package"] = match_count / len(segments)
        else:
            features["package"] = 0.0

        # -- basename match --
        cand_basename = os.path.basename(normed).replace(".py", "")
        if cand_basename == "__init__":
            # For __init__.py, use the parent dir name
            cand_basename = os.path.basename(os.path.dirname(normed))
        features["basename"] = 1.0 if (segments[-1] == cand_basename) else 0.0

        # -- same_package bonus --
        if source_dir == cand_dir:
            features["same_pkg"] = 1.0
        elif os.path.dirname(source_dir) == os.path.dirname(cand_dir):
            features["same_pkg"] = 0.5
        else:
            features["same_pkg"] = 0.0

        # -- stdlib conflict penalty --
        features["stdlib"] = 1.0 if is_stdlib_name else 0.0

        # Compute score (ambiguity applied later)
        features["ambiguity"] = 0.0  # placeholder
        score = calibrated_score(features)

        if score > 0.0:
            candidates.append(ImportCandidate(
                source_file=source_file,
                target_file=normed,
                score=score,
                reason=_primary_reason(features),
                ambiguity=0,  # filled in below
                raw_import=import_line.strip(),
            ))

    # -- ambiguity adjustment --
    above_threshold = sum(1 for c in candidates if c.score > 0.3)
    for c in candidates:
        c.ambiguity = above_threshold
        if above_threshold > 1:
            # Apply ambiguity penalty: 1/n bonus becomes smaller
            penalty = 1.0 - (1.0 / above_threshold)
            c.score = max(0.0, c.score - _WEIGHTS["ambiguity"] * penalty)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _primary_reason(features: dict[str, float]) -> str:
    """Pick the dominant feature as the human-readable reason."""
    # Only consider positive features
    positive = {k: _WEIGHTS.get(k, 0) * v for k, v in features.items()
                if _WEIGHTS.get(k, 0) * v > 0}
    if not positive:
        return "weak_signal"
    best = max(positive, key=positive.get)  # type: ignore[arg-type]
    reason_names = {
        "relative": "relative_import",
        "package": "exact_package_path",
        "basename": "basename_match",
        "same_pkg": "same_package",
        "ambiguity": "low_ambiguity",
    }
    return reason_names.get(best, best)


# ── Batch resolution ──────────────────────────────────────────────

_IMPORT_LINE_RE = re.compile(r"^\s*(from\s+|import\s+)")


def resolve_all_imports(
    files: list[str],
    file_contents: dict[str, str],
    mapper: FileIDMapper,
    threshold: float = 0.1,
) -> list[ImportCandidate]:
    """Resolve every import line in every file.

    Args:
        files: All file paths in the project.
        file_contents: Mapping of ``full_path -> source_text``.
        mapper: ``FileIDMapper`` already built from *files*.
        threshold: Minimum score to keep a candidate (default 0.1).

    Returns:
        Flat list of all ``ImportCandidate`` objects above *threshold*,
        across all files and all import lines.
    """
    results: list[ImportCandidate] = []

    for src_path in files:
        content = file_contents.get(src_path, "")
        for line in content.splitlines():
            if not _IMPORT_LINE_RE.match(line):
                continue
            candidates = resolve_import(line, src_path, files, mapper)
            results.extend(c for c in candidates if c.score >= threshold)

    return results
