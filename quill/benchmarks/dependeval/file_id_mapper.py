"""Stable file-ID mapping for dependency evaluation.

Assigns short IDs (F0, F1, F2...) to files so that LLM prompts
and responses can reference files unambiguously, regardless of
duplicate basenames across different directories.
"""

from __future__ import annotations

import os
import re
from collections import Counter


class FileIDMapper:
    """Bidirectional mapping between full file paths and short IDs.

    Args:
        files: Ordered list of full file paths.  IDs are assigned in
            the order provided (F0 for files[0], F1 for files[1], ...).
    """

    def __init__(self, files: list[str]) -> None:
        self._path_to_id: dict[str, str] = {}
        self._id_to_path: dict[str, str] = {}

        for idx, path in enumerate(files):
            fid = f"F{idx}"
            normed = os.path.normpath(path)
            self._path_to_id[normed] = fid
            self._id_to_path[fid] = normed

    # ── lookups ────────────────────────────────────────────────────

    def id_for(self, path: str) -> str:
        """Return the file ID for a full path (e.g. ``"F0"``)."""
        normed = os.path.normpath(path)
        return self._path_to_id[normed]

    def path_for(self, fid: str) -> str:
        """Return the full path for a file ID."""
        return self._id_to_path[fid]

    def basename_for(self, fid: str) -> str:
        """Return just the filename component (e.g. ``"file.py"``)."""
        return os.path.basename(self._id_to_path[fid])

    def parent_for(self, fid: str) -> str:
        """Return the immediate parent directory name."""
        return os.path.basename(os.path.dirname(self._id_to_path[fid]))

    def all_ids(self) -> list[str]:
        """Return all assigned IDs in order (``["F0", "F1", ...]``)."""
        return [f"F{i}" for i in range(len(self._id_to_path))]

    # ── formatting ─────────────────────────────────────────────────

    def format_file_table(self) -> str:
        """Human-readable file table for LLM prompts.

        Appends ``(in dir/)`` only when two or more files share the
        same basename, so that the table stays compact.

        Returns:
            Multi-line string, one file per line, e.g.::

                F0 = utils.py
                F1 = __init__.py (in criteria/)
                F2 = __init__.py (in ls/)
        """
        # Count basenames to detect duplicates
        basename_counts: Counter[str] = Counter()
        for fid in self.all_ids():
            basename_counts[self.basename_for(fid)] += 1

        lines: list[str] = []
        for fid in self.all_ids():
            bname = self.basename_for(fid)
            if basename_counts[bname] > 1:
                parent = self.parent_for(fid)
                lines.append(f"{fid} = {bname} (in {parent}/)")
            else:
                lines.append(f"{fid} = {bname}")
        return "\n".join(lines)

    # ── parsing LLM responses ─────────────────────────────────────

    def parse_id_response(self, text: str) -> list[str]:
        """Extract file IDs from free-form LLM output.

        Handles JSON arrays (``["F0", "F2"]``), comma-separated
        (``F0, F2, F1``), and space-separated (``F0 F2 F1``) formats.

        Returns:
            Ordered list of valid IDs found in *text*.
        """
        # Find all tokens that look like F<digits>
        candidates = re.findall(r"\bF\d+\b", text)
        # Keep only IDs that actually exist in our mapping
        valid = self._id_to_path.keys()
        return [c for c in candidates if c in valid]

    # ── convenience ────────────────────────────────────────────────

    def ids_to_basenames(self, ids: list[str]) -> list[str]:
        """Convert a list of IDs to their basenames."""
        return [self.basename_for(fid) for fid in ids]

    def __len__(self) -> int:
        return len(self._id_to_path)

    def __repr__(self) -> str:
        return f"FileIDMapper({len(self)} files)"
