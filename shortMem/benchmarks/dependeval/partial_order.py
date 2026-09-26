"""Partial order with confidence-weighted constraints.

A partial order over file IDs where each pairwise constraint carries a
confidence score.  Strong constraints (conf > 0.7) are treated as hard
edges that MUST be respected in the final ordering.  Weaker constraints
are advisory — they influence tie-breaking but can be violated.

Unlike bayesian_order.py which blends everything into soft scores and
hopes the sort respects them, this data structure cleanly separates
"known ordering" from "ambiguous ordering" so the constrained decoder
can guarantee correctness on the hard edges.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

STRONG_THRESHOLD = 0.7   # above this: constraint MUST be respected
AMBIGUOUS_CEILING = 0.3  # below this (or absent): pair is ambiguous


class PartialOrder:
    """Directed graph of ordering constraints with confidence weights.

    Nodes are file IDs (e.g. ``"F0"``, ``"F1"``).  An edge
    ``(before, after, conf)`` means *before* should appear earlier than
    *after* in the final total order, with the given confidence.

    If multiple pieces of evidence add the same edge, the maximum
    confidence is kept (optimistic merge — one strong signal is enough).
    """

    def __init__(self, file_ids: list[str]) -> None:
        self.file_ids: list[str] = list(file_ids)
        self._id_set: set[str] = set(file_ids)
        # (before, after) → confidence
        self._constraints: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_constraint(
        self,
        before: str,
        after: str,
        confidence: float,
    ) -> None:
        """Record that *before* must come before *after*.

        If the same directed pair already has a constraint, the maximum
        confidence wins (one strong signal overrides earlier weak ones).

        Args:
            before: File ID that should appear earlier.
            after:  File ID that should appear later.
            confidence: Confidence in [0, 1].

        Raises:
            ValueError: If either ID is not in the original file_ids.
        """
        if before not in self._id_set:
            raise ValueError(f"Unknown file ID: {before!r}")
        if after not in self._id_set:
            raise ValueError(f"Unknown file ID: {after!r}")
        if before == after:
            return  # self-loop is meaningless

        key = (before, after)
        self._constraints[key] = max(
            self._constraints.get(key, 0.0),
            min(max(confidence, 0.0), 1.0),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_strong_constraints(self) -> list[tuple[str, str]]:
        """Return ``(before, after)`` pairs where confidence > STRONG_THRESHOLD.

        These are the HARD edges that the decoder must never violate.
        """
        return [
            (b, a)
            for (b, a), c in self._constraints.items()
            if c > STRONG_THRESHOLD
        ]

    def get_ambiguous_pairs(self) -> list[tuple[str, str]]:
        """Return unordered pairs with no constraint or max conf < AMBIGUOUS_CEILING.

        These are pairs where we have essentially no evidence of ordering,
        so the decoder is free to use heuristics or pairwise scores.
        """
        # Build a lookup of max confidence for each unordered pair.
        pair_conf: dict[tuple[str, str], float] = {}
        for (b, a), c in self._constraints.items():
            canon = (min(b, a), max(b, a))
            pair_conf[canon] = max(pair_conf.get(canon, 0.0), c)

        ambiguous: list[tuple[str, str]] = []
        ids = sorted(self.file_ids)
        for i, x in enumerate(ids):
            for y in ids[i + 1 :]:
                canon = (x, y)
                if pair_conf.get(canon, 0.0) < AMBIGUOUS_CEILING:
                    ambiguous.append(canon)
        return ambiguous

    def get_confidence(self, before: str, after: str) -> Optional[float]:
        """Return the confidence for a specific directed constraint, or None."""
        return self._constraints.get((before, after))

    # ------------------------------------------------------------------
    # Topological analysis (strong edges only)
    # ------------------------------------------------------------------

    def _strong_graph(self) -> tuple[dict[str, set[str]], dict[str, int]]:
        """Build adjacency list and in-degree map from strong edges.

        Returns:
            (adj, in_deg) where adj[u] = set of nodes u points to
            (u before v ⟹ v in adj[u]), and in_deg[v] = count of
            strong predecessors.
        """
        adj: dict[str, set[str]] = {fid: set() for fid in self.file_ids}
        in_deg: dict[str, int] = {fid: 0 for fid in self.file_ids}
        for (b, a), c in self._constraints.items():
            if c > STRONG_THRESHOLD:
                if a not in adj[b]:
                    adj[b].add(a)
                    in_deg[a] += 1
        return adj, in_deg

    def topological_layers(self) -> list[list[str]]:
        """Partition file IDs into layers by dependency depth.

        Uses Kahn's algorithm on the strong-edge subgraph.  Layer 0
        contains files with no strong predecessors; layer 1 contains
        files whose only strong predecessors are in layer 0; and so on.

        Files within a layer have no strong ordering between them, so
        they can be freely reordered by the decoder.

        If a cycle exists among strong edges, remaining nodes are placed
        in a final catch-all layer (the decoder should flag this).
        """
        adj, in_deg = self._strong_graph()

        layers: list[list[str]] = []
        queue = sorted(fid for fid in self.file_ids if in_deg[fid] == 0)

        visited = 0
        while queue:
            layers.append(list(queue))
            visited += len(queue)
            next_queue: list[str] = []
            for node in queue:
                for nxt in sorted(adj[node]):
                    in_deg[nxt] -= 1
                    if in_deg[nxt] == 0:
                        next_queue.append(nxt)
            queue = sorted(next_queue)

        # Remaining nodes are in a cycle — dump them in a final layer.
        if visited < len(self.file_ids):
            remaining = sorted(
                fid for fid in self.file_ids if fid not in {
                    f for layer in layers for f in layer
                }
            )
            if remaining:
                layers.append(remaining)

        return layers

    def decode_total_order(self) -> list[str]:
        """Best total order using strong constraints alone.

        Kahn's algorithm with tie-breaking by out-degree (higher
        out-degree = more things depend on this file = more important
        = should come first).
        """
        adj, in_deg = self._strong_graph()

        # out-degree in strong graph = importance
        out_deg: dict[str, int] = {fid: len(adj[fid]) for fid in self.file_ids}

        def _sort_key(fid: str) -> tuple[float, str]:
            return (-out_deg[fid], fid)  # higher out-degree first, then alpha

        queue = sorted(
            (fid for fid in self.file_ids if in_deg[fid] == 0),
            key=_sort_key,
        )
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for nxt in adj[node]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)
                    queue.sort(key=_sort_key)

        # Remaining (cycle) — append in importance order.
        if len(result) < len(self.file_ids):
            remaining = [fid for fid in self.file_ids if fid not in set(result)]
            remaining.sort(key=_sort_key)
            result.extend(remaining)

        return result

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def has_cycle(self) -> bool:
        """Return True if the strong constraints contain a cycle.

        Uses DFS colouring (WHITE/GREY/BLACK).
        """
        WHITE, GREY, BLACK = 0, 1, 2
        colour: dict[str, int] = {fid: WHITE for fid in self.file_ids}
        adj, _ = self._strong_graph()

        def _dfs(u: str) -> bool:
            colour[u] = GREY
            for v in adj[u]:
                if colour[v] == GREY:
                    return True
                if colour[v] == WHITE and _dfs(v):
                    return True
            colour[u] = BLACK
            return False

        return any(
            _dfs(fid)
            for fid in self.file_ids
            if colour[fid] == WHITE
        )

    def check_violations(
        self,
        order: list[str],
    ) -> list[tuple[str, str, float]]:
        """Return all constraints violated by *order*.

        A constraint ``(before, after, conf)`` is violated when
        ``before`` appears after ``after`` in the given order.

        Returns:
            List of ``(before, after, confidence)`` tuples for every
            violated constraint, sorted by descending confidence.
        """
        pos = {fid: i for i, fid in enumerate(order)}
        violations: list[tuple[str, str, float]] = []
        for (b, a), c in self._constraints.items():
            if b in pos and a in pos and pos[b] > pos[a]:
                violations.append((b, a, c))
        violations.sort(key=lambda t: -t[2])
        return violations

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.file_ids)

    def __repr__(self) -> str:
        n_strong = len(self.get_strong_constraints())
        return (
            f"PartialOrder({len(self.file_ids)} files, "
            f"{len(self._constraints)} constraints, "
            f"{n_strong} strong)"
        )
