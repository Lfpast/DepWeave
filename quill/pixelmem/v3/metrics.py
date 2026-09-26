"""Structured metrics collection for PixelMem V3.

Provides fine-grained instrumentation for reads, writes, and operator
executions.  Supports JSONL and CSV export for offline analysis.

Typical usage::

    from pixelmem.v3.metrics import Metrics

    metrics = Metrics()

    with metrics.timed("read", query_mode="entity") as ev:
        result = pipeline.query("Where does Alice work?")
        ev.n_facts = len(result.evidences)
        ev.cache_hit = False

    print(metrics.summary())
"""

from __future__ import annotations

import csv
import io
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Generator, Optional


# ---------------------------------------------------------------------------
# MetricEvent
# ---------------------------------------------------------------------------

@dataclass
class MetricEvent:
    """A single instrumentation event.

    Attributes:
        timestamp: Unix timestamp when the event was recorded.
        operation: Operation type (``"read"``, ``"write"``, ``"cache_lookup"``,
            ``"operator"``, etc.).
        latency_ms: Wall-clock latency in milliseconds.
        tokens_in: Prompt / input tokens consumed (0 if non-LLM).
        tokens_out: Completion / output tokens consumed.
        cache_hit: Whether the result came from cache.
        plan_id: Identifier of the retrieval plan (empty if N/A).
        query_mode: Query mode string (``"entity"``, ``"relational"``, etc.).
        content_lane: Content lane for write operations.
        extraction_method: Extraction method used (``"fast"``, ``"cli"``,
            ``"ast"``, etc.).
        n_facts: Number of facts in the result / bundle.
        n_operators: Number of algebra operators executed.
        n_entities_scanned: Number of distinct entities scanned.
        n_shards_touched: Number of shards accessed.
        confidence: Aggregate confidence score.
        budget_used_pct: Percentage of token budget consumed by the answer.
    """

    timestamp: float = 0.0
    operation: str = ""
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_hit: bool = False
    plan_id: str = ""
    query_mode: str = ""
    content_lane: str = ""
    extraction_method: str = ""
    n_facts: int = 0
    n_operators: int = 0
    n_entities_scanned: int = 0
    n_shards_touched: int = 0
    confidence: float = 0.0
    budget_used_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class Metrics:
    """Structured metrics accumulator for PixelMem V3.

    Thread-safe for single-writer workloads (the common case for
    PixelMem).  For multi-threaded use, wrap mutation methods with a lock.
    """

    def __init__(self) -> None:
        self._events: list[MetricEvent] = []

        # Running aggregates for fast summary()
        self._op_counts: dict[str, int] = defaultdict(int)
        self._op_latency: dict[str, float] = defaultdict(float)
        self._op_facts: dict[str, int] = defaultdict(int)
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, event: MetricEvent) -> None:
        """Append a fully-populated MetricEvent.

        Also updates running aggregates.
        """
        if event.timestamp == 0.0:
            event.timestamp = time.time()

        self._events.append(event)

        op = event.operation
        self._op_counts[op] += 1
        self._op_latency[op] += event.latency_ms
        self._op_facts[op] += event.n_facts

        if event.cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        self._total_tokens_in += event.tokens_in
        self._total_tokens_out += event.tokens_out

    @contextmanager
    def timed(
        self, operation: str, **kwargs: Any
    ) -> Generator[MetricEvent, None, None]:
        """Context manager that times a block and records a MetricEvent.

        The yielded event can be mutated inside the ``with`` block to
        attach result-dependent fields (e.g. ``n_facts``, ``cache_hit``).

        Parameters
        ----------
        operation : str
            The operation name (``"read"``, ``"write"``, etc.).
        **kwargs
            Additional fields forwarded to the MetricEvent constructor.

        Yields
        ------
        MetricEvent
            The event being built.  Modify in-place before the block exits.
        """
        event = MetricEvent(operation=operation, **kwargs)
        t0 = time.monotonic()
        try:
            yield event
        finally:
            event.latency_ms = round((time.monotonic() - t0) * 1000, 3)
            self.record(event)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return aggregated statistics across all recorded events.

        Returns
        -------
        dict
            Keys: ``total_events``, ``operations`` (per-op breakdown),
            ``cache_hit_rate``, ``total_tokens_in``, ``total_tokens_out``,
            ``avg_latency_ms``.
        """
        total = len(self._events)
        total_latency = sum(self._op_latency.values())
        cache_total = self._cache_hits + self._cache_misses

        per_op: dict[str, dict[str, Any]] = {}
        for op in self._op_counts:
            count = self._op_counts[op]
            latency = self._op_latency[op]
            per_op[op] = {
                "count": count,
                "total_latency_ms": round(latency, 2),
                "avg_latency_ms": round(latency / max(1, count), 2),
                "total_facts": self._op_facts[op],
            }

        return {
            "total_events": total,
            "operations": per_op,
            "cache_hit_rate": round(
                self._cache_hits / max(1, cache_total), 4
            ),
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
            "avg_latency_ms": round(total_latency / max(1, total), 2),
        }

    def operator_stats(self) -> dict[str, Any]:
        """Return per-operator statistics.

        Filters events to those with ``operation == "operator"`` and
        groups by ``plan_id``.

        Returns
        -------
        dict
            Keyed by plan_id with counts, latencies, and facts.
        """
        by_plan: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_latency_ms": 0.0, "total_facts": 0}
        )
        for ev in self._events:
            if ev.operation == "operator" and ev.plan_id:
                entry = by_plan[ev.plan_id]
                entry["count"] += 1
                entry["total_latency_ms"] += ev.latency_ms
                entry["total_facts"] += ev.n_facts

        # Round latencies
        result: dict[str, Any] = {}
        for plan_id, entry in by_plan.items():
            result[plan_id] = {
                "count": entry["count"],
                "total_latency_ms": round(entry["total_latency_ms"], 2),
                "avg_latency_ms": round(
                    entry["total_latency_ms"] / max(1, entry["count"]), 2
                ),
                "total_facts": entry["total_facts"],
            }
        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_jsonl(self, path: str | Path) -> None:
        """Write all events to a JSONL file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False))
                f.write("\n")

    def export_csv(self, path: str | Path) -> None:
        """Write all events to a CSV file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        field_names = [f.name for f in fields(MetricEvent)]

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for event in self._events:
                writer.writerow(event.to_dict())

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all events and aggregates."""
        self._events.clear()
        self._op_counts.clear()
        self._op_latency.clear()
        self._op_facts.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return f"Metrics(events={len(self._events)})"
