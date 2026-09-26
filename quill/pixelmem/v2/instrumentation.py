"""Instrumentation — metrics collection for comparing old vs new pipelines.

Tracks: latency, token costs, cache hits, extraction method distribution,
retrieval accuracy by query mode.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class MetricEvent:
    timestamp: float = 0.0
    operation: str = ""  # "read" or "write"
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_hit: bool = False
    extraction_method: str = ""  # "fast", "llm", "skip"
    query_mode: str = ""
    content_lane: str = ""
    n_facts: int = 0
    n_entities_scanned: int = 0
    n_shards_touched: int = 0
    n_tool_calls: int = 0
    confidence: float = 0.0


class Instrumentor:
    """Collects metrics events and produces aggregated reports."""

    def __init__(self):
        self._events: list[MetricEvent] = []

    def record(self, event: MetricEvent) -> None:
        if event.timestamp == 0:
            event.timestamp = time.time()
        self._events.append(event)

    @contextmanager
    def timed(self, operation: str, **kwargs):
        """Context manager that auto-records latency.

        Usage:
            with instrumentor.timed("read", query_mode="fact_lookup") as event:
                # ... do work ...
                event.n_facts = 42
                event.tokens_in = 500
        """
        event = MetricEvent(operation=operation, **kwargs)
        event.timestamp = time.time()
        t0 = time.perf_counter()
        try:
            yield event
        finally:
            event.latency_ms = (time.perf_counter() - t0) * 1000
            self._events.append(event)

    def summary(self) -> dict:
        """Aggregated stats across all events."""
        if not self._events:
            return {"n_events": 0}

        reads = [e for e in self._events if e.operation == "read"]
        writes = [e for e in self._events if e.operation == "write"]

        def _avg(events, attr):
            vals = [getattr(e, attr) for e in events]
            return round(sum(vals) / max(1, len(vals)), 2) if vals else 0

        # Cache stats
        cache_hits = sum(1 for e in reads if e.cache_hit)

        # Extraction method distribution
        method_counts = defaultdict(int)
        for e in writes:
            method_counts[e.extraction_method] += 1

        # Accuracy by query mode
        mode_counts = defaultdict(int)
        for e in reads:
            mode_counts[e.query_mode] += 1

        return {
            "n_events": len(self._events),
            "reads": {
                "count": len(reads),
                "avg_latency_ms": _avg(reads, "latency_ms"),
                "avg_tokens_in": _avg(reads, "tokens_in"),
                "avg_tokens_out": _avg(reads, "tokens_out"),
                "avg_facts": _avg(reads, "n_facts"),
                "avg_entities_scanned": _avg(reads, "n_entities_scanned"),
                "avg_tool_calls": _avg(reads, "n_tool_calls"),
                "cache_hit_rate": round(cache_hits / max(1, len(reads)) * 100, 1),
            },
            "writes": {
                "count": len(writes),
                "avg_latency_ms": _avg(writes, "latency_ms"),
                "avg_tokens_in": _avg(writes, "tokens_in"),
                "extraction_methods": dict(method_counts),
                "avg_confidence": _avg(writes, "confidence"),
            },
            "query_modes": dict(mode_counts),
        }

    def export_jsonl(self, path: str | Path) -> None:
        """Write events as JSONL."""
        with open(path, "w") as f:
            for e in self._events:
                f.write(json.dumps(asdict(e)) + "\n")

    def export_csv(self, path: str | Path) -> None:
        """Write events as CSV."""
        if not self._events:
            return
        fields = list(asdict(self._events[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for e in self._events:
                writer.writerow(asdict(e))

    def clear(self) -> None:
        self._events.clear()
