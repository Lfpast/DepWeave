"""TaskCard — the JSON/YAML spec that describes one benchmark task.

A TaskCard is the single artifact the synthesis loop reads. It declares:

- what the domain is (free-text + stable domain key),
- what a valid input looks like,
- what the LLM should output,
- a small handful of few-shot examples,
- a held-out set to evaluate against,
- the eval metric and accuracy threshold.

Example file::

    domain: python_dependency_ordering
    description: Order a small set of Python files by dependency.
    input_schema:
      kind: file_set
      files: list[str]
      file_contents: dict[str, str]
    query:
      kind: ordering
      output: list[str]
    eval:
      metric: exact_match
      threshold: 0.80
    few_shot:
      - input: {...}
        expected_output: [...]
    holdout:
      - ...
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from pixelmem.v5.core.types import Example, TaskSpec


def _load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parse_document(text: str, path: str | Path) -> dict:
    p = str(path)
    if p.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as err:
            raise RuntimeError(
                "PyYAML is required to load YAML task cards; "
                "install pyyaml or convert to .json"
            ) from err
        return yaml.safe_load(text)
    return json.loads(text)


def _example_from_dict(d: dict) -> Example:
    return Example(
        input=d.get("input", {}),
        expected_output=d.get("expected_output"),
        qid=d.get("qid"),
    )


class TaskCard:
    """Parsed TaskCard. Prefer the classmethod loaders over direct construction."""

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "TaskCard":
        return cls.from_dict(_parse_document(_load_text(path), path))

    @classmethod
    def from_dict(cls, d: dict) -> "TaskCard":
        missing = {"domain", "description", "input_schema", "query"} - set(d)
        if missing:
            raise ValueError(f"TaskCard missing required keys: {sorted(missing)}")

        eval_block = d.get("eval", {}) or {}
        few_shot = [_example_from_dict(e) for e in d.get("few_shot", [])]
        holdout = [_example_from_dict(e) for e in d.get("holdout", [])]

        spec = TaskSpec(
            domain=str(d["domain"]),
            description=str(d["description"]),
            input_schema=dict(d["input_schema"]),
            query=dict(d["query"]),
            eval_metric=str(eval_block.get("metric", "exact_match")),
            eval_threshold=float(eval_block.get("threshold", 0.80)),
            few_shot=few_shot,
            holdout=holdout,
            options=dict(d.get("options", {}) or {}),
        )
        return cls(spec)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def domain(self) -> str:
        return self.spec.domain

    def split(self) -> tuple[list[Example], list[Example]]:
        """Return (few_shot, holdout)."""
        return list(self.spec.few_shot), list(self.spec.holdout)

    def to_dict(self) -> dict:
        """Dump as a round-trippable dict (few_shot/holdout lose qid=None)."""
        d = {
            "domain": self.spec.domain,
            "description": self.spec.description,
            "input_schema": self.spec.input_schema,
            "query": self.spec.query,
            "eval": {
                "metric": self.spec.eval_metric,
                "threshold": self.spec.eval_threshold,
            },
            "few_shot": [asdict(e) for e in self.spec.few_shot],
            "holdout": [asdict(e) for e in self.spec.holdout],
            "options": self.spec.options,
        }
        return d


__all__ = ["TaskCard"]
