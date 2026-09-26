"""V5Pipeline — the plugin host.

Pipeline stages::

    documents ─► extract ─► resolve ─► derive ─► build_prompt
                                                       │
                                                       ▼
                                                   LLMCaller
                                                       │
                                                       ▼
                                                     parse ─► prediction + stats

The core imports nothing from V4. All V4 coupling (if any) is inside the
PluginSet the caller constructs.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from quill.core.derivation import DefaultDerivationEngine
from quill.core.plugins import (
    DerivationEngine,
    LLMCaller,
    PluginSet,
)
from quill.core.types import (
    EvidenceBundle,
    PipelineStats,
    Primitive,
    TaskSpec,
)


class V5Pipeline:
    """Stage-wise pipeline host.

    Usage::

        pipeline = V5Pipeline(plugins=my_plugins, task=my_task_spec, llm=my_llm)
        prediction, stats = pipeline.run(query_input={"files": [...]})

    The pipeline is stateful across ``run()`` calls: primitives extracted
    once can be reused across queries by passing ``reuse_index=True`` — this
    is how benchmarks amortize extraction cost across many questions per
    document set.
    """

    def __init__(
        self,
        plugins: PluginSet,
        task: TaskSpec,
        llm: LLMCaller,
        debug: bool = False,
    ) -> None:
        plugins.validate()
        self._plugins = plugins
        self._task = task
        self._llm = llm
        self._debug = debug

        self._engine: DerivationEngine = (
            plugins.derivation_engine or DefaultDerivationEngine()
        )

        # Cached state (populated by index())
        self._documents: Optional[dict[str, str]] = None
        self._primitives: Optional[list[Primitive]] = None
        self._evidence: Optional[EvidenceBundle] = None

    # ------------------------------------------------------------------
    # Indexing — extract + resolve + derive, cached
    # ------------------------------------------------------------------

    def index(self, documents: dict[str, str], **extract_kwargs: Any) -> None:
        """Extract + resolve + derive; cache results for subsequent queries."""
        ex = self._plugins.extractor
        primitives = list(ex.extract(documents, **extract_kwargs))

        if self._plugins.resolver is not None:
            primitives = list(self._plugins.resolver.resolve(primitives, documents))

        evidence = self._engine.derive(
            primitives, self._plugins.derivation_rules, self._task,
        )

        self._documents = documents
        self._primitives = primitives
        self._evidence = evidence

        if self._debug:
            print(
                f"[V5] indexed: {len(primitives)} primitives, "
                f"{len(evidence.strong)} strong, "
                f"{len(evidence.ambiguous)} ambiguous"
            )

    # ------------------------------------------------------------------
    # Run — index (if needed) + prompt + LLM + parse
    # ------------------------------------------------------------------

    def run(
        self,
        query_input: dict,
        documents: Optional[dict[str, str]] = None,
        reuse_index: bool = False,
    ) -> tuple[Any, PipelineStats]:
        """Execute one query.

        Args:
            query_input: task-specific query payload (e.g., ``{"files": [...]}``).
            documents: document set to index. Required unless ``reuse_index``.
            reuse_index: if True, reuse primitives/evidence from a prior
                ``index()`` or ``run()`` call on the same pipeline instance.
        """
        start = time.perf_counter()

        if not reuse_index:
            if documents is None:
                raise ValueError(
                    "documents must be provided when reuse_index=False"
                )
            self.index(documents)

        assert self._primitives is not None and self._evidence is not None, (
            "Pipeline must be indexed before run()"
        )

        # Let the prompt template attach raw primitives if it wants them;
        # derivation returns an empty raw_primitives list by convention.
        evidence = EvidenceBundle(
            strong=list(self._evidence.strong),
            ambiguous=list(self._evidence.ambiguous),
            raw_primitives=list(self._primitives),
            ordering_hint=list(self._evidence.ordering_hint),
            metadata=dict(self._evidence.metadata),
        )

        prompt = self._plugins.prompt_template.build(
            self._task, query_input, evidence,
        )

        completion, t_in, t_out = self._llm(prompt)

        parse_error = False
        try:
            prediction = self._plugins.prompt_template.parse(completion, self._task)
        except (ValueError, KeyError, TypeError, IndexError):
            prediction = None
            parse_error = True

        stats = PipelineStats(
            n_primitives=len(self._primitives),
            n_strong_edges=len(evidence.strong),
            n_ambiguous_edges=len(evidence.ambiguous),
            tokens_in=t_in,
            tokens_out=t_out,
            llm_calls=1,
            parse_error=parse_error,
            wallclock_s=time.perf_counter() - start,
        )

        return prediction, stats

    # ------------------------------------------------------------------
    # Introspection (used by the test harness)
    # ------------------------------------------------------------------

    @property
    def primitives(self) -> list[Primitive]:
        if self._primitives is None:
            return []
        return list(self._primitives)

    @property
    def evidence(self) -> Optional[EvidenceBundle]:
        return self._evidence

    @property
    def plugins(self) -> PluginSet:
        return self._plugins


__all__ = ["V5Pipeline"]
