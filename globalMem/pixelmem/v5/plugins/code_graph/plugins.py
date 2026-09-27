"""V5 PluginSet for the code graph: derivation, prompt, factory.

Two things worth highlighting, because they are exactly the levers the
competitive analysis identified as PixelMem's moat:

1. **Derivation, not just retrieval.** ``CodeGraphDerivationEngine`` lifts
   thousands of fine-grained ``calls`` edges into a small ``file_depends_on``
   graph plus a base-first ordering. This is the same trick V4 uses on imports,
   but generalised to *call*-based dependencies — a strictly richer signal than
   imports alone (it catches dynamic wiring imports miss). The declarative
   equivalent is ``CALL_FILE_DEP_RULE`` below; the engine computes it in O(edges)
   instead of the rule matcher's O(defs·calls·defs), but they agree.

2. **Workload-reducing prompt.** ``CodeNavPrompt`` never dumps source. It hands
   the LLM a ranked candidate list + token-budgeted ego views (from
   ``CodeGraphIndex``) and asks it to *confirm/select*. Same answer quality,
   a fraction of the tokens, no graph-query turns.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Optional

from pixelmem.v5.core.plugins import (
    DerivationEngine,
    DerivationRule,
    PluginSet,
    PromptTemplate,
)
from pixelmem.v5.core.types import EvidenceBundle, Primitive, TaskSpec
from pixelmem.v5.plugins.code_graph.extractor import (
    CodeGraphExtractor,
    REL_CALLS,
    REL_DEFINES,
    REL_FILE_DEPENDS_ON,
)
from pixelmem.v5.plugins.code_graph.index import (
    CodeGraphIndex,
    approx_tokens,
    render_ego,
)


# Declarative form of the call -> file-dependency derivation (see engine below).
CALL_FILE_DEP_RULE = DerivationRule(
    name="call_implies_file_dep",
    pattern=[
        ("?fa", REL_DEFINES, "?f", "?k1"),
        ("?f", REL_CALLS, "?g", "?c"),
        ("?fb", REL_DEFINES, "?g", "?k2"),
    ],
    derived=("?fa", REL_FILE_DEPENDS_ON, "?fb", "via_call"),
    confidence=0.90,
    notes="A resolved call from a symbol in file A to a symbol defined in file "
          "B makes A depend on B. Self-loops (A==B) are dropped by the engine.",
)


# ---------------------------------------------------------------------------
# Derivation engine
# ---------------------------------------------------------------------------


class CodeGraphDerivationEngine(DerivationEngine):
    """Lift the code graph to a file-dependency graph + base-first ordering."""

    def derive(
        self, primitives: list[Primitive], rules: list[DerivationRule], task: TaskSpec,
    ) -> EvidenceBundle:
        idx = CodeGraphIndex.from_primitives(primitives)

        dep: dict[str, set[str]] = {f: set() for f in idx.by_file}
        strong: list[Primitive] = []
        for f in idx.by_file:
            res = idx.find_dependencies(f)
            for row in res.get("depends_on", []):
                tgt = row["file"]
                dep[f].add(tgt)
                strong.append(Primitive(
                    f, REL_FILE_DEPENDS_ON, tgt, "via_call",
                    provenance={"via": row["via"], "n": row["n"]}))

        order = _toposort_files(list(idx.by_file), dep)

        return EvidenceBundle(
            strong=strong,
            ambiguous=[],
            raw_primitives=[],
            ordering_hint=order,
            metadata={
                "files": sorted(idx.by_file),
                "n_file_deps": len(strong),
                "graph_stats": idx.stats(),
                "has_cycle": _has_cycle(list(idx.by_file), dep),
            },
        )


def _toposort_files(nodes: list[str], dep: dict[str, set[str]]) -> list[str]:
    """Base-first order: if A depends_on B, B precedes A. Stable; cycle-tolerant."""
    nodeset = set(nodes)
    dependents: dict[str, set[str]] = defaultdict(set)
    indeg: dict[str, int] = {}
    for f in nodes:
        deps = {g for g in dep.get(f, set()) if g in nodeset and g != f}
        indeg[f] = len(deps)
        for g in deps:
            dependents[g].add(f)

    ready = sorted(f for f in nodes if indeg[f] == 0)
    order: list[str] = []
    seen: set[str] = set()
    while ready:
        f = ready.pop(0)
        if f in seen:
            continue
        seen.add(f)
        order.append(f)
        newly = []
        for d in sorted(dependents.get(f, set())):
            indeg[d] -= 1
            if indeg[d] <= 0 and d not in seen:
                newly.append(d)
        ready = sorted(set(ready) | set(newly))
    # Append any nodes left in a cycle, stably.
    for f in nodes:
        if f not in seen:
            order.append(f)
    return order


def _has_cycle(nodes: list[str], dep: dict[str, set[str]]) -> bool:
    return len(_toposort_files(nodes, dep)) != len(set(nodes)) or any(
        f in dep.get(g, set()) and g in dep.get(f, set())
        for f in nodes for g in dep.get(f, set()))


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


class CodeNavPrompt(PromptTemplate):
    """Compact, workload-reducing prompt over the code graph.

    Modes (``task.options['mode']`` or ``query_input['mode']``):

    * ``"order"``  — base-first file dependency ordering (DependEval-style).
    * ``"locate"`` — given an NL description, name the target symbol/file
      (RepoQA / SWE-bench-localization-style).
    * ``"qa"``     — free-form question answered from ego views (default).
    """

    def __init__(self, default_mode: str = "qa", top_k: int = 8) -> None:
        self._default_mode = default_mode
        self._top_k = top_k

    # -- build --------------------------------------------------------------
    def build(self, task: TaskSpec, query_input: dict, evidence: EvidenceBundle) -> str:
        mode = query_input.get("mode") or task.options.get("mode") or self._default_mode
        budget = int(task.options.get("prompt_token_budget", 600))
        idx = CodeGraphIndex.from_primitives(evidence.raw_primitives)

        if mode == "order":
            return self._build_order(idx, evidence)
        return self._build_locate_or_qa(idx, query_input, evidence, mode, budget)

    def _build_order(self, idx: CodeGraphIndex, evidence: EvidenceBundle) -> str:
        order = evidence.ordering_hint or sorted(idx.by_file)
        names = [f.split("/")[-1] for f in order]
        dep_lines = []
        for p in evidence.strong:
            dep_lines.append(f"  {p.subject.split('/')[-1]} depends on "
                             f"{p.object.split('/')[-1]} (via {', '.join((p.provenance or {}).get('via', [])[:3])})")
        dep_text = "\n".join(dep_lines) or "  (none detected)"
        return (
            "You are given Python source files. Return a dependency ordering "
            "where base files (depended on by others) come first.\n\n"
            f"Confirmed call-based dependencies:\n{dep_text}\n\n"
            f"Computed base-first order: {json.dumps(names)}\n\n"
            "Is this correct? Fix if needed. Return ONLY a JSON array of filenames."
        )

    def _build_locate_or_qa(
        self, idx: CodeGraphIndex, query_input: dict, evidence: EvidenceBundle,
        mode: str, budget: int,
    ) -> str:
        question = (query_input.get("question") or query_input.get("description")
                    or query_input.get("query") or "").strip()
        focus = query_input.get("focus") or []
        if not focus and question:
            focus = [c["qual"] for c in idx.search_symbol(question, self._top_k)]

        # Token-budgeted ego views for the ranked candidates.
        blocks: list[str] = []
        used = approx_tokens(question) + 80
        per = max(120, budget // max(1, min(len(focus), 6)))
        for qual in focus[:6]:
            ego = render_ego(idx.ego_graph(qual, hops=1, budget_tokens=per))
            if used + approx_tokens(ego) > budget:
                break
            blocks.append(ego)
            used += approx_tokens(ego)

        candidate_list = "\n".join(
            f"  - {c['name']}{c['signature']}  [{c['file']}]"
            for c in idx.search_symbol(question, self._top_k)) if question else ""

        head = ("You are navigating a code graph (no raw source shown — only "
                "derived structure). Use the candidates and their neighbourhoods "
                "to answer.\n")
        ask = ("Name the single function/class that best matches the description. "
               "Return ONLY its name."
               if mode == "locate" else "Answer the question concisely.")
        return (
            f"{head}\n"
            f"Question: {question}\n\n"
            f"Ranked candidates:\n{candidate_list or '  (none)'}\n\n"
            f"Neighbourhoods:\n" + "\n".join(blocks) + "\n\n"
            f"{ask}"
        )

    # -- parse --------------------------------------------------------------
    def parse(self, completion: str, task: TaskSpec) -> Any:
        mode = task.options.get("mode") or self._default_mode
        if mode == "order":
            m = re.search(r"\[.*\]", completion, re.DOTALL)
            if not m:
                raise ValueError("no JSON array found")
            items = json.loads(m.group(0))
            if not isinstance(items, list):
                raise ValueError("not a list")
            return [_basename(str(x)) for x in items]
        # locate / qa: prefer a bare identifier, else the stripped text.
        text = completion.strip()
        if mode == "locate":
            m = re.search(r"[A-Za-z_][A-Za-z0-9_\.]*", text)
            return m.group(0) if m else text
        return text


def _basename(x: str) -> str:
    m = re.search(r"([A-Za-z0-9_\-]+\.[A-Za-z0-9_]+)", x)
    return m.group(1) if m else x.split("/")[-1]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_code_graph_plugins(
    language: str = "python", mode: str = "qa", top_k: int = 8,
) -> PluginSet:
    """A complete code-graph PluginSet, ready for ``V5Pipeline``.

    Example::

        from pixelmem.v5 import V5Pipeline, TaskSpec
        from pixelmem.v5.plugins.code_graph import build_code_graph_plugins

        plugins = build_code_graph_plugins(mode="locate")
        pipe = V5Pipeline(plugins, task=TaskSpec(...), llm=my_llm)
        name, stats = pipe.run({"question": "the fn that retries on 429"},
                               documents={"client.py": SRC, ...})
    """
    return PluginSet(
        name=f"code_graph_{mode}",
        extractor=CodeGraphExtractor(language=language),
        resolver=None,
        derivation_rules=[CALL_FILE_DEP_RULE],
        derivation_engine=CodeGraphDerivationEngine(),
        prompt_template=CodeNavPrompt(default_mode=mode, top_k=top_k),
    )


__all__ = [
    "CodeGraphDerivationEngine",
    "CodeNavPrompt",
    "CALL_FILE_DEP_RULE",
    "build_code_graph_plugins",
]
