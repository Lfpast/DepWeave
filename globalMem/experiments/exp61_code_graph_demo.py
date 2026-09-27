"""Exp 61 — Code-graph plugin demo: Codebase-Memory capability, PixelMem economy.

End-to-end, no network required. Shows that the new ``code_graph`` plugin set:

  1. Builds a rich symbol graph (calls / inherits / has_method / uses / imports)
     from real source via Python ``ast`` — the Codebase-Memory edge set.
  2. Derives a file-dependency graph + base-first order FROM the call edges
     (generalises DependEval's import-only dependency signal).
  3. Serves the navigation surface (search / ego / trace / deps / usage) as
     compact, token-budgeted views.
  4. Quantifies the economy: one ego view vs. dumping the whole primitive graph
     vs. the raw source it summarises.
  5. Runs through ``V5Pipeline`` end-to-end with a deterministic fake LLM, so the
     full extract -> derive -> prompt -> parse path is exercised offline.

Run:  python experiments/exp61_code_graph_demo.py [ROOT]
The real-LLM localisation eval is a separate job (see the end of this file).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelmem.v5.core.pipeline import V5Pipeline
from pixelmem.v5.core.types import TaskSpec
from pixelmem.v5.plugins.code_graph import (
    CodeGraphExtractor,
    CodeGraphIndex,
    build_code_graph_plugins,
    render_ego,
)
from pixelmem.v5.plugins.code_graph.index import approx_tokens


def load_repo(root: Path, max_files: int = 60) -> dict[str, str]:
    docs: dict[str, str] = {}
    for f in sorted(root.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        try:
            docs[str(f.relative_to(root.parent))] = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if len(docs) >= max_files:
            break
    return docs


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "pixelmem" / "v5"
    docs = load_repo(root)
    raw_source_tokens = approx_tokens("\n".join(docs.values()))

    ex = CodeGraphExtractor()
    prims = ex.extract(docs)
    idx = CodeGraphIndex.from_primitives(prims)
    stats = idx.stats()

    print("=" * 70)
    print(f"CODE GRAPH over {len(docs)} files  ({root})")
    print("=" * 70)
    print("stats:", json.dumps(stats))
    print(f"parse_errors: {ex.n_parse_errors}")

    # --- navigation demo on a real, well-connected DEFINED symbol ----------
    defined = set(idx.defs)
    hot = max((q for q in defined if idx.calls_in.get(q)),
              key=lambda q: len(idx.calls_in[q]), default=None)
    print("\n--- navigation surface (Codebase-Memory parity) ---")
    if hot:
        print(f"\n[code_ego] {hot}")
        ego = idx.ego_graph(hot, hops=1, budget_tokens=220)
        print(render_ego(ego))
        print(f"\n[code_trace out] {idx._short(hot)}:",
              json.dumps(idx.trace_call_chain(hot, 'out', 3)['chains']))
        print(f"[code_usage] {idx._short(hot)}:",
              f"{idx.locate_usage(hot)['n_sites']} sites")

    # a file-dependency view
    busy_file = max(idx.by_file, key=lambda f: len(idx.by_file[f]), default=None)
    if busy_file:
        dep = idx.find_dependencies(busy_file)
        print(f"\n[code_dependencies] {busy_file} depends on:",
              json.dumps([d['file'] for d in dep.get('depends_on', [])]))

    # --- the economy story -------------------------------------------------
    full_dump = "\n".join(f"{p.subject} {p.relation} {p.object} {p.condition}"
                          for p in prims)
    dump_tokens = approx_tokens(full_dump)
    ego_tokens = approx_tokens(render_ego(idx.ego_graph(hot))) if hot else 0
    print("\n--- economy: tokens to answer a localisation query ---")
    print(f"  raw source of {len(docs)} files ........ ~{raw_source_tokens:>7} tok")
    print(f"  full primitive-graph dump ............. ~{dump_tokens:>7} tok")
    print(f"  one derived ego view (our path) ....... ~{ego_tokens:>7} tok")
    if ego_tokens:
        print(f"  ego-view vs source  ................... {raw_source_tokens/ego_tokens:>7.0f}x smaller")
        print(f"  ego-view vs graph-dump ................ {dump_tokens/ego_tokens:>7.0f}x smaller")

    # --- full V5Pipeline path, offline (deterministic fake LLM) ------------
    print("\n--- V5Pipeline end-to-end (offline, fake LLM) ---")
    captured = {}

    def fake_llm(prompt: str):
        captured["prompt"] = prompt
        # 'locate' mode: echo the top ranked candidate name from the prompt.
        import re
        m = re.search(r"(?m)^\s*-\s*([A-Za-z_][A-Za-z0-9_]*)", prompt)
        return (m.group(1) if m else "unknown", approx_tokens(prompt), 3)

    task = TaskSpec(domain="code_localize", description="name the symbol",
                    input_schema={}, query={}, options={"mode": "locate"})
    plugins = build_code_graph_plugins(mode="locate")
    pipe = V5Pipeline(plugins, task=task, llm=fake_llm)
    question = "extract code graph primitives from documents"
    pred, pstats = pipe.run({"question": question, "mode": "locate"}, documents=docs)
    print(f"  question: {question!r}")
    print(f"  prompt tokens: {pstats.tokens_in}  (vs ~{raw_source_tokens} to dump source)")
    print(f"  parsed prediction: {pred!r}")
    assert pred is not None and not pstats.parse_error, "pipeline path failed"
    print("  OK — extract -> derive -> prompt -> parse path works")

    print("\nDepWeave's root exp56/exp58 drivers use the MCP evidence interface.")
    print("This demo only checks the standalone graph plugin with a fake caller.")


if __name__ == "__main__":
    main()
