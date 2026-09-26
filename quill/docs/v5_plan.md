# PixelMem V5 Plan — From Hand-Written Tool to Synthesized Tool

> V4 gave the LLM a tool. V5 teaches the LLM to build the tool.

## 1. Why V5

V4 is specialized to one task: **Python file dependency ordering on DependEval**. Every stage — AST-based primitive extraction, Python import resolution, `imports_symbol + defined_in → depends_on` chain, and the hybrid "imports + confirmed edges + topo order" prompt — was hand-written for that task. The LLM only *uses* the pipeline; it never designs or validates one.

To generalize PixelMem we need the LLM to:

1. **Design** a primitive-quadruple schema for a new domain.
2. **Develop** the extractor / resolver / graph-chain / prompt template.
3. **Test** the pipeline on held-out examples and diagnose failures.
4. **Refine** the pipeline from failure traces until accuracy plateaus.

V5 is the skeleton + guidelines that let an LLM do all four. The output of V5 is still a V4-compatible pipeline (same PNG pixel storage, same `RetrievalPipeline` contract) — only the components inside are machine-generated for the target domain.

## 2. What's Rigid in V4 (and must become pluggable)

| V4 component | V4 coupling | V5 abstraction |
|---|---|---|
| `primitive_extractor.py` | Hard-coded `ast.parse`; Python-only relation vocab (`imports_symbol`, `extends`, `calls`) | `Extractor` plugin: `(documents, namespace) → list[Triple]` |
| `symbol_resolver.py` | Python stdlib set, `__init__.py` re-export tracing | `Resolver` plugin: `(primitives, context) → corrected primitives` |
| `dependency_graph.py` | Chain rules literally written for imports | `Derivation` plugin: list of `(chain_pattern, derived_relation, confidence)` |
| `natural_labels.py` | File-basename disambiguation only | `LabelStrategy` plugin: entity → human-readable label |
| `_build_structural_prompt` | Hard-coded template for ordering | `PromptTemplate` plugin per `QueryMode` |
| `QueryMode` enum | 5 code-shaped modes | Free-form `query_spec` with typed inputs/outputs |

Every arrow above is an LLM-writable plugin in V5.

## 3. V5 Architecture

```
  TASK CARD (YAML/JSON)
  ├─ domain: "python dependency ordering" | "legal contracts" | "medical notes" | ...
  ├─ input: {kind: "file_set" | "doc_set" | "dialogue", schema: ...}
  ├─ query: {kind: "ordering" | "lookup" | "classification" | "qa", output_type: ...}
  ├─ few_shot: [ {input, expected_output}, ... ]
  └─ eval: {metric: "exact_match" | "f1" | "fuzzy", threshold: 0.80}
                 │
                 ▼
  ┌────────── SYNTHESIS LOOP ──────────┐
  │  1. Schema Designer  (LLM)         │  → relation vocab + condition channels + entity typing
  │  2. Extractor Synth  (LLM → code)  │  → writes Extractor plugin (sandboxed exec)
  │  3. Resolver Synth   (LLM → code)  │  → writes Resolver plugin (optional)
  │  4. Derivation Synth (LLM → rules) │  → writes chain rules
  │  5. Prompt Synth     (LLM → tmpl)  │  → writes PromptTemplate for the query
  │  6. Test Harness                    │  → runs synthesized pipeline on few-shot + holdout
  │  7. Refinement       (LLM)         │  → reads failure diagnostics, patches any stage
  └────────────────────────────────────┘
                 │ accuracy ≥ threshold OR budget exhausted
                 ▼
  FROZEN PIPELINE (V4-compatible RetrievalPipeline subclass)
  ├─ runs on pixel matrix storage (unchanged)
  └─ ships with synthesis audit log (which LLM call produced which plugin)
```

## 4. The Synthesis Loop — Stage by Stage

### Stage 1. Schema Designer

**Prompt contract:** given a Task Card + few-shot examples, emit:

```json
{
  "relations":   ["imports_symbol", "extends", "calls", ...],
  "conditions":  ["repo_level", "symbol_level", "internal", "external"],
  "entity_kinds": {"file": "path", "symbol": "name@file", "module": "dotted"},
  "query_output_schema": {"type": "list<file>"}
}
```

Guidelines the LLM must follow (baked into the system prompt):
- **Keep the vocabulary small** — the pixel-color palette has 24 maximally-distinct RGBs per shard. More than ~20 relations = split shards.
- **Separate fact channels via condition** — don't invent `imports_symbol_internal` + `imports_symbol_external`; use relation `imports_symbol` with two conditions.
- **All derivable facts MUST be reachable by chaining primitives** — if no chain produces `depends_on`, add the primitives that allow it.

### Stage 2. Extractor Synthesis

LLM writes a Python module implementing:

```python
def extract_primitives(documents, namespace, **kwargs) -> list[Triple]: ...
```

Guidelines:
- **One parser per document kind** — AST for code, `.sentence_split()` for text, turn loop for dialogue.
- **Every subject/object must be a canonical alias** from `AliasNamespace` — never a raw path or free-form string.
- **Emit at minimum: containment + reference** — `(container, contains_x, item, kind)` + `(item_a, refers_to, item_b, channel)` is the floor for any domain.
- **Instrument cost** — return `{n_docs, n_primitives, parse_errors}` alongside the triples.

The synthesized module is executed in a restricted subprocess (no network, bounded CPU/memory). Syntax errors or sandbox violations are returned as LLM-readable errors, triggering Stage 7 refinement.

### Stage 3. Resolver Synthesis (optional)

Only runs if the Task Card flags `disambiguation: true`. LLM writes filters for domain-specific collisions, e.g.:
- code: `from types import X` where `types` is stdlib, not local `types.py`
- legal: "Article 5" in jurisdiction A vs. jurisdiction B
- medical: ICD-10 code overlap between siblings

Guidelines:
- Filters are **subtractive only** — they remove suspected-false primitives; they never invent new ones.
- Each filter must carry a `reason` string that gets logged and shown in failure diagnostics.

### Stage 4. Derivation Synthesis

LLM emits a list of chain rules:

```python
[
  ChainRule(
    pattern=[(X, "imports_symbol", Y), (Y, "defined_in", Z)],
    derived=(X, "depends_on", Z),
    confidence=0.90,
  ),
  ChainRule(
    pattern=[(X, "extends", Y), (Y, "defined_in", Z)],
    derived=(X, "depends_on", Z),
    confidence=0.80,
  ),
]
```

Guidelines:
- **Confidence ≥ 0.70** = strong edge shown to LLM; **< 0.70** = ambiguous, shown only if no strong evidence exists.
- **Chains are deterministic** — no LLM in the loop at derivation time (keeps per-query cost at zero marginal).
- Unit test: derivation on few-shot must yield ≥ 1 strong edge per ground-truth relation that any question actually requires.

### Stage 5. Prompt Template Synthesis

LLM writes a template builder:

```python
def build_prompt(query, evidence_objects) -> str: ...
```

Guidelines the prompt must obey:
- **Hybrid by default** — include both raw primitives (for edges the graph missed) and derived evidence (for edges already resolved). V4 ablation: hybrid 81% vs raw-only 63% vs structural-only 60%; don't skip either side without an ablation run.
- **No internal aliases** — pass every entity through `NaturalLabeler` first.
- **Bounded tokens** — prompt must fit in a budget set by the Task Card (default 500 tokens).
- **Instruction + expected output format** at the bottom, not the top (keeps it out of KV-cache-invalidating position).

### Stage 6. Test Harness

Fixed — not LLM-generated. Runs:

1. All few-shot examples (to guard against the LLM overfitting on them).
2. Held-out examples split from the Task Card (70/30 or user-specified).
3. Per-case diagnostics:
   - `n_primitives`, `n_derived_strong`, `n_derived_ambiguous`
   - predicted vs. expected output
   - failure category classifier: `missing_primitive | wrong_chain | prompt_parse_fail | llm_logic_error | true_ambiguity`
4. Emits a machine-readable failure log + a human-readable markdown report.

### Stage 7. Refinement

LLM reads the failure log + current plugin sources and decides **which one stage to patch**:

- `missing_primitive` → patch Stage 2 extractor.
- `wrong_chain` → patch Stage 4 derivation.
- `prompt_parse_fail` or `llm_logic_error` → patch Stage 5 prompt.
- `true_ambiguity` → report as unfixable (stops eating budget on impossible cases).

Guidelines:
- **Minimum necessary change** — edit a single plugin; leave the others untouched. A refinement that touches 3 stages usually means the schema is wrong → bounce back to Stage 1.
- **Regression test first** — re-run passing few-shot after patching; any regression blocks the patch.
- **Budget-aware** — max N=5 refinement iterations per Task Card, then freeze with best-so-far.

## 5. Implementation Phases

| Phase | Scope | Deliverable | Effort |
|---|---|---|---|
| **P1. Refactor V4 to plugin interfaces** | Extract `Extractor`, `Resolver`, `Derivation`, `PromptTemplate` protocols; keep V4's current plugins as default impls; V4 must still pass `tests/test_v4.py` | `pixelmem/v5/core/` + default `plugins/python_deps/` | 1 week |
| **P2. Task Card format + Test Harness** | Schema, loader, evaluator, failure classifier | `pixelmem/v5/task_card.py`, `pixelmem/v5/harness.py`, `tests/test_v5_harness.py` | 3–4 days |
| **P3. Template-driven synthesis** | LLM fills slots in pre-written plugin templates (safest; narrowest blast radius) | `pixelmem/v5/synth/templates/` + 1 new domain (e.g., legal clauses) | 1 week |
| **P4. Free-form synthesis + sandbox** | LLM writes whole plugin modules; executed under `seccomp`/subprocess with no net + rlimits | `pixelmem/v5/synth/sandbox.py` | 1–2 weeks |
| **P5. Refinement loop** | Failure-log → patch → re-test; with regression guard + budget | `pixelmem/v5/synth/refine.py` | 1 week |
| **P6. Cross-domain benchmarks** | Run on §6 benchmarks; report synthesis-time cost + query-time cost | `experiments/exp22_v5_*.py`, `docs/v5_results.md` | ongoing |

**Milestone exit criteria for P1 alone**: V4 refactored, DependEval accuracy unchanged at 81% (no regression), `tests/test_v4.py` passes.

## 6. Benchmarks V5 Should Prove Itself On

These fall into three buckets: (a) extensions of V4's code-dep story, (b) graph-shaped memory tasks, (c) genuinely different domains that stress-test the synthesis loop.

### 6a. Code-Dep Extensions (same shape, different data)

| Benchmark | Why | Availability |
|---|---|---|
| **DependEval Task 1 (File Identification)** | Same data we already have; different query ("which file fixes this issue?"). Tests V5's query-mode generality with zero schema change. | Local — just need Task 1 split from the existing JSON |
| **DependEval Java / other-lang splits** | Swap AST extractor plugin; rest should stay identical. Direct test of extractor pluggability. | HF `LorryML/DependEval` |
| **CrossCodeEval** (Ding et al., NeurIPS 2023) | Cross-file code completion — the graph tells the LLM which other file defines the needed symbol. | HF `microsoft/CrossCodeEval` |
| **RepoBench v1.1** | Cross-file line/API prediction; graph retrieval as the RAG. | HF `tianyang/repobench_python_v1.1` |
| **RepoEval** (Zhang et al., ICML 2023) | Repo-level completion, three granularities (line/API/function). | GitHub `microsoft/CodeT` |
| **SWE-bench Lite** | Given issue + repo, produce patch. Ambitious — tests whether graph-guided retrieval beats full-file RAG for bug localization. | HF `princeton-nlp/SWE-bench_Lite` |

### 6b. Graph-Shaped Memory (tests derivation + chains)

| Benchmark | Why |
|---|---|
| **MuSiQue** | 2–4 hop QA; every hop is a derivation chain. |
| **HotpotQA (distractor)** | 2-hop QA with supporting facts — ideal for checking primitive quality. |
| **MetaQA 1/2/3-hop** | Synthetic KB QA; controls chain depth cleanly. |
| **WebQSP / CWQ** | Freebase QA — mid-size KG, realistic noise. |
| **LoCoMo** | Long conversation memory (~200 turns). V5 must synthesize a dialogue-turn extractor. |

### 6c. Genuinely Different Domains (stress-test synthesis loop)

| Benchmark | Domain | What it forces V5 to synthesize |
|---|---|---|
| **CUAD** | Legal contracts — 41 clause categories | Clause-level extractor, jurisdictional resolver |
| **PubMedQA / BioASQ** | Biomedical QA over abstracts | Entity-linking extractor to UMLS codes |
| **FinQA / TAT-QA** | Financial QA over tables + text | Cell-level primitives + arithmetic derivation chains |
| **LongMemEval** (already done on V1) | Conversational memory | Now re-run *through V5 synthesis* and compare against V1's hand-tuned pipeline — the key "does automation match hand-tuning?" baseline |

**Recommended first three**: DependEval Task 1 (trivial plug-in), CrossCodeEval (new query shape), LongMemEval (non-code domain, direct V1 comparison).

## 7. Success Criteria for V5

- **Zero regression on V4**: DependEval Task 2 with default plugins stays at 81.3% / 389 tok.
- **Automation parity**: V5-synthesized pipeline for LongMemEval matches V1 hand-tuned accuracy (67% fuzzy) within ±5pp, or identifies a failure mode V1 didn't.
- **Cross-lang generalization**: DependEval Java ≥ 70% using a synthesized Java extractor, no other changes.
- **Synthesis cost amortization**: Per-task synthesis cost (all LLM calls across the loop) must pay back within ≤ 200 queries vs. naive full-context baselines.
- **Auditable output**: Every synthesized plugin ships with its originating LLM trace + test-harness report.

## 8. Known Risks and Open Questions

1. **Sandbox escapes** — free-form code synthesis (P4) is the highest-risk stage. Start with template-only synthesis (P3) and only unlock free-form behind a per-repo opt-in.
2. **Schema thrashing** — the LLM may bounce between schemas across refinement iterations. Mitigation: lock schema after Stage 1, require an explicit "schema rewrite" signal to revisit it.
3. **Failure-classifier accuracy** — the Test Harness's failure category drives refinement targeting. If it mislabels a `missing_primitive` as a `wrong_chain`, the LLM patches the wrong stage and loops forever. Needs its own unit tests with labeled failure cases.
4. **Budget explosion on hard tasks** — some domains simply won't hit threshold. The freeze-with-best-so-far policy + hard N-iteration cap keeps this bounded.
5. **Hardcoded API key**: `experiments/exp21_depeval_v4.py:14` has a live OpenAI key in plaintext. Rotate it and move to `os.environ["OPENAI_API_KEY"]` before V5 work lands on GitHub — V5 will make many more LLM calls per run.

## 9. File Layout (proposed)

```
pixelmem/
  v5/
    core/
      pipeline.py            # generalized RetrievalPipeline, plugin host
      plugins.py             # Protocol defs: Extractor, Resolver, Derivation, PromptTemplate
    plugins/
      python_deps/           # V4's current logic, unchanged, registered as default
      legal_clauses/         # first non-code demo
    synth/
      schema_designer.py     # Stage 1
      extractor_synth.py     # Stage 2
      resolver_synth.py      # Stage 3
      derivation_synth.py    # Stage 4
      prompt_synth.py        # Stage 5
      refine.py              # Stage 7 loop
      sandbox.py             # subprocess exec, rlimits, net-off
      templates/             # P3 template-driven synthesis slots
    harness.py               # Stage 6 test harness
    task_card.py             # Task Card schema + loader

docs/
  v5_plan.md                 # this doc
  v5_results.md              # populated in P6

experiments/
  exp22_v5_crosscodeeval.py
  exp23_v5_longmemeval_synth.py
  exp24_v5_depeval_java.py
```

## 10. First Concrete Step

Start P1. The single smallest commit that moves us toward V5:

1. Create `pixelmem/v5/core/plugins.py` with four `Protocol` definitions.
2. Wrap V4's existing modules as default plugin implementations.
3. Make `RetrievalPipeline` accept plugin instances via constructor (kwarg-only, defaulting to V4 plugins).
4. Re-run `exp21_depeval_v4.py` — must still produce 135/166.

No LLM synthesis yet. Once that lands and the regression test passes, P2 (Task Card + Harness) is the next unit of work.
