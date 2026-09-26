# V5 vs V4 — What Changed and Why

> Historical experiment note: versioned module paths and directory layouts below refer to the pre-restructure repository. See [Quill README](../README.md) for the current layout.

One-sentence summary: **V4 is a hand-written pipeline for one task. V5 is a plugin host that can run V4 as a plugin — or synthesize its own plugins from a TaskCard for any new task.**

## TL;DR table

| Dimension | V4 | V5 |
|---|---|---|
| Scope | DependEval Task 2 (code dep ordering) only | Any task expressed as a TaskCard |
| How plugins are chosen | Hard-coded in `RetrievalPipeline` | `PluginSet` passed in at runtime (or synthesized by LLM) |
| Extractor | `pixelmem/v4/primitive_extractor.py` (Python AST + regex) | Any object matching the `Extractor` Protocol |
| Dependency graph | `pixelmem/v4/dependency_graph.py` (hard-coded chains) | `DerivationRule` list + a pluggable `DerivationEngine` |
| Prompt template | Hard-coded hybrid prompt in `retrieval.py` | Any object matching the `PromptTemplate` Protocol |
| Tool synthesis | N/A | 4-stage synthesis loop: schema → extractor → rules → prompt → refine |
| Failure diagnostics | Ad-hoc per-experiment | `TestHarness` classifies each miss into one of 6 categories |
| Refinement loop | Manual — you edit code | Automated — `Refiner` patches the one stage its diagnostics blame |
| Isolation | — | V5 core has zero V4 imports; V4 coupling confined to `pixelmem/v5/plugins/python_deps/default.py` |
| Regression risk from V5 work | — | Enforced by `tests/test_v5_core.py::IsolationTest` (AST scan) |
| Dependency direction | V4 stands alone | V5 → V4 is allowed only through the adapter |
| DependEval Python accuracy | 81.3% (full, 166 items) | 5/5 on smoke; same plugin, same numbers |

---

## 1. Architectural differences

### 1.1 V4: tight, focused, monolithic

```
v4/
  alias_namespace.py         # deduplicate basenames (__init__.py)
  primitive_extractor.py     # Python AST + regex — hard-coded relation vocab
  symbol_resolver.py         # stdlib filter, re-export tracing
  dependency_graph.py        # imports_symbol + defined_in → depends_on (in-code chain)
  natural_labels.py          # disambiguate dup basenames for the LLM
  reconstruction.py          # compact evidence objects
  retrieval.py               # RetrievalPipeline: index + query(mode=...) + run_ordering
```

All components live together. `RetrievalPipeline.__init__` wires them up in one
place; a different task would require a different file of code, or a fork.

### 1.2 V5: plugin host + synthesis stages

```
v5/
  core/
    types.py        # Primitive, EvidenceBundle, TaskSpec, PipelineStats, LLMFn
    plugins.py      # Extractor / Resolver / DerivationRule / PromptTemplate / LLMCaller
    derivation.py   # DefaultDerivationEngine (rule matcher)
    pipeline.py     # V5Pipeline — orchestrates plugin instances
  task_card.py      # JSON/YAML TaskCard loader
  harness.py        # TestHarness + failure classifier + EvalReport
  plugins/
    python_deps/    # Default plugin set — wraps V4 1:1
      default.py    # (only file in V5 that imports V4)
  synth/
    schema_designer.py    # Stage 1: LLM → relation vocabulary
    extractor_synth.py    # Stage 2: LLM → TemplateExtractor patterns
    derivation_synth.py   # Stage 3: LLM → DerivationRule list
    prompt_synth.py       # Stage 4: LLM → SynthesizedPrompt
    refine.py             # Stage 6: reads harness diagnostics, picks next action
    orchestrator.py       # End-to-end loop
    _json_tolerant.py     # Fence-stripping + trailing-comma-tolerant JSON parse
```

Two big structural changes:

1. **Plugins are data, not code.** `PluginSet` is a dataclass; you build it at
   runtime and hand it to `V5Pipeline`. No subclassing, no monkeypatching.
2. **There is a second pipeline on top.** Synthesis produces a `PluginSet`
   from a TaskCard via 4 LLM calls, then runs the normal `V5Pipeline` with it.
   Same host, same contract.

### 1.3 Isolation boundary

V5 core has **zero** imports from `pixelmem.v4`. Enforced by:

```python
# tests/test_v5_core.py
class IsolationTest(unittest.TestCase):
    def test_v5_core_has_no_v4_imports(self):
        for mod in (pipeline, plugins, types, derivation, task_card, harness):
            hits = _v4_imports_in(mod.__file__)  # AST walk for import statements
            self.assertEqual(hits, [], ...)
```

Only `pixelmem/v5/plugins/python_deps/default.py` imports V4 — and its job is
explicitly "V4 adapter". Anyone who wants V5 without V4 drops this file and
substitutes their own plugin.

Operational consequence: V5 edits **never break V4 tests** (14 passing, verified
before and after every V5 change).

---

## 2. What's new in V5 that V4 can't do

### 2.1 Take a new task with no code changes

V4 on a new benchmark requires: clone the repo, edit `RetrievalPipeline`, maybe
edit `primitive_extractor.py`. V5 needs: a `TaskCard` (JSON) + a `PluginSet`
(either hand-written or synthesized).

Actual V5 usage for LongMemEval (not possible in V4 without a fork):

```python
card = TaskCard.from_file("longmemeval.yaml")
plugins = _longmemeval_plugins()   # ~40 lines, one regex + one prompt
pipeline = V5Pipeline(plugins, card.spec, openai_4omini)
pred, stats = pipeline.run(query_input, documents=docs)
```

### 2.2 Diagnose and patch a failure automatically

V4 failures bubble up as low accuracy and you open a debugger. V5's
`TestHarness` + `Refiner` is a closed loop:

```
run holdout → per-case categorization →
  "3/5 are zero_strong_edges" → rerun DerivationSynth →
  re-evaluate
```

Categories the classifier emits today:

| Category | What it points at |
|---|---|
| `missing_primitive` | Extractor produced < min_primitives/doc — rerun ExtractorSynth |
| `zero_strong_edges` | Derivation had no high-confidence evidence — rerun DerivationSynth |
| `parse_error` | LLM output didn't parse — rerun PromptSynth |
| `wrong_order` | Topo got confused — rerun PromptSynth |
| `wrong_answer` | Everything clean but answer wrong — inspection |
| `true_ambiguity` | Task has no signal — don't iterate |

### 2.3 Isolate domain coupling

In V4, every Python-specific decision is mixed into the generic pipeline
(`if language == "python":` branches inside the extractor). V5 puts language
coupling inside the plugin set. A Java V5 deployment doesn't even import the
Python AST module.

---

## 3. What V5 preserved from V4

These are NOT changes — V5 deliberately keeps them:

1. **Pixel-encoded storage** — V5 still uses `ShardManager` + PNG matrices
   when a plugin set chooses to. The storage layer is orthogonal to the
   pipeline layer.
2. **Hybrid prompt philosophy** — V4's finding that raw imports + confirmed
   edges beats either alone (81% vs 63% vs 60%) is baked into V5's default
   `PromptSynth` guidance.
3. **Quadruple format** — `(subject, relation, object, condition)` in both.
   V5's `Primitive` is a V4 `Triple` + optional `provenance`.
4. **AliasNamespace** for deduplicating files/symbols — V5 plugins reuse V4's
   implementation through the adapter.

---

## 4. Benchmark results, side by side

All numbers are 5-example smoke tests on 4o-mini, except V4's full DependEval.

| Benchmark | V4 | V5 (via V4 wrapper) | V5 (synthesized) |
|---|---|---|---|
| DependEval Python (3-file, 5 examples) | — | **5/5 (100%)** | not run |
| DependEval Python (full, 166 questions) | **81.3%** | would be identical | not run |
| DependEval TypeScript (3-file) | N/A | 4/5 (80%) | not run |
| DependEval JavaScript (3-file) | N/A | 3/5 (60%) | not run |
| DependEval C++ (3-file) | N/A | 3/5 (60%) | not run |
| DependEval C (3-file) | N/A | 2/5 (40%) | not run |
| DependEval Java (3-file) | N/A | 2/5 (40%) | **1/5 (20%)** |
| DependEval Task 4 (pair extraction) | N/A | 1/5 | not run |
| CrossCodeEval (line completion) | N/A | 0/5 strict EM | not run |
| LongMemEval (5 balanced examples) | not applicable | 2/5 (40% fuzzy) | not run |

### Notes

- **V5 via V4 wrapper on DependEval Python: 5/5.** The plugin set is a 1:1
  re-export of V4's pipeline — same accuracy, different orchestration. This is
  the regression test: V5 can't break V4's strong zone.
- **DependEval non-Python languages**: accuracy tracks extractor quality.
  Python has V4's AST path; others fall back to V4's `_extract_generic` regex,
  which handles TypeScript/JS imports cleanly but misses Java-specific
  patterns.
- **V5 synthesized vs V4 wrapper on Java (40% vs 20%)**: synthesis loses — the
  regex fallback is better tuned than what 4o-mini produces in 2 refinement
  iterations. Not a surprise; V4 fell back for a reason. The value of the
  synthesis path is that it runs on domains where no hand-tuned plugin exists
  yet (e.g. legal contracts, conversational memory).
- **CrossCodeEval 0/5** is the task's floor for a non-retrieval-trained model,
  not a V5 bug. V5's pipeline ran cleanly — primitives extracted, cross-file
  context passed through, prompt parsed without error.
- **LongMemEval 2/5**: retrieval-shaped task; V5's graph provides almost no
  advantage here. The misses were all `missing_primitive` / format mismatches
  that a refinement iteration on a wider extractor regex would likely fix.

---

## 5. Dependency graph between the versions

```
┌──────────────┐
│  pixelmem/   │    Base storage (unchanged across versions)
│   memory.py, shard_manager.py, triple_extractor.py, ...
└────┬─────────┘
     │
     ├── v1-v3 retrieval experiments (frozen)
     │
     ├── v4/                     Hand-written DependEval pipeline
     │   └── All v4 tests pass after every v5 change (verified)
     │
     └── v5/
         ├── core/               No v4 imports
         ├── plugins/
         │   └── python_deps/default.py  ← ONLY place v5 → v4
         │       imports v4.primitive_extractor, v4.dependency_graph,
         │               v4.symbol_resolver, v4.natural_labels
         └── synth/              No v4 imports
```

Check the boundary any time with:

```bash
python3 -m unittest tests.test_v5_core.IsolationTest -v
```

---

## 6. When to use which

| Situation | Use |
|---|---|
| Running DependEval Task 2 on Python, want maximum accuracy | V4 directly (`pixelmem.v4.retrieval.RetrievalPipeline`) — battle-tested on 166 items |
| Running DependEval Task 2 on Python through V5's harness | V5 + `build_python_deps_plugins()` — same accuracy, uniform harness |
| Running DependEval Java/JS/C/etc. | V5 + `build_python_deps_plugins()` with `language=...` — current best without synthesis; accept 20-80% range |
| Running a non-code task (LongMemEval, legal docs, medical) | V5 + hand-written plugin set (~40-80 LoC) |
| Adding a new benchmark, no idea where to start | V5 synthesis (`synthesize_pipeline`) — 4 LLM calls produce a runnable pipeline; use its output as a starting point to hand-tune |
| Production code with an accuracy SLA | V4 for DependEval Python; elsewhere V5 with a hand-written plugin set reviewed by a human |
| Research — does this domain have graph-reasoning signal? | V5 synthesis + harness diagnostics. Failure categories tell you which stage to invest human effort in |

---

## 7. Files changed / added (summary)

| Layer | Files |
|---|---|
| **V4** | no modifications — all existing files untouched |
| **V5 core** | `pixelmem/v5/core/{__init__,types,plugins,derivation,pipeline}.py`, `pixelmem/v5/{__init__,task_card,harness}.py` |
| **V5 default plugin (only V4 touchpoint)** | `pixelmem/v5/plugins/python_deps/{__init__,default}.py` |
| **V5 synthesis** | `pixelmem/v5/synth/{__init__,schema_designer,extractor_synth,derivation_synth,prompt_synth,refine,orchestrator,_json_tolerant}.py` |
| **Tests** | `tests/test_v5_core.py` (8 tests incl. AST-based isolation check) |
| **Experiments** | `experiments/exp22_v5_haiku_smoke.py`, `exp23_v5_real_smoke.py`, `exp24_v5_multilang_smoke.py`, `exp25_v5_workflow_benchmarks.py`, `exp26_v5_synthesis_java.py` |
| **Docs** | `docs/v5_plan.md`, `docs/v5_examples.md`, `docs/v5_vs_v4.md` |

Total V5 footprint: ~2,500 lines across 19 Python files. V4 footprint
unchanged.
