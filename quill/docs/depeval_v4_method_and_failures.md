# PixelMem V4: Graph-Native Retrieval for DependEval

> Historical experiment note: versioned module paths and directory layouts below refer to the pre-restructure repository. See [Quill README](../README.md) for the current layout.

## 1. Task

**DependEval Task 2 (Repository Construction)**: Given 3-5 Python source files with full code (~40K tokens), determine the correct dependency ordering -- base files first, dependent files last.

**Paper baselines** (all see full source code):

| Model | EM% | Tokens/query |
|-------|-----|-------------|
| Qwen2.5-Coder-32B | 70.0 | ~40,000 |
| Qwen-2.5-72B | 67.2 | ~40,000 |
| GPT-4o-mini (direct) | 60.0 | ~40,000 |
| Claude-3.5-Sonnet | 55.6 | ~40,000 |

---

## 2. V4 Architecture

V4 redesigns the retrieval layer while keeping core PixelMem storage unchanged. The key insight: store **primitive quadruples** (simple facts), build a **dependency graph** from them, and present **structural evidence** (not raw triples) to the LLM.

### 2.1 Pipeline Overview

```
Step 1: ALIAS NAMESPACE (bookkeeping)
   Detect duplicate basenames: __init__.py, __init__(1).py
   Detect duplicate symbols: helper@a.py, helper@a(1).py
   Deterministic, stable aliases for internal reasoning

Step 2: PRIMITIVE QUADRUPLE EXTRACTION (AST-based)
   (repo, contains_file, main.py, repo_level)
   (main.py, contains_function, run@main.py, symbol_level)
   (main.py, imports_symbol, Base@base.py, internal)
   (Model@model.py, extends, Base, symbol_level)
   ~95 primitives per question (3-5 files)

Step 3: PIXELMEM STORAGE
   Encode primitives into N x N PNG pixel matrices
   Relation matrix: RGB color per relation type
   Condition matrix: RGB color per channel (repo/file/symbol/internal/external)

Step 4: DEPENDENCY GRAPH (derived from primitives)
   Chain: imports_symbol + defined_in => file A depends_on file B
   Chain: extends + defined_in => file A depends_on file B
   Confidence: 0.90 (resolved import), 0.50 (heuristic)
   Topological sort for initial ordering

Step 5: NATURAL LABEL RECONSTRUCTION
   a(1).py => "the a.py that defines other"
   __init__(1).py => "__init__.py (in pkg_b/)"
   Internal aliases NEVER shown to LLM

Step 6: HYBRID LLM PROMPT (structural evidence + raw imports)
   Per-file summaries: raw import lines + definitions
   Confirmed dependencies: graph edges with evidence
   Computed order: topological sort suggestion
   Single API call, ~371 tokens

Step 7: PARSE + FALLBACK
   Parse JSON array from LLM response
   Map natural labels back to file aliases
   Fall back to computed order if parse fails
```

### 2.2 New Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `v4/alias_namespace.py` | 297 | Deterministic deduplication for files and symbols |
| `v4/primitive_extractor.py` | 473 | AST-based extraction of simple quadruples |
| `v4/dependency_graph.py` | 487 | File-level dependency graph with topo sort |
| `v4/natural_labels.py` | 187 | Convert internal aliases to human-readable labels |
| `v4/reconstruction.py` | 270 | Build compact evidence objects from primitives |
| `v4/retrieval.py` | 485 | 4-stage retrieval pipeline + hybrid prompt |

### 2.3 Key Design Decisions

1. **Primitive facts, not summaries**: V3 stored `"imports: from X import Y; defines: func1, func2"` as a text blob in the condition field. V4 stores atomic facts: `(file, imports_symbol, Y@target, internal)`. This enables graph reasoning.

2. **Graph-derived dependencies**: Instead of pattern-matching raw import lines, V4 chains `imports_symbol + defined_in` to build file-level dependency edges with confidence scores.

3. **Hybrid prompt**: Neither raw imports alone (63%) nor structural evidence alone (60%) matches the hybrid (67%). The LLM needs both:
   - Raw imports: to catch edges the graph missed
   - Confirmed edges: to avoid re-deriving known dependencies

4. **Natural labels hide aliases**: The LLM never sees `a(1).py`. Instead it sees `"the a.py that defines helper"` or `"a.py (in utils/)"`.

---

## 3. Concrete Example: Correct Prediction (Q5)

### Input Files
```
classify.py:   defines get_clarifai_api, predict_image_bias, combine_text_and_image
views.py:      from popnews.classify import combine_text_and_image
urls.py:       from popnews import views
```

### V4 Primitives Extracted (68 total, key ones shown)
```
(repo, contains_file, classify.py, repo_level)
(repo, contains_file, views.py, repo_level)
(repo, contains_file, urls.py, repo_level)
(classify.py, contains_function, combine_text_and_image@classify.py, symbol_level)
(views.py, imports_symbol, combine_text_and_image@classify.py, internal)
(urls.py, imports_symbol, views@views.py, internal)
```

### V4 Dependency Graph
```
views.py ──depends_on──> classify.py  (conf=0.90, imports symbol combine_text_and_image)
urls.py  ──depends_on──> views.py     (conf=0.90, imports symbol views)
```

### Hybrid Prompt Sent to LLM
```
File summaries:
classify.py:
  imports: from django.conf import settings; from clarifai import rest
  defines: get_clarifai_api, get_concept_scores, predict_image_bias, combine_text_and_image
views.py:
  imports: import json; from django.contrib.auth.decorators import login_required; ...
           from popnews.classify import combine_text_and_image
  defines: index, save_article, user_stats, test_save
urls.py:
  imports: from django.conf.urls import url; from popnews import views

Confirmed dependencies:
  views.py depends on classify.py (imports symbol combine_text_and_image)
  urls.py depends on views.py (imports symbol views)

Computed dependency order: ["classify.py", "views.py", "urls.py"]

Is this correct? If yes, return it. If not, fix and return the corrected order.
Return ONLY a JSON array of filenames, preserving exact case.
```

### LLM Response
```json
["classify.py", "views.py", "urls.py"]
```

**Result: CORRECT.** The graph found both edges, topo sort produced the right order, LLM confirmed. Total tokens: 167.

---

## 4. Concrete Example: Duplicate Basenames (Q2)

### Input Files
```
ParaGen/paragen/criteria/base_criterion.py    (defines BaseCriterion)
ParaGen/paragen/criteria/__init__.py           (from .abstract_criterion import ...)
ParaGen/examples/lightseq/ls/lightseq_label_smoothed_cross_entropy.py  (imports BaseCriterion)
ParaGen/examples/lightseq/ls/__init__.py       (from .lightseq_label_smoothed_cross_entropy import ...)
```

### Alias Namespace
```
base_criterion.py           -> base_criterion.py        (unique)
__init__.py                 -> __init__.py              (first)
lightseq_label_smoothed...  -> lightseq_label_smoothed_cross_entropy.py  (unique)
__init__.py                 -> __init__(1).py           (second, INTERNAL ONLY)
```

### Natural Labels (what LLM sees)
```
__init__.py      -> "__init__.py (in criteria/)"
__init__(1).py   -> "__init__.py (in ls/)"
```

The LLM NEVER sees `__init__(1).py`. It sees the natural disambiguated label.

---

## 5. Results

### 5.1 V4 Hybrid: 112/166 (67%)

```
Accuracy: 112/166 (67%)
Avg tokens: 371/query
Avg primitives: 95/question
Strong edges: 2.1 avg
```

### 5.2 By File Count

| Files | Correct | Total | V4 | V3 |
|-------|---------|-------|-----|-----|
| 3 | 65 | 79 | **82%** | 81% |
| 4 | 39 | 71 | 55% | **59%** |
| 5 | 8 | 16 | **50%** | 25% |

V4 doubles 5-file accuracy (50% vs 25%) while matching 3-file performance.

### 5.3 Duplicate Basenames

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Has duplicates | 17 | 31 | 55% |
| No duplicates | 95 | 135 | 70% |

### 5.4 Comparison Across Methods

| Method | Total | 3-file | 4-file | 5-file | Tokens |
|--------|-------|--------|--------|--------|--------|
| V3 depeval (exp20) | 110/166 (66%) | 64/79 (81%) | 42/71 (59%) | 4/16 (25%) | 347 |
| V4 structural only | 99/166 (60%) | 59/79 (75%) | 33/71 (46%) | 7/16 (44%) | 230 |
| V4 raw imports only | 105/166 (63%) | 60/79 (76%) | 36/71 (51%) | 9/16 (56%) | 336 |
| **V4 hybrid** | **112/166 (67%)** | **65/79 (82%)** | 39/71 (55%) | **8/16 (50%)** | 371 |

### 5.5 Prompt Ablation

The hybrid prompt combines raw imports + graph evidence. Each component alone is insufficient:

| Prompt type | Accuracy | Why |
|-------------|----------|-----|
| Raw imports only | 63% | LLM must re-derive all dependencies from scratch |
| Structural evidence only | 60% | Too sparse -- LLM can't catch edges graph missed |
| **Hybrid (both)** | **67%** | Graph provides confirmed edges; imports catch the rest |

---

## 6. Failure Analysis (54 failures)

### 6.1 Error Taxonomy

| Error Type | Count | % | Description |
|-----------|-------|---|-------------|
| ambiguous | 37 | 69% | Not enough strong edges to fully order |
| wrong_order | 13 | 24% | Enough edges but LLM ordered incorrectly |
| resolver | 4 | 7% | Zero extractable edges |

### 6.2 Ambiguous Failures (37 cases)

**Edge gap distribution**: 29 cases need just 1 more edge, 7 need 2, 1 needs 3.

#### Example: Q24 (ambiguous, 1 edge short)

```
Files:
  host_ops_type.py:  imports from typing, mlos_bench (all external)
  saas_env.py:       NO imports at all (empty file)
  __init__.py:       from mlos_bench...host_env import HostEnv (external)

Expected: [host_ops_type.py, saas_env.py, __init__.py]
Got:      [saas_env.py, host_ops_type.py, __init__.py]

Graph: 0 strong edges between these files
       All imports reference external packages, not each other.
```

**Root cause**: `saas_env.py` is empty -- no code signal exists to determine its position. The ground truth ordering reflects developer convention.

#### Example: Q4 (ambiguous, __init__.py placement)

```
Files:
  iq_privacy_get.py:      imports from yowsup (external)
  __init__.py:             from .iq_unregister import ...; from .iq_status_set import ...
  test_iq_privacy_get.py:  from yowsup...protocolentities import GetPrivacyIqProtocolEntity

Expected: [iq_privacy_get.py, __init__.py, test_iq_privacy_get.py]
Got:      [iq_privacy_get.py, test_iq_privacy_get.py, __init__.py]

Graph: 1 strong edge (test_ -> iq_privacy_get via heuristic)
       __init__.py imports from OTHER package files not in our subset.
```

**Root cause**: `__init__.py` imports from files outside the 3-file subset. No edge exists between `__init__.py` and `test_iq_privacy_get.py`.

### 6.3 Wrong Order Failures (13 cases)

Three subcategories:

#### A. Stdlib false edges (Q15, Q23, Q44)

**Example: Q15**

```
Files:
  spoc_admin.py:  import spoc (external)
  types.py:       from typing import Optional; import fastberry as fb
  framework.py:   from types import SimpleNamespace; from .spoc_admin import spoc
  __init__.py:    from .spoc_admin import spoc; from .scripts import ...

V4 Prompt (confirmed dependencies section):
  __init__.py depends on spoc_admin.py (imports symbol spoc)
  __init__.py depends on framework.py (imports symbol Fastberry as App)
  framework.py depends on types.py (imports symbol SimpleNamespace)   <-- FALSE
  framework.py depends on spoc_admin.py (imports symbol spoc)

Expected: [spoc_admin.py, framework.py, __init__.py, types.py]
Got:      [spoc_admin.py, types.py, framework.py, __init__.py]
```

**Root cause**: `from types import SimpleNamespace` in `framework.py` references the Python stdlib `types` module, NOT the local `types.py` file. The V4 graph creates a false edge `framework.py -> types.py`, which forces `types.py` before `framework.py`. In reality, `types.py` has no dependents and should be LAST.

The graph's confirmed dependency section shows this false edge to the LLM, making it harder to correct than if the LLM only saw raw imports.

#### B. LLM reasoning errors (Q61, Q84, Q126, Q148)

**Example: Q61 (LLM swaps correct order)**

```
Files: constants.py, escsm.py, escprober.py, universaldetector.py, chardetect.py

Graph edges (5 strong, all correct):
  escsm.py -> constants.py (imports constants)
  escprober.py -> constants.py (imports constants)
  escprober.py -> escsm.py (imports HZSMModel, ISO2022CNSMModel...)
  universaldetector.py -> constants.py (imports constants)
  chardetect.py -> universaldetector.py (implied)

Expected: [constants, escsm, escprober, universaldetector, chardetect]
Got:      [constants, escprober, escsm, universaldetector, chardetect]
                     ^^^^^^^^^ ^^^^^^^  SWAPPED
```

**Root cause**: The graph correctly has `escprober -> escsm` (escprober imports from escsm). The topo sort should place escsm before escprober. But the LLM sees both files import from `constants` and swaps them, ignoring the confirmed edge.

#### C. Duplicate confusion (Q108, Q150, Q160)

**Example: Q160 (two task.py files)**

```
Files: task.py (celery), task.py (airflow), decorator.py, __init__.py, kubernetes.py

Expected: [task.py, decorator.py, __init__.py, task.py, kubernetes.py]
Got:      [task.py, task.py, decorator.py, __init__.py, kubernetes.py]

Natural labels: "the task.py that defines ..." vs "the task.py from ..."
```

**Root cause**: Two `task.py` files with similar structures. The natural label disambiguation helps, but the LLM still groups them together instead of interleaving with other files.

### 6.4 Resolver Failures (4 cases)

**Example: Q7 (relative import not resolved)**

```
Files:
  __init__.py:         from .rerank_options import ...  (relative import)
  rerank_options.py:   external imports only
  __init__(1).py:      external imports only

Expected: [rerank_options.py, __init__.py, __init__.py]
Got:      [__init__.py, rerank_options.py, __init__.py]

Graph: 0 strong edges
```

**Root cause**: `from .rerank_options import ...` should create an edge `__init__.py -> rerank_options.py`, but the relative import resolution fails because the module map doesn't match the alias. This is a fixable bug in the cross-file import resolver.

---

## 7. What We Tried and Why It Worked/Didn't

### 7.1 Structural-only prompt -> 60% (-7pp)
```
Showed only: file identities + confirmed edges + ambiguous pairs
Removed: raw import lines
```
**Why it failed**: Too sparse. The LLM couldn't catch edges the graph missed. When the graph had 0 strong edges, the LLM had nothing to work with.

### 7.2 Raw-import-only prompt -> 63% (-4pp)
```
Showed only: raw import lines + definitions + computed order
Removed: confirmed dependency edges
```
**Why it's mediocre**: The LLM must re-derive all dependencies from scratch. It often gets confused by stdlib names and ambiguous imports.

### 7.3 Hybrid prompt -> 67% (best)
```
Showed: raw imports + definitions + confirmed edges + computed order
```
**Why it works**: Best of both worlds. Confirmed edges prevent the LLM from making obvious mistakes. Raw imports let it catch what the graph missed.

### 7.4 __init__.py goes-late heuristic -> 58% (-6pp regression)
```
Added weak edges: __init__.py depends_on every other file (conf=0.15)
```
**Why it failed**: Creates bidirectional evidence conflicts. Some `__init__.py` files are bases (go early), some are aggregators (go late). The blanket heuristic hurts more than it helps.

---

## 8. V3 vs V4 Architecture Comparison

| Aspect | V3 | V4 |
|--------|-----|-----|
| **Storage** | Text blobs in condition field | Atomic primitives (95 per question) |
| **Deduplication** | File IDs (F0, F1) in prompt | Internal aliases, natural labels in prompt |
| **Graph** | PartialOrder with conf thresholds | DependencyGraph with evidence chains |
| **Prompt** | File IDs + raw imports | Natural labels + raw imports + confirmed edges |
| **5-file accuracy** | 25% | **50%** |
| **Token usage** | 347 | 371 |
| **Extensibility** | Monolithic pipeline | Modular: namespace, extractor, graph, labels, reconstruction |

### Why V4 wins on 5-file cases

V4's AST-based extractor finds more edges than V3's regex resolver:
- **extends** edges (class inheritance): `Model(Base)` -> `Model depends on Base's file`
- **Symbol-level resolution**: `imports_symbol + defined_in` chain is more precise than module-name matching
- These extra edges matter most when there are 5 files and many possible orderings (5! = 120)

### Why V4 is slightly behind on 4-file cases

- Natural labels are less stable than FileIDs for LLM parsing
- Some 4-file cases with duplicate `__init__.py` get natural labels that 4o-mini handles less reliably than F0/F1
- V3's pairwise refinement (extra LLM calls for 2-file ambiguous layers) helps some 4-file cases

---

## 9. Theoretical Limits

| Category | Count | Fixable? |
|----------|-------|----------|
| Correct | 112 | -- |
| Ambiguous (1 edge short) | 29 | **Partially** -- some have no code signal |
| Ambiguous (2+ edges short) | 8 | **Hard** -- multiple missing relationships |
| LLM reasoning error | 7 | **Partially** -- better prompts might help |
| Stdlib false edges | 3 | **Yes** -- targeted symbol-level filter |
| Dup basename confusion | 3 | **Hard** -- LLM struggles with similar files |
| Resolver bug (relative imports) | 4 | **Yes** -- fix relative import resolution |

**Practical ceiling**: ~130/166 (78%) if all fixable issues resolved.
**Current best**: 112/166 (67%).

---

## 10. Token Economics

| Component | V4 Tokens |
|-----------|-----------|
| File summaries (imports + defs) | ~200 |
| Confirmed dependency edges | ~50 |
| Computed order | ~30 |
| LLM prompt overhead | ~70 |
| LLM response | ~20 |
| **Total per query** | **~371** |
| Paper baselines (full code) | ~40,000 |
| **Savings** | **~108x fewer** |

---

## 11. Key Takeaways

1. **67% on DependEval using ~108x fewer tokens** -- matching Qwen-72B (67.2%) with GPT-4o-mini.

2. **Primitive quadruples enable graph reasoning**: Atomic facts like `(file, imports_symbol, sym@target, internal)` allow building a proper dependency graph, which is impossible from text-blob summaries.

3. **Hybrid prompt is the sweet spot**: The LLM needs both structural evidence (what we know) and raw imports (what it can discover). Neither alone is sufficient.

4. **5-file accuracy doubled**: V4's AST-based graph finds extends/inheritance edges that V3's regex misses. This matters most in complex cases.

5. **Natural labels work**: Internal aliases like `a(1).py` are successfully hidden from the LLM. The disambiguation descriptions ("the a.py that defines helper") are unambiguous.

6. **The dominant failure is ambiguity**: 69% of failures are cases where files have no direct import relationship. This is a fundamental information limit.

---

## 12. Repository

GitHub: https://github.com/NingWang0123/pixel_mem

### V4 Key Files
- `pixelmem/v4/alias_namespace.py` -- deterministic deduplication
- `pixelmem/v4/primitive_extractor.py` -- AST-based fact extraction
- `pixelmem/v4/dependency_graph.py` -- graph builder + topo sort
- `pixelmem/v4/natural_labels.py` -- alias-free label reconstruction
- `pixelmem/v4/reconstruction.py` -- compact evidence objects
- `pixelmem/v4/retrieval.py` -- 4-stage retrieval + hybrid prompt
- `tests/test_v4.py` -- 14 tests covering all components
- `experiments/exp21_depeval_v4.py` -- DependEval benchmark script

### Results
- `results/exp21_depeval_v4.json` -- 112/166 (67%)
