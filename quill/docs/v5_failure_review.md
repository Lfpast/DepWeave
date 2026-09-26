# V5 Failure Review

> Historical experiment note: versioned module paths and directory layouts below refer to the pre-restructure repository. See [Quill README](../README.md) for the current layout.

A root-cause walk through every failure across the 6 V5 benchmark runs
(`exp22`–`exp26`). Total: **32 failures across 60 attempts**.

## Summary by root cause

| Root cause | Failures | Where | Fixability |
|---|---|---|---|
| **A. Generic regex extractor misses internal import edges** (graph has 0 strong edges → topo sort collapses to listing order → LLM echoes it) | **10** | `exp24` c / c++ / js / ts (10/10 wrong_order cases) | **Easy** — write per-language Extractor plugins emitting `imports_symbol + defined_in` |
| **B. Java imports not captured at all** (pipeline drops files or emits prefix) | **4** | `exp24_java_0..3`, `exp26_baseline_java_3` | **Easy** — Java extractor with `import x.y.Z;` handling |
| **C. LLM reasoning slip** (graph correct, LLM swaps adjacent files) | **2** | `exp26_baseline_java_6/7` | **Medium** — tighter prompt wording or use sonnet for ordering |
| **D. CrossCodeEval floor** (line completion w/o training on the repo) | **5** | `exp25_crosscodeeval_*` | **Not a V5 bug** — task ceiling, SOTA is ~25% EM |
| **E. Task-format mismatch** (metric expected chains / long strings) | **4** | `exp25_dependeval_task4_*`, `exp23_75f70248` | **Easy** — update prompt + metric together |
| **F. Extractor regex too narrow** (LongMemEval multi-line content) | **2** | `exp23_51b23612`, `exp23_2ce6a0f2` | **Easy** — wider regex, per-session chunking |
| **G. Synthesis instability** (LLM emits prose or bad JSON under flood context) | **5** | `exp26_synth_java_3..7` | **Medium** — size-caps on extractor output + stricter prompt schemas |

Categories D + E + F are **output-format / LLM-floor issues**, not architectural problems. Categories A + B + C are all about **extractor quality**, which is exactly what V5's plugin boundary was designed for. Category G is the weakest spot and the clearest next target.

---

## A. Generic-regex extractor misses internal edges (10 cases)

Every `wrong_order` failure in exp24 has **`set(pred) == set(expected)`** — the right files, wrong order. The graph had 0 strong internal edges, so topo sort returned the files in the order they were listed, and 4o-mini trusted that order.

### Evidence

| Case | Pred | Expected | Position diff |
|---|---|---|---|
| `c_0` | `[cram.h, cram_stats.h, cram_index.c]` | `[cram_stats.h, cram.h, cram_index.c]` | swap 0↔1 |
| `c_1` | `[driverlib.h, pmm.h, usbdma.c]` | `[pmm.h, driverlib.h, usbdma.c]` | swap 0↔1 |
| `c_4` | `[status-codes.c, thread-utils.h, time-utils.h]` | `[time-utils.h, thread-utils.h, status-codes.c]` | full reversal |
| `c++_0` | `[...h, ...cpp, ...h]` | `[...h, ...h, ...cpp]` | swap 1↔2 |
| `c++_4` | `[command.hpp, main.cpp, vector.hpp]` | `[vector.hpp, command.hpp, main.cpp]` | 3-way |
| `javascript_1` | `[light.js, app.js, main.js]` | `[light.js, main.js, app.js]` | swap 1↔2 |
| `javascript_2` | `[Collection.js, __instance.js, index.js]` | same swapped | swap 1↔2 |
| `typescript_2` | `[App.tsx, index.ts, update.tsx]` | `[update.tsx, App.tsx, index.ts]` | 3-way |

### Why the regex fallback fails

V4's `_extract_generic` matches `import|from|require|include|use|using <name>` but *always* tags the edge as `COND_EXTERNAL`. Cross-file resolution then doesn't promote these to internal edges because the name format doesn't match the file basenames. So the graph ends up with 0 strong edges.

### Recommended fix

Write per-language `Extractor` plugins that produce `imports_symbol + defined_in` pairs the same way V4's Python extractor does:

- **C / C++**: `#include "header.h"` → `(impl.c, includes_header, header.h, internal)`. Treat `.c/.cpp` as dependent on `.h/.hpp` in the same directory.
- **JavaScript / TypeScript**: `import X from './mod'` → `(file, imports_symbol, X@mod.js, internal)`, plus resolve `./mod` → `mod.js` by stripping extension and matching basenames.
- **Java**: `import pkg.Class;` → `(file, imports_class, Class, internal)`; plus `class X extends Y` → `(X@file, extends, Y)`.

Each of these is ~50-100 LoC. With them in place, 8/10 of these failures should flip to correct on 4o-mini, because the graph topo sort will produce the right answer and the LLM will just confirm.

---

## B. Java import coverage gap (4 cases)

| Case | Pred | Expected | Root cause |
|---|---|---|---|
| `exp24_java_0` | 2 files only | 3 files expected | Extractor didn't see one file at all; pipeline dropped it |
| `exp24_java_2` | 2 files only | 3 files expected | Same |
| `exp24_java_3` | 2 files only | 3 files expected | Same |
| `exp26_baseline_java_3` | 2-file prefix | 3-file expected | Same |

The generic regex doesn't emit any primitives for a Java file whose only symbols are inside nested classes or interfaces. V4's `_extract_generic` requires a top-level `class` or `function` declaration to register the file.

**Fix is the Java extractor plugin from §A**.

---

## C. LLM reasoning slip (2 cases)

Graph correct, confirmed edges shown to LLM, LLM still swapped adjacent files.

| Case | Pred | Expected |
|---|---|---|
| `exp26_baseline_java_6` | `[AvroSchemaUtil, PruneColumns, TypeToSchema]` | `[TypeToSchema, AvroSchemaUtil, PruneColumns]` |
| `exp26_baseline_java_7` | `[SparkSchemaUtil, SparkTypeVisitor, Writer]` | `[SparkTypeVisitor, SparkSchemaUtil, Writer]` |

These are the same category V4 paper labels as **LLM_ERROR** — graph is right, LLM reorders. 4o-mini treats tied-by-alphabetical-order files as interchangeable.

**Fix options**: (a) add a one-shot example of a correctly-resolved topo-tie in the prompt, (b) use sonnet instead of 4o-mini for the ordering step, (c) weight the LLM's output against the graph's topo and break ties with the graph.

---

## D. CrossCodeEval is a task floor (5 cases — not a V5 bug)

All 5 CrossCodeEval failures have legitimate V5 traces (primitives extracted, cross-file context present, prompt parsed). The LLM just produces a plausible-but-wrong function name:

| Case | Pred | Expected |
|---|---|---|
| `1440` | `response(user_message, ...)` | `converse(message=..., ...)` |
| `206`  | `calculate_threshold(freq, search_range)` | `turkey_box_plot([freq[k] for k in search_range])[4]` |
| `277`  | `get_header_value_from_response(...)` | `get_header_value(response.headers, ...)` |
| `441`  | `np.zeros((depth, depth))` | `zeros((depth + 1,))` |
| `468`  | `self.mission_space = mission_space` | `from_string("open the red door ...")` |

Line completion requires specific knowledge of the target repo. Published SOTA on CrossCodeEval is ~25% EM; V5 on 5 samples without retrieval-specific fine-tuning falls below that floor.

**Nothing to fix in V5** — this is a plug-the-right-LLM-in issue. A bigger model or a retrieval-aware completion model would lift this.

---

## E. Task-format mismatches (4 cases)

Two sub-patterns:

### E.1 DependEval Task 4: GT has chains, not pairs (3 cases)

| Case | Pred shape | GT shape |
|---|---|---|
| `t4_0` | 6 pairs `[[a,b],...]` | 7 chains, some 3-element `[[a,b,c],...]` |
| `t4_3` | 2 pairs | 3 chains incl. `[mychrome.py, actionCNN.py, main.py]` |
| `t4_4` | 5 pairs | 4 chains incl. `[kan.py, __init__.py, mnist.py]` |

My prompt asked for 2-element pairs; GT is variable-length dependency chains. Pipeline fires correctly — I wrote the metric wrong.

**Fix**: change `PairExtractionPrompt` to ask for chains, update `_task4_metric` to handle variable-length chains (partial credit on chain prefix match).

### E.2 LongMemEval preference question (1 case)

| Case | Pred | Expected |
|---|---|---|
| `75f70248` | `living room` | `The user would prefer responses that consider the potential impact of their cat, ...` |

Question type: `single-session-preference`. GT is a long preference description; `fuzzy` metric expects short answer. The LLM did extract the right scene (cat in living room) but didn't inflate it into a preference statement.

**Fix**: per-question-type prompts. Preference questions need a "describe the preference in a sentence" prompt variant, not the "single-phrase answer" prompt.

---

## F. Extractor too narrow for LongMemEval (2 cases)

| Case | Pred | Expected | Why |
|---|---|---|---|
| `51b23612` | `[]` | `Nu, pogodi!` | Answer is a Russian cartoon name mentioned inside a multi-line message; regex `^(?P<s>user|assistant):\s*(?P<o>.+)$` didn't match a multi-line block |
| `2ce6a0f2` | `None` | `4` | LLM returned empty response or prose; extractor got turns but the numeric answer lived in free-text context |

**Fix**: widen the LongMemEval extractor to include multi-line continuations and to emit one primitive per sentence rather than per turn line. Also add a chunk-based primitive (cf. V1's chunk-based retrieval).

---

## G. Synthesis instability (5 cases — the weakest spot)

All 5 `exp26_synth_java_*` failures show the same signature:

| qid | n_primitives | tokens | Error |
|---|---|---|---|
| java_3 | 297 | 2692 | parse_error |
| java_4 | 261 | 2953 | parse_error |
| java_5 | 167 | 2561 | parse_error |
| java_6 | 504 | 2916 | parse_error |
| java_7 | 532 | 2813 | parse_error |

**Chain of events**:
1. `SchemaDesigner` succeeded but used 20 relations (at the cap; indicates sprawl).
2. `ExtractorSynth` *validation failed* because the LLM used conditions not declared in the schema — fell back to the catch-all `any_line` pattern.
3. `any_line` emits one primitive per non-blank source line → 167-532 primitives per Java file.
4. `PromptSynth` output was a JSON-array-expecting template, fed 60+ raw primitive lines → prompt was 2500+ tokens → LLM returned prose, not a JSON array.
5. Parser rejects → `parse_error`.
6. `Refiner` reran `PromptSynth` in iteration 2; same drift.

**Three fixes, increasing effort**:

1. **Size-cap the extractor fallback.** When we fall back to `any_line`, cap at N=50 primitives or refuse to run. This avoids the context flood that breaks downstream stages.
2. **Tighter schema validation** — enforce condition-channel consistency across Stage-1 and Stage-2 output. If ExtractorSynth emits a condition not in the schema, retry *that* stage instead of falling back.
3. **Stricter LLM prompts for the synthesis stages** — require "return ONLY valid JSON, no explanation" with a few-shot example of a correct response. This is the single highest-leverage change for synthesis quality.

---

## What to tackle first (ranked by ROI)

| Priority | Task | Expected lift | Effort |
|---|---|---|---|
| 1 | Java/C/C++/JS/TS Extractor plugins | +14 cases (A+B) | 2-3 days, ~400 LoC total |
| 2 | Tighten synthesis prompts + size-cap fallbacks | +~3/5 on Java synthesis | 1 day |
| 3 | Per-question-type LongMemEval prompts | +~1/5 on LongMemEval | 0.5 day |
| 4 | DependEval Task 4 chain-aware prompt+metric | +~2/5 on Task 4 | 0.5 day |
| 5 | Better LLM for ordering (sonnet vs 4o-mini) | +2 cases (C) | instant, more cost |

Everything in the top 4 is **plugin-level** — no changes to V5 core. That's a deliberate consequence of the architecture: failures localize to specific plugins, so fixes do too.
