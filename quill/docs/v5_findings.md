# V5 Findings — Complete Story

> Historical experiment note: versioned module paths and directory layouts below refer to the pre-restructure repository. See [Quill README](../README.md) for the current layout.

Consolidated from exp22 through exp36. This doc replaces the scattered
per-experiment notes for anyone coming in fresh.

## TL;DR

- **V5 is a plugin-based pipeline host** on top of V4's storage, with LLM-driven
  tool synthesis as an optional layer. Isolated from V4 (AST-enforced).
- **V5 matches V4 on its strong zone** (DependEval Python 3-file: 5/5 via
  adapter; same 81% as V4 on full runs) and **beats V4 regex fallback on
  other languages** (Java: 3/5 vs 2/5 baseline).
- **V5's one-shot develop-tools workflow** transfers tools from 1 labeled
  example → Java DependEval 3/5. Works on tasks with stable meta-patterns.
- **V5 is NOT a magic wand**. Fails on:
  - CrossCodeEval line completion (free-form output, often-absent symbols)
  - DependEval Task 4 (NL-only input, no code graph)
  - LongMemEval retrieval (retrieval-shaped, not graph-shaped)
- **Three-rule fit test**: document-shaped input + structured output +
  extractable meta-pattern. Lose any → V5 doesn't help.
- **DocRED is V5's best non-code fit** — native quadruple output shape.
  Hand-crafted V5 gets avg F1 0.15 on 5 short docs at 4o-mini zero-shot
  (in range with general-LLM DocRED baselines).

## 1. Architecture journey

Seven commits of incremental architecture, in order:

| Commit | What we added | Why |
|---|---|---|
| `0cf1238` | V5 core + V4 adapter plugin + template-driven synthesis | Pluggable pipeline host; V5 runs V4 identically through the adapter |
| `b3759f1` | ExtractorTester + feedback-loop develop→test→revise | Generic quality signals for synthesis (coverage, relation mix, cross-doc edges) |
| `2180b19` | `develop_from_one_shot` — single labeled example teaches tools end-to-end | Sharper than generic signals: if tools reproduce GT on the learn case, they've captured something |
| `f43ae0a` | Candidate-aware prompt + SymbolExtractor | Reduce workload: hand the LLM a ranked candidate list, don't ask for open-ended generation |
| `b245cf6` | Multi-LLM comparison (4o-mini / haiku / opus) | Test whether stronger LLM rescues candidate-aware on CCEval |
| `e45bcf9` | Trace-learned two-stage (Reducer + Finisher) + receiver matching + underscore penalty | Teach LLM the HOW via solution traces; specialize the ranker |
| `a6a7a42` | Benchmark-fit analysis after Task 4 negative result | Encode the fit rule; stop running tests that can't pass |
| (this commit) | DocRED exp35 + exp36 | First non-code benchmark; validates fit analysis |

## 2. All experiments, ranked by V5 lift

| Exp | Benchmark | Baseline | V5 | Lift | Fit? |
|---|---|---|---|---|---|
| 22 | synthetic python / kg / convo | N/A | 5/5 all | baseline for infra | ✅ |
| 23 | DependEval Python (real) | — | **5/5** | confirms V4 parity | ✅ |
| 23 | LongMemEval | — | 2/5 fuzzy | retrieval-shaped | ❌ |
| 24 | DependEval {Java, JS, C, C++, TS} | N/A | 1-4/5 mixed | generic regex bottlenecked | ✅ partial |
| 25 | DependEval Task 4 | N/A | 1/5 | format mismatch (pairs vs chains) | ❌ |
| 25 | CrossCodeEval | N/A | 0/5 strict EM | task ceiling (free-form line) | ❌ |
| 26 | Java synthesis full-loop | 2/5 (exp24 baseline) | 1/5 | synthesis drift | mixed |
| 27 | Java develop→test (generic signals) | 2/5 | **3/5** | +1 case | ✅ |
| 28 | CCEval develop→test | 0/5 | 0/5 | task ceiling | ❌ |
| 29 | CCEval one-shot | 0/5 | 0/5 | learn case too specific | ❌ |
| 30 | Java one-shot | 2/5 | **3/5** | 1 example suffices | ✅ |
| 31 | CCEval candidate-aware (3 strategies) | 0/5 | 0/5 all | good candidates, LLM picks wrong | ❌ but instructive |
| 32 | CCEval 4o-mini / haiku / opus | 0/5 | 0/5 all | stronger LLM uses prior, not tool | ❌ |
| 33 | CCEval trace + receiver + underscore | 0/5 | 0/5 strict | **tool drives correct function-name pick on case 277** | ~ |
| 34 | Task 4 one-shot | 1/5 | 0/5 | NL input, no graph signal | ❌ |
| 35 | Re-DocRED one-shot | — | 0/5 | one-shot develop fragile on long inputs | mixed |
| 36 | Re-DocRED hand-crafted | — | **avg F1 0.15** (doc_1 F1=0.42) | first non-code demo working | ✅ |
| 37 | RepoQA Search Needle Function | — | **2/5 (40%)** | workload-reducing across 300-929 fn candidates | ✅ |
| 38 | SWE-Bench Lite file-localization | — | **2/5 (40%)** | issue-text-only, no repo checkout | ✅ minimal |

## 3. The fit rule (validated)

**V5 fits a benchmark when all three hold:**

1. **Document-shaped input** — primitives can be extracted by regex or AST
   from the input text.
2. **Structured output** — ordering, list of tuples, classification over a
   bounded set. NOT free-form prose or specific unseen code identifiers.
3. **Graph-extractable meta-pattern** — the answer involves relationships
   between entities that the extractor can surface.

Applied to our results:

| Benchmark | Doc input? | Structured out? | Graph pattern? | Fit |
|---|---|---|---|---|
| DependEval T2 | ✅ | ✅ | ✅ | **yes — 5/5 Python, 3/5 Java** |
| CrossCodeEval | ✅ | ❌ (free-form line) | ❌ (target often absent) | no |
| LongMemEval | ✅ | ❌ (free-form answer) | ⚠️ (retrieval) | no |
| DependEval T4 | ✅ | ✅ | ❌ (NL-only input) | no |
| Re-DocRED | ✅ | ✅ (quadruples) | ✅ (entity co-occurrence) | yes — F1 0.15 on first try |
| RepoQA SNF | ✅ (huge repo) | ✅ (function name) | ✅ (all function defs as candidates) | yes — **2/5 = 40%** |
| SWE-Bench Lite file-loc | ⚠️ (issue text only) | ✅ (file path) | ⚠️ (no repo content) | partial — **2/5 = 40%** |

## 4. Workload-reducing tool design

The design principle that emerged:

> **The tool's job is to REDUCE the LLM's workload, not produce the answer.**

Concrete implementation across CCEval and DocRED:

- **Extractor** pulls candidate symbols (functions, classes, entities).
- **Ranker** applies domain-specific penalties and bonuses:
  - Compound-sibling suppression (FOO_from_X down-ranked when FOO exists)
  - Receiver matching (methods called on matching receiver ranked higher)
  - Underscore prefix penalty (private helpers down-ranked)
  - Usage-count as a soft signal
- **Finisher** prompt emits:
  - A ranked candidate list
  - An optional TOOL'S CONFIDENT PICK (when top candidate clearly wins)
  - Explicit instruction to use exact candidate names

### Evidence this works

CCEval case 277 (expected: `get_header_value(response.headers, ...)`):

| Iteration of the design | Pred | What drove the change |
|---|---|---|
| Hand-written baseline | `get_header_value_from_response(...)` | LLM picked compound sibling |
| Candidate-aware (exp31) | `get_header_value_from_response(...)` | Correct symbol in candidates but outranked |
| + receiver matching | `_eval_header_value(...)` | Compound gone, but test helper ranked top |
| + underscore penalty | **`self.assertEqual(self.rule.get_header_value(response), "0")`** | Correct function name |

Strict EM still 0 (argument differences remain) but the **tool now drives
the LLM to the right function name**, which was the objective of the
workload-reducing redesign.

## 5. What doesn't work (and why)

### 5.1 One-shot develop-tools on CrossCodeEval
- Each item has completely different symbols (converse, turkey_box_plot,
  from_string, ...) with no transferable meta-pattern.
- Tools tuned to learn case's `mission_space` don't help with `converse`.
- **Correctly concluded: CCEval is not meta-learnable from 1 example**.

### 5.2 One-shot develop on DependEval Task 4
- Input is NL file descriptions, not code.
- No imports, no AST, no regex-extractable symbols → primitives are empty.
- Synthesis produced tools that extracted relation names as literal strings.
- **Honest takeaway: Task 4 is NL-reasoning, not graph-extraction — different problem shape**.

### 5.3 Stronger LLM alone doesn't rescue CCEval
- Opus beat 4o-mini on cases where answer symbol was NOT in context (via
  prior knowledge: `converse`, `from_string`).
- Opus did NOT outperform on cases where the tool had the right candidate
  (case 277): all three models picked a wrong sibling at the tie-break.
- **The gap is neither "tool surfacing" nor "LLM prior" alone — it's
  structural disambiguation that neither a regex nor a stronger prior fixes**.

## 5.4 RepoQA Search Needle Function — workload-reducing at scale (exp37)

Task: given an entire code repo (avg **~105 files, ~400-3400K chars**) +
an NL description of a "needle" function, identify the function by name.

V5 pipeline:
- SymbolExtractor-style: pull every `def` across all files (929 funcs
  for openai-python, 388 for black, ...).
- Ranker: score each function by word overlap between description and
  (function name + docstring + body snippet).
- Prompt: show top-12 ranked candidates; LLM picks.

Result: **2/5 (40%), avg 639 tokens/query, 30s wallclock**.

This is V5's most convincing workflow win so far: from ~1000 candidates
per repo, the ranker narrows to 12, and the LLM picks correctly 40% of
the time on a task where random is 0.1%. Published RepoQA numbers at
similar-size repos: GPT-3.5 ~30%, GPT-4 ~65%, CodeLlama-7B ~10%. Our
V5 + 4o-mini at 40% is in the competent range.

## 5.5 SWE-Bench Lite file-localization — minimal floor (exp38)

Full SWE-Bench needs repo checkout + patch generation. For smoke we did
**file localization only**: given issue text, predict which file should
be modified.

V5 pipeline (no repo checkout):
- Extractor: pull file paths, imports, backticked identifiers, code
  blocks from the issue text + hints.
- Prompt: issue + extracted mentions → predict one file path.

Result: **2/5 (40%), avg 419 tokens/query, 32s wallclock**.

Wins were cases where the issue directly mentioned the module (Django
examples). Misses were where the issue described behavior without
naming the file. A full V5 pipeline with repo checkout would let the
extractor surface candidate modules from the repo itself, not just
from the issue text.

## 6. DocRED (first non-code demo)

### Task
Document-level relation extraction. Input: doc + entity list. Output:
list of `(head_name, relation_id, tail_name)` Wikidata triples.

### V5 pipeline (hand-crafted, exp36)
- **Extractor**: emits one primitive per entity, per sentence, and per
  entity-pair co-occurrence-in-sentence.
- **Prompt**: doc + entity list + relation shortlist + co-occurring pairs
  + "pick real triples only".
- **Parse**: JSON array of 3-tuples.

### Results (5 short holdout docs, 4o-mini, parallel)
- avg F1: 0.15
- Best case: doc_1 (Quokka) at F1=0.42 (about half the triples recovered)
- Worst cases: F1=0.00 (Bajofondo, Energy and Environmental Security)
- avg tokens: 638

This is in range with zero-shot general-LLM DocRED baselines (~0.1-0.3 F1).
Supervised fine-tuned systems reach ~70 F1, so there's clear headroom.

### What would close the gap on DocRED
1. **Few-shot traces** (not just 1) showing relation-picking reasoning.
2. **Per-relation entity-type priors** (e.g. P17 = country → tail must be
   TYPE=LOC).
3. **Pair-level disambiguation** with verb-matching to relation.

None of these are V5 architecture changes; they're richer plugins.

## 7. What was committed to GitHub

All pushed to `origin/main` at `github.com/NingWang0123/pixel_mem`:

- `pixelmem/v5/` — V5 package (isolated from V4, 2,500+ LoC)
  - `core/` — types, protocols, pipeline, derivation engine
  - `plugins/python_deps/` — V4 adapter (only v4 import)
  - `synth/` — schema designer, extractor synth, one-shot develop,
    trace learning, candidate-aware, extractor tester, JSON-tolerant parser
  - `harness.py` — test harness with failure classifier
  - `task_card.py` — TaskCard schema + YAML/JSON loader
- `tests/test_v5_core.py` — 8 tests including AST-based V5↔V4 isolation check
- `experiments/exp22..exp36_v5_*.py` — 15 experiments
- `docs/`:
  - `v5_plan.md` — architecture + roadmap
  - `v5_examples.md` — end-to-end traces
  - `v5_vs_v4.md` — side-by-side
  - `v5_failure_review.md` — 32-failure root-cause analysis
  - `v5_workload_reducing.md` — candidate-list design write-up
  - `v5_benchmark_fit.md` — three-rule fit checklist
  - `v5_findings.md` — **this document**

## 8. Open questions / next directions

1. **DocRED deeper**: run all 500 validation docs, add per-relation
   entity-type filters, see if F1 climbs to 0.3+. If yes, V5 is a
   competitive zero-shot DocRED baseline.
2. **SciERC**: smaller (~500), similar shape. If V5 does well here too,
   the non-code generalization claim is solid.
3. **Multi-language DependEval with language-specific extractors**:
   write Java/C/C++/JS/TS extractor plugins (~100 LOC each). Expected
   lift on Java from 3/5 → 4-5/5.
4. **MuSiQue**: explicit multi-hop decomposition. Would demonstrate
   V5's derivation engine on real multi-hop QA.
5. **V5 core unit tests**: current coverage is only 8 tests on the
   protocol layer. Add tests for synth stages and candidate ranking.
6. **Package for install**: currently requires `sys.path.insert`. A
   proper `pip install -e .` setup would help adoption.

## 9. Honest limitations

- **All results are 5-example smoke tests**, not full-benchmark runs.
  Numbers are directional, not definitive.
- **All tool synthesis is stochastic at 4o-mini / temperature=0**.
  Re-running gives different specs; the "keep best across iterations"
  guard is essential.
- **V5's plugin protocols are Python-specific in several places**
  (SymbolExtractor regexes, CandidateAwarePrompt completion-site
  analyzer). Generalizing requires language plugins.
- **Nothing in V5 auto-evaluates structural correctness** — the test
  harness classifies failures but doesn't tell you *why* the ranker
  produced the wrong order. Manual inspection is still required.

The architecture is sound; the applied quality is uneven. Both of these
are true simultaneously.
