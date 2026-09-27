# V5 — Workload-Reducing Tool Design

> Historical experiment note: versioned module paths and directory layouts below refer to the pre-restructure repository. See [Quill README](../README.md) for the current layout.

Follow-up to `docs/v5_failure_review.md` on the question: **is V5's extractor too weak for CrossCodeEval, or is the task the ceiling?**

## Reframe

Instead of the tool attempting to *produce* the answer on every case, the tool's job is to **reduce the LLM's workload** — extract candidate symbols from whatever context is available, rank them by relevance to the completion site, and hand the LLM a short list. The LLM still writes the final line, but with the search space shrunk.

This matches how humans actually use IDE tooling: grep narrows the candidate pool; judgment picks the right one.

## Implementation (new in this commit)

Two new components under `pixelmem/v5/synth/candidate_aware.py`:

### `SymbolExtractor`

Pulls bare symbol names (not full lines) from Python documents:

| Relation | What it emits |
|---|---|
| `defines_function` | `def X` (top-level) — also catches `# def X` in commented-out code |
| `defines_class` | `class X` |
| `imports_symbol` | `from x import Y` → `Y` |
| `imports_module` | `import x.y` / `import x as y` |
| `assigns_local` | top-level `X = …` |
| **`used_as_method`** | `.X(` usage sites (with count) |
| **`used_as_function`** | bare `X(` calls (with count) |

The usage-site patterns were the key addition — in case 277, the correct symbol `get_header_value` only ever appears in **commented-out** example assertions inside the crossfile context. Comment-aware extraction is necessary to catch it.

### `CandidateAwarePrompt`

1. **Analyzes the completion site** on the focal file tail — is this a method call on a receiver? Attribute access? Argument position? Bare statement?
2. **Ranks extracted symbols** by:
   - Completion-site compatibility (functions rank higher at `recv.METHOD(`)
   - Focal-file recency (symbols already visible nearby)
   - Usage-count signal (frequently-referenced symbols)
3. **Emits a prompt** that shows the top-K candidates + completion-site kind + a picking rule:
   - Info present → "use the exact candidate name inside a full continuation"
   - Info absent → "no candidate fits; keep the guess short and plausible"

## Results (5 holdout × 4o-mini × 3 strategies)

| Strategy | strict EM | partial | avg tokens |
|---|---|---|---|
| A. baseline (hand-regex + generic prompt) | 0/5 | 0.00 | 521 |
| B. SymbolExtractor + raw-primitives prompt | 0/5 | 0.00 | 934 |
| C. SymbolExtractor + **CandidateAwarePrompt** | 0/5 | 0.00 | 625 |

Zero strict EM movement. But the pred SHAPES changed meaningfully — see the per-case analysis.

## Per-case diagnosis (why EM still zero)

### Case 277 — `get_header_value` (info in context)

- **Expected**: `get_header_value(response.headers, self.rule.HEADER_NAME), "0")`
- **Candidate-aware pred**: `assertEqual(self.rule.get_header_value_from_response(response, self.rule.HEADER_NAME), "0"))`
- **Tool-level diagnosis (traced manually)**: Both `get_header_value` (score 49) AND `get_header_value_from_response` (score 57) appear in the top-15 ranked candidates. The sibling outranks the target because the sibling has higher usage count across docs (4 vs 3).
- **LLM-level failure**: 4o-mini picks the higher-scored sibling even though the correct answer is structurally distinguishable by argument type (`response.headers` vs `response`).

### Case 364 — `graph_view` (info in context)

- **Expected**: `graph_view, selected, vty)`
- **Candidate-aware pred**: a plausible-shaped continuation, but LLM didn't pick `graph_view`. Completion site is `arg_position` (trailing `,`) — `graph_view` should have surfaced as a `used_as_function` or `assigns_local` candidate. It did appear 4× in crossfile. The LLM chose an unrelated candidate from the top-K.

### Cases 1440 / 206 / 441 / 468 — info NOT in context

- `converse`, `turkey_box_plot`, `zeros`, `from_string` — none of these symbols appear anywhere in the focal file OR the crossfile context.
- **Tool correctly does not surface them** (can't extract what isn't there).
- **Candidate-aware prompt tells the LLM "no candidate fits strongly"**, which led to the LLM producing hedged guesses (e.g., `up_threshold = Utils.append(freq, search_range)`) instead of made-up function names.
- These are still wrong because the LLM has no information about the right symbol — by design.

## Direct answer to "is the tool too weak?"

**Partially yes, partially no.**

- **Yes on surfacing**: before this commit, the correct answer symbol wasn't even in the extracted primitives (case 277). With usage-site extraction it IS surfaced. The tool has stopped being the bottleneck on surfacing.
- **No on tie-breaking**: the LLM at inference time is now the bottleneck for picking between `get_header_value` and `get_header_value_from_response`. That's an LLM-capability issue (or a prompt-engineering issue), not a tool-building issue.
- **Ceiling on 3/5 cases**: the ground-truth symbol is nowhere in the provided prompt. Neither hand-written nor LLM-built nor workload-reducing tools can extract what isn't there. CCEval's `rg1_bm25` retrieval slice simply doesn't retrieve the defining file for these cases.

## Follow-ups that could move EM

In rough order of expected lift:

1. **Structural signature matching.** Extract `def X(a, b, c)` *with parameter list*, then rank candidates whose arity matches the number of args already visible in the focal tail. For case 277, `get_header_value(headers, name)` would rank higher than `get_header_value_from_response(response, name)` because the focal tail passes `response.headers` (headers first).
2. **Sibling suppression in the ranker.** When candidate `FOO_from_X` exists and candidate `FOO` also exists, de-rank the compound name in favor of the simpler one on the assumption that compound names are alternative code paths.
3. **Stronger LLM for the pick step.** Keep the tool the same; swap 4o-mini → sonnet. This costs more per query but the tie-breaking between similar names is exactly where a stronger model has most leverage.
4. **Change the retrieval setting.** CCEval's `rg1_openai_cosine_sim` variant retrieves different crossfile snippets; some previously-unsolvable cases may become solvable. Full repo indexing (V5 runs its own Extractor over the whole repo) is the strongest lift but changes the benchmark setup.

None of the above is a V5 architectural change. All slot in as additional `Extractor` features, `PromptTemplate` tweaks, or a swap of the `LLMCaller`.
