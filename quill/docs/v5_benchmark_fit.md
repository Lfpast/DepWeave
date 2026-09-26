# V5 Benchmark Fit — Which Benchmarks Match the Method

> Historical experiment note: versioned module paths and directory layouts below refer to the pre-restructure repository. See [Quill README](../README.md) for the current layout.

Updated after exp34 (DependEval Task 4 one-shot = 0/5).

## Rule of thumb

V5 fits a benchmark when **all three** are true:

1. **Input is document-shaped**: code files, text passages, structured records — something primitives can be extracted from.
2. **Output is structured**: ordering, classification, list of tuples, selection from candidates. **NOT** free-form prose, novel code, or strings whose content isn't inferrable from inputs.
3. **Has a graph-extractable meta-pattern**: the answer involves relationships between entities (imports, relations, calls, citations). V5's derivation chains encode these.

Lose any one → V5's architecture doesn't help. Two examples:

- **CrossCodeEval** breaks (2): predicted line is often a specific function name not in the prompt — free-form output bottlenecked by LLM prior, not by what primitives surface.
- **DependEval Task 4** breaks (3): input is NL file descriptions, output is dependency chains — but the "graph" has to be inferred semantically from NL (which files share concepts?), not extracted structurally from code.

## Evidence so far

| Benchmark | V5 score | Shape | Fit verdict |
|---|---|---|---|
| DependEval Task 2 Python | **5/5** | code → ordering | ✅ perfect |
| DependEval Task 2 Java | **3/5** one-shot | code → ordering | ✅ good |
| DependEval Task 2 TypeScript | 4/5 V4 wrapper | code → ordering | ✅ good |
| DependEval Task 2 JS / C / C++ | 2–3/5 V4 fallback | code → ordering | ⚠️ needs lang extractor |
| CrossCodeEval | 0/5 strict EM | code → free-form line | ❌ rule 2 violated |
| LongMemEval | 2/5 fuzzy | chat → free-form answer | ❌ rule 2 violated |
| DependEval Task 4 (exp25 + exp34) | 0–1/5 | **NL desc → chain list** | ❌ rule 3 violated (no code signal) |

## Top-ranked V5-fitting benchmarks to try next

### Tier A — strong fit, worth the effort

| # | Benchmark | Input → Output | Why it fits |
|---|---|---|---|
| 1 | **DocRED** | document + entity spans → list of `(head, rel, tail, evidence)` | Output IS V5's native quadruple. ~5k docs, HF accessible. |
| 2 | **SciERC** | scientific abstract → entities + typed relations + coref clusters | Small (~500), structured triples, two-hop coref reasoning. |
| 3 | **MuSiQue** | question + 20 paragraphs → 2-4 hop sub-question decomposition + answer | Explicit derivation chain is V5's derivation engine 1:1. |

### Tier B — decent fit but shallower

| # | Benchmark | Note |
|---|---|---|
| 4 | 2WikiMultihopQA | Triple-path reasoning, but answer is a free-form span. |
| 5 | ChemProt / DDI | Relation classification from a fixed label set. No graph construction step — primitives are given. |
| 6 | MetaQA (1/2/3-hop) | Graph is pre-built, V5 doesn't need to extract it. Tests derivation engine only. |
| 7 | DependEval Task 1 | File identification — selection from candidates, one-hop. |
| 8 | BFCL (function calling) | Structured output, but no graph/multi-hop. |

### Tier C — known bad fit, skip

| # | Benchmark | Why not |
|---|---|---|
| — | RepoBench v1.1 | Same ceiling as CCEval (line completion). |
| — | SWE-Bench | Free-form patch generation. |
| — | ClassEval / HumanEval | Free-form code generation. |
| — | WebQSP / CWQ | Requires SPARQL generation; V5 doesn't do KG query generation. |

## What the Task 4 result actually means

Task 4 looked V5-shaped but wasn't. The input is `{file_path: function_description}` NL text. There's no code to AST-parse, no imports to regex. The "graph" (which files depend on which) is inferred *semantically* from NL descriptions.

V5's primitive extractor pulled `(file, has_function_description, "<NL blob>")` triples — correct by spec but useless for dependency inference, because the LLM would need to reason over the NL blobs themselves to infer structure. V5's derivation engine doesn't run on NL content; it runs on atomic symbols.

This is a legitimate mismatch — not a V5 bug, a task-shape mismatch. A different V5 pipeline *could* fit Task 4 (LLM-extracts-concepts-per-file primitive, concept-overlap derivation rule, chain composer), but it'd be a different plugin set tuned for semantic matching — not the same one we use for Task 2.

## Recommended next experiment

**DocRED** with V5's trace-learning + candidate-aware architecture:

- Two hand-annotated trace examples showing entity extraction + relation classification.
- ReducerTool extracts entity mentions + sentence pairs.
- FinisherTool composes `(head, rel, tail, evidence)` quadruples.
- This is the single experiment that would most convincingly demonstrate V5 on a non-code domain.

SciERC would be a sharper, smaller version of the same test.
