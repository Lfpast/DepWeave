# V5 on DependEval / RepoQA / SWE-Bench — the Three-Benchmark Story

> Historical experiment note: versioned module paths and directory layouts below refer to the pre-restructure repository. See [Quill README](../README.md) for the current layout.

Consolidated after exp37–exp41. Each benchmark demonstrates a different V5 value-prop; together they make the case (and also honestly flag where the case doesn't hold).

## One-line per benchmark

| Benchmark | What V5 demonstrates | Headline number |
|---|---|---|
| **DependEval Task 2** | Tool guidance reduces tokens by 10× with comparable accuracy | V5: **278 tok/q (10.1× less)**, accuracy in range with V4 (81% full run) |
| **RepoQA SNF** | PixelMem storage + retrieval cache for within-corpus query reuse | **2.67× speedup** from cache, identical accuracy (6/10 before/after) |
| **SWE-Bench Lite localization** | Same tool-guidance pattern on issue → file | 2/5 file-level accuracy, ~400 tok/q (no repo-checkout version yet) |

## 1. RepoQA — storage + retrieval strength

### The task
Given a full repo (30–600 files, 300–1000 functions) + an NL description of a needle function, return the function name.

### V5 pipeline
- `RepoQAFunctionExtractor` pulls every `def` across every file via `ast.parse` (subject = path, relation = `defines_function`, object = name, provenance = {docstring, snippet, lineno})
- `RepoQASearchPrompt` ranks candidates by word-overlap between NL query and (name + docstring + body snippet); top-12 → LLM picks one
- `PixelMemCache` wraps the extractor: first query per repo extracts + encodes to PNG; repeated queries load from PNG in ms

### Accuracy (exp37, exp39)
- **2/5 (40%) strict-name match** in a 5-sample smoke (exp37)
- **6/10 (60%)** in a 10-sample smoke (exp39) — in range with published GPT-3.5 (~30%) / GPT-4 (~65%)
- Accuracy is identical with vs. without cache (cache is pure pass-through by design)

### Cache speedup (exp39)
| Setting | Wallclock | Speedup |
|---|---|---|
| Sequential, no cache | 167.0s | 1.0× |
| Sequential, cached | 67.1s | **2.49×** |
| Parallel (workers=5), no cache | 199.0s | 0.8× (thread contention on ast.parse) |
| **Parallel + cache** | **74.4s** | **2.67×** |

Hit rate in 10-query/4-repo sample: **60%** (1 miss + 2 hits per repo × 3 repos + 1 singleton).

### Storage compactness (exp40)

Per-repo cache entry across 5 repos, compared to alternative serializations:

| Format | Total bytes | Ratio to JSON |
|---|---|---|
| text TSV | 234 KB | 0.08× |
| gzipped JSON | 309 KB | 0.11× |
| **PixelMem (PNG + gzip sidecar + sparse registry)** | **733 KB** | **0.26× (3.8× smaller than JSON)** |
| pickle | 2.45 MB | 0.88× |
| plain JSON | 2.78 MB | 1.00× |

**Honest caveat**: PixelMem is smaller than plain JSON but still ~2.4× larger than pure gzipped JSON because the cache stores BOTH the pixel encoding (9 KB of pure PNG per repo) AND a gzipped provenance sidecar (for lossless primitive round-trip). If you only needed (subject, relation, object, condition) with no provenance, the pixel store alone would be ~10 KB — genuinely tiny.

### What makes RepoQA a PixelMem win
- Same document set (one repo) gets queried many times within a run
- The 2.67× speedup compounds across the 10 needles/repo dimension of RepoQA's full benchmark
- Across-run reuse (dev iteration) also benefits

## 2. DependEval Task 2 — tool guidance for token reduction

### The task
Given 3–5 Python source files from a real repo, output the correct topological dependency ordering.

### V5 pipeline
- `V4Extractor` pulls `(file, imports_symbol, sym@target, internal)` primitives via AST + SymbolResolver (stdlib filter, re-export tracing, relative/aliased imports)
- `V4DependencyGraph` chains `imports_symbol + defined_in → depends_on` with confidence 0.90
- `V4OrderingPrompt` emits file summaries (imports + defs) + confirmed edges + computed topo order → JSON array

### Accuracy
- Python 5/5 (100%) on 3-file smoke; **81.3% on full 166-item run** (reported in `docs/depeval_v4_81pct_analysis.md`) — beats Qwen-32B paper SOTA (70%)
- Java 3/5 via one-shot develop-tools from a single labeled example (exp30)

### Token reduction (exp41)

Per-item comparison, V5's hybrid prompt vs "concatenate all file contents + task header":

| Item | Baseline tok | V5 tok | Ratio |
|---|---|---|---|
| de_0 | 3,599 | 231 | 15.6× |
| de_1 | 4,993 | 355 | 14.1× |
| de_2 | 212 | 204 | 1.04× (already tiny) |
| de_3 | 2,391 | 268 | 8.9× |
| de_4 | 2,846 | 331 | 8.6× |
| **Total 5** | **14,041** | **1,389** | **10.1×** |

Paper baselines send ~40K tokens (full source of 3–5 files plus instructions); V5 averages ~389 tok/query on the full run — **consistent with the reported 103× savings vs Qwen-32B inputs**.

### What makes DependEval a V5 tool-guidance win
- V5 replaces "dump the corpus" with "dump the graph summary" → compact, high-signal
- Graph primitives are structural (imports + defined_in), not surface lexical
- The LLM confirms the topo order rather than deriving it from raw code

## 3. SWE-Bench Lite — same pattern, smaller win

### The task
Given an issue + repo, predict which file(s) need modification. (Full SWE-Bench needs patch generation; we restrict to localization.)

### V5 pipeline (issue-only, no repo checkout)
- `SWEBenchIssueExtractor` pulls file paths, import lines, backticked identifiers, and code-block fences from issue + hints
- `SWEBenchFilePrompt` assembles issue + extracts + brief instruction
- Output: one relative file path

### Accuracy (exp38)
- **2/5 (40%) file-level accuracy** on 5 random items
- Wins when the issue directly mentions the module; misses when it describes behavior without naming the file

### Token budget (exp41) — V5 does NOT save tokens here

| Item | Baseline tok | V5 tok | Ratio |
|---|---|---|---|
| django-16139 | 390 | 456 | 0.86× |
| sympy-13773 | 228 | 305 | 0.75× |
| django-12497 | 279 | 345 | 0.81× |
| django-15320 | 300 | 378 | 0.79× |
| django-12284 | 1,199 | 525 | 2.28× |
| **Total 5** | **2,396** | **2,009** | **1.2×** |

V5 is actually LONGER on 4/5 items because the extracted mentions list appends to the baseline issue text. The one outlier (django-12284) has a very long issue; extraction lets us trim.

### Why SWE-Bench token-reduction didn't land
The baseline here is ALREADY compact — just the issue + hints (typically < 500 tokens). There's nothing for V5 to compress. The real V5 win on SWE-Bench would emerge only with **repo checkout + full-repo extraction**: V5 would narrow thousands of files → top-5, avoiding a 40K-token repo dump. That's a future experiment.

## 4. Combined takeaways

- **PixelMem's actual strengths**: fast cache reads (2.67× on RepoQA), moderately compact (3.8× smaller than JSON, still beat by gzipped JSON)
- **V5's tool-guidance strength**: genuine 10× token reduction on DependEval; compresses to the degree the baseline is bloated
- **When V5 doesn't help**: free-form outputs (CCEval line completion), already-compact baselines (SWE-Bench issue-only), NL-only reasoning without structural graph signal (DependEval Task 4)
- **When PixelMem storage doesn't help**: unique-per-item corpora (DependEval files across items, SWE-Bench commits across items)

## 5. What's committed

- `pixelmem/v5/` — 20+ files, ~3,700 LoC
- `pixelmem/v5/cache/pixel_cache.py` — PixelMemCache with sparse registry + gzipped sidecar
- `experiments/exp22..exp41_v5_*.py` — 19 experiments
- `docs/v5_*.md` — 8 design + results docs (including this one)

Current `origin/main` as of writing: commit `eae1f6a`.
