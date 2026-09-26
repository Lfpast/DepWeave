# PixelMem V4: 81% on DependEval — Current State and Remaining Failures

## Results Summary

| Method | Accuracy | 3-file | 4-file | 5-file | Tokens |
|--------|----------|--------|--------|--------|--------|
| Paper SOTA (Qwen2.5-Coder-32B) | 70% | — | — | — | ~40,000 |
| V3 baseline | 66% | 81% | 59% | 25% | 347 |
| **V4 (current)** | **81%** | **87%** | **76%** | **75%** | 389 |

## What Got Us from 67% to 81%

### Resolver fixes (+13pp: 67% → 80%)
Four bugs in the import resolver were causing 30+ "ambiguous" failures where edges existed in the code but the resolver missed them:

1. **Wildcard imports skipped**: `from .FlatlandModel import *` was completely ignored. Now creates a module-level dependency edge.

2. **Relative imports not resolved**: `from .auth import OAuth2` — the dot-stripped module name `auth` wasn't in the module map. Fixed by registering basename stems and parent directory names.

3. **Deep package paths missed**: `import skipper_lib.workflow.workflow_helper` — module map only had full paths. Fixed by adding last-segment and suffix matching.

4. **`import X as Y` not handled**: `import models.backbone.dino_vision_transformer as dino_vits` — the `as` alias broke parsing. Fixed by stripping alias before resolution.

### Stdlib filtering (+1pp: 80% → 81%)
`from types import SimpleNamespace` was matching local `types.py` (stdlib collision). The SymbolResolver detects these by checking if the top-level module is in the stdlib set AND the imported symbol is not defined locally.

## Remaining 31 Failures

### Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| **INIT_CYCLE** | 10 | File imports symbol through package `__init__.py`; `__init__.py` re-exports from submodule; creates false bidirectional edge |
| **OTHER_CYCLE** | 8 | Mutual imports between non-init files, or complex package-level cycles |
| **LLM_ERROR** | 6 | Graph is correct (no false edges) but LLM picks wrong order |
| **AMBIGUOUS** | 7 | No import between the misordered pair in the code |

### INIT_CYCLE: 10 cases (the main remaining problem)

**Pattern**: File A does `from package import Symbol`. Our resolver maps `package` to `__init__.py`. But `__init__.py` doesn't define `Symbol` — it re-exports it via `from .submodule import Symbol`. This creates a false edge `A → __init__.py`, while `__init__.py` also has a real edge `__init__.py → submodule`. If A and submodule are both in our file set, we get a cycle.

**Example Q113**:
```
composite_encoder.py:  from fairseq.models import FairseqEncoder
__init__.py:           from .fairseq_encoder import FairseqEncoder  (re-export)
fairseq_model.py:      (defines BaseFairseqModel, no local imports)

Our graph:
  composite_encoder → __init__.py (FALSE: FairseqEncoder is re-exported, not defined here)
  __init__.py → composite_encoder (TRUE: __init__ re-exports CompositeEncoder)
  → CYCLE

GT: [composite_encoder, __init__, fairseq_model]
Got: [composite_encoder, fairseq_model, __init__]
```

**Example Q148**:
```
transformer.py:  from praxis import pax_fiddle  (praxis = external library, NOT __init__.py)
module.py:       from praxis import pax_fiddle  (same — external)
__init__.py:     from .module import FusedSoftmax; from .transformer import DotProductAttention

Our graph:
  transformer → __init__.py (FALSE: pax_fiddle is EXTERNAL, not from __init__)
  module → __init__.py (FALSE: same reason)
  __init__ → transformer (TRUE)
  __init__ → module (TRUE)
  → CYCLE

GT: [module, __init__, transformer]
```

**Why re-export tracing didn't work**: We tried tracing through `__init__.py`'s symbol table to redirect edges to the actual source file. But:
- When the re-exported symbol comes from a file NOT in our 3-5 file subset, the trace fails
- When `__init__.py` imports from external packages (not submodules), we can't distinguish "re-export from local file" vs "import from external library"
- Conservative fallback (keep original edge) preserves false edges; aggressive fallback (drop edge) removes valid edges

**The fundamental difficulty**: To resolve `from package import Symbol` correctly, we need to know:
1. Is `Symbol` defined in `__init__.py` itself? → edge to `__init__.py` ✓
2. Is `Symbol` re-exported from a sibling in our file set? → edge to sibling ✓
3. Is `Symbol` re-exported from a file NOT in our set? → edge to `__init__.py` (imprecise but best we can do)
4. Is `Symbol` from an external library with a name collision? → NO edge (hardest to detect)

Case 4 is what breaks Q148: `from praxis import pax_fiddle` where `praxis` matches our `__init__.py` but `pax_fiddle` is an external library.

### OTHER_CYCLE: 8 cases

**Pattern**: Bidirectional imports between non-`__init__` files, or imports through package names that create unexpected edges.

**Example Q33**:
```
validation.py:  from .disk_volume import DiskVolume    → validation depends on disk_volume
disk_volume.py: from .validation import validate_name  → disk_volume depends on validation
→ TRUE CIRCULAR DEPENDENCY in the code

GT says: [image_uri, validation, disk_volume]  (validation before disk_volume)
```

These are actual circular imports in the Python code. The GT ordering reflects which file was written first (temporal order), not a DAG — because there IS no DAG.

**Example Q126**:
```
more_comments.py:  from .listing import MoreChildren  → more_comments depends on listing
listing.py:        from .more_comments import MoreComments  → listing depends on more_comments
→ TRUE CIRCULAR DEPENDENCY

GT: [submission, more_comments, listing]
```

For true circular dependencies, there is no "correct" topological order. The GT reflects the developer's construction order.

### LLM_ERROR: 6 cases

The dependency graph is correct (no false edges, no cycles) but the LLM verification step reorders files incorrectly.

**Example Q61**:
```
Graph correctly has: escprober → escsm (escprober imports from escsm)
Topo sort correctly puts: escsm before escprober
LLM SWAPS them: outputs [constants, escprober, escsm, ...]
GT: [constants, escsm, escprober, ...]
```

The LLM sees both files import from `constants` and treats them as interchangeable, ignoring the `escprober → escsm` edge shown in the "Confirmed dependencies" section.

**Example Q42**:
```
Graph has: main → builder, main → state, launcher → state
Missing: builder → launcher (no import between them)
Topo puts: [builder, state, launcher, main]
GT: [state, launcher, builder, main]
```

The tie between builder and launcher is broken by alphabetical order in our topo sort, but GT has the opposite.

### AMBIGUOUS: 7 cases

No import exists between the misordered pair. The GT ordering comes from the full repository DAG (files we don't have access to).

## Theoretical Ceiling

| Category | Count | Could fix? |
|----------|-------|------------|
| Correct | 135 | — |
| INIT_CYCLE | 10 | **Partially** — need package-vs-external disambiguation |
| OTHER_CYCLE | 8 | **No** — true circular imports, GT is temporal |
| LLM_ERROR | 6 | **Partially** — better tie-breaking or constrained decoding |
| AMBIGUOUS | 7 | **No** — ordering info only in full repo |

**Optimistic ceiling**: ~151/166 (91%) if we fix all INIT_CYCLE + LLM_ERROR
**Realistic ceiling**: ~145/166 (87%) accounting for partial fixes
**Hard ceiling**: ~150/166 (90%) — 8 OTHER_CYCLE + 7 AMBIGUOUS are unfixable from our subset

## Architecture Summary

```
Input: 3-5 Python files with full source code

Step 1: AliasNamespace — deduplicate files/symbols
Step 2: PrimitiveExtractor — AST-based, ~96 quadruples per question
Step 3: SymbolResolver — stdlib filtering
Step 4: PixelMem storage — encode into PNG matrices
Step 5: DependencyGraph — chain imports_symbol + defined_in
Step 6: NaturalLabeler — hide internal aliases
Step 7: Hybrid LLM prompt — raw imports + confirmed graph edges
Step 8: Parse + fallback

Avg: 389 tokens/query (~103x fewer than paper baselines)
```

## Key Files
- `pixelmem/v4/alias_namespace.py` — deduplication (297 lines)
- `pixelmem/v4/primitive_extractor.py` — AST extraction (509 lines)
- `pixelmem/v4/dependency_graph.py` — graph builder (510 lines)
- `pixelmem/v4/symbol_resolver.py` — stdlib + re-export analysis (249 lines)
- `pixelmem/v4/natural_labels.py` — alias-free labels (187 lines)
- `pixelmem/v4/reconstruction.py` — evidence objects (268 lines)
- `pixelmem/v4/retrieval.py` — pipeline + hybrid prompt (530 lines)
- `tests/test_v4.py` — 14 tests (499 lines)
- `experiments/exp21_depeval_v4.py` — benchmark script
