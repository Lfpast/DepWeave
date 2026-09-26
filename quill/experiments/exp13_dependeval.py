"""Exp 13: V2 vs V3 on DependEval Task 2 (10 questions, dependency ordering).

DependEval provides file contents and ground-truth dependency ordering.
We extract imports via AST, encode into PixelMem, then use topo_sort
and LLM to produce the ordered file list.

Both V2 and V3 compared head-to-head. Batched parallel.
"""

import json, os, re, subprocess, sys, time, tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelmem.shard_manager import ShardManager
from pixelmem.triple_extractor import Triple
from pixelmem.workflow.extractor import extract_file
from pixelmem.v2.pipeline import V2ReadPipeline
from pixelmem.v3 import ReadPipeline as V3ReadPipeline, SummaryBuilder
from pixelmem.v3 import retrieval_algebra as ra


def ask_cli(prompt, model="haiku"):
    try:
        r = subprocess.run(
            ["claude", "-p", prompt[:4000], "--max-turns", "1", "--model", model],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        return r.stdout.strip()
    except Exception:
        return "[]"


def parse_dependeval_content(item):
    """Parse DependEval item into file_name → code mapping.

    DependEval format has file headers like:
      'path/to/file.py'
      :code here...
    or sometimes:
      'path/to/file.py':
      code here...
    """
    files = [f.strip("'\" ") for f in item["files"]]
    content = item["content"]
    gt = [f.strip("'\" ") for f in item["gt"]]

    # Build regex to match any of the file paths as headers
    # Headers appear as: 'path/to/file.py'\n: or 'path/to/file.py':
    file_contents = {}
    current_file = None
    current_lines = []

    for line in content.split("\n"):
        # Strip quotes and colons from the line to check for file headers
        stripped = line.strip()
        cleaned = stripped.strip("'\"").rstrip(":")

        matched_file = None
        for f in files:
            f_clean = f.strip("'\"")
            if cleaned == f_clean:
                matched_file = f_clean
                break

        # Also check if line starts with : (continuation of header)
        if not matched_file and stripped == ":" and current_file:
            continue  # skip the colon line after header

        if matched_file:
            if current_file:
                file_contents[current_file] = "\n".join(current_lines)
            current_file = matched_file
            current_lines = []
        elif current_file:
            # Skip leading colon lines
            if stripped.startswith(":") and not current_lines:
                current_lines.append(stripped[1:])
            else:
                current_lines.append(line)

    if current_file:
        file_contents[current_file] = "\n".join(current_lines)

    return files, file_contents, gt


def extract_imports_from_content(files, file_contents):
    """Extract import dependency edges between files.

    Two-pass approach:
    1. Build module→file lookup from our file paths
    2. For each import, check if the FULL import path is a suffix of
       one of our file's module paths (not just basename matching)
    3. Also handle relative imports by resolving relative to source dir
    4. For __init__.py: treat 'from package import X' as importing
       the package's __init__.py if X matches a submodule
    """
    clean_files = [f.strip("'\"") for f in files]

    # Build lookup: full module path → file, and all suffixes
    module_to_file: dict[str, str] = {}
    # Also track which parent packages each file belongs to
    file_packages: dict[str, set[str]] = {}  # file → set of parent package names

    for fpath in clean_files:
        basename = fpath.split("/")[-1].replace(".py", "")
        full_module = fpath.replace("/", ".").replace(".py", "")
        parts = full_module.split(".")

        # Register full path and all suffixes
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            module_to_file[suffix] = fpath

        # Register basename (but only if unambiguous)
        if basename not in module_to_file or module_to_file[basename] == fpath:
            module_to_file[basename] = fpath

        # Track parent packages
        file_packages[fpath] = set()
        for i in range(len(parts) - 1):
            file_packages[fpath].add(".".join(parts[i:]))
            file_packages[fpath].add(parts[i])

    # Also build basename-only lookup for relative import resolution
    basename_to_file: dict[str, str] = {}
    for fpath in clean_files:
        bn = fpath.split("/")[-1].replace(".py", "")
        basename_to_file[bn] = fpath

    # Use FULL PATHS as entity names — disambiguates duplicate basenames.
    # The condition field stores the readable edge label.

    triples = []
    seen = set()

    def _is_internal_import(from_module: str, source_file: str) -> bool:
        """Check if an import references our file set (not an external package)."""
        clean = from_module.lstrip(".")
        if not clean:
            return True  # relative import is always internal
        # Check if any of our files' packages match the import prefix
        for fpath in clean_files:
            pkg = fpath.replace("/", ".").replace(".py", "")
            # Check if the import is a prefix/suffix of one of our module paths
            if clean in pkg or pkg.endswith(clean):
                return True
            # Check if they share a common package root
            parts = pkg.split(".")
            import_parts = clean.split(".")
            if parts[0] == import_parts[0]:
                return True
        return False

    def _resolve_import(from_module: str, name: str, source_file: str) -> str | None:
        """Resolve an import to a target FULL file path.

        Strategy:
        1. Try full qualified path: from_module.name
        2. Try from_module alone (for 'from X import Y' where X is the file)
        3. Try name alone (for 'import X')
        4. For relative imports: resolve relative to source file directory
        5. Only fall back to basename matching if the import looks internal
        """
        clean_mod = from_module.lstrip(".")

        # Priority 1: exact module path matches
        candidates = []
        if clean_mod and name:
            candidates.append(f"{clean_mod}.{name}")
        if clean_mod:
            candidates.append(clean_mod)
        if name:
            candidates.append(name)

        for candidate in candidates:
            target = module_to_file.get(candidate)
            if target and target != source_file:
                return target

        # Priority 2: relative imports — resolve from source directory
        if from_module.startswith("."):
            dots = len(from_module) - len(from_module.lstrip("."))
            source_parts = source_file.split("/")
            if dots < len(source_parts):
                base_parts = source_parts[:len(source_parts) - dots]
                if clean_mod:
                    rel_path = "/".join(base_parts) + "/" + clean_mod.replace(".", "/")
                    # Try as .py file
                    for fpath in clean_files:
                        if fpath.startswith(rel_path) and fpath != source_file:
                            return fpath
                    # Try basename
                    for part in clean_mod.split("."):
                        target = module_to_file.get(part)
                        if target and target != source_file:
                            return target

        # Priority 3: basename matching — but ONLY if the import looks internal
        # This prevents 'from shapely.validation import X' matching our validation.py
        if _is_internal_import(from_module, source_file):
            all_parts = set()
            if clean_mod:
                all_parts.update(clean_mod.split("."))
            if name:
                all_parts.add(name)
            for part in all_parts:
                target = module_to_file.get(part)
                if target and target != source_file:
                    return target

        return None

    # Also extract ALL imports (including external) with different conditions
    # and function definitions for implicit dependency detection
    for fpath, code in file_contents.items():
        # Store function/class definitions
        for line in code.split("\n"):
            stripped = line.strip()
            m = re.match(r'^(?:def|class)\s+(\w+)', stripped)
            if m:
                triples.append(Triple(fpath, "defines", m.group(1), "definition"))

        for line in code.split("\n"):
            line = line.strip()

            # Pattern 1: "from X import Y"
            m = re.match(r'^from\s+(\.{0,3}[\w.]*)\s+import\s+([\w, *]+)', line)
            if m:
                from_module = m.group(1)
                imported_names = [n.strip() for n in m.group(2).split(",")]

                for name in imported_names:
                    name = name.strip()
                    if name == "*":
                        target = _resolve_import(from_module, "", fpath)
                    else:
                        target = _resolve_import(from_module, name, fpath)
                    if target:
                        key = (fpath, target)
                        if key not in seen:
                            seen.add(key)
                            # Store ORIGINAL import line in condition for LLM context
                            triples.append(Triple(
                                fpath, "imports", target,
                                line,  # original import statement
                            ))
                        break
                continue

            # Pattern 2: "import X" or "import X, Y, Z"
            m = re.match(r'^import\s+([\w., ]+)', line)
            if m:
                modules = [mod.strip() for mod in m.group(1).split(",")]
                for module in modules:
                    module = module.strip()
                    if not module or not re.match(r'^[\w.]+$', module):
                        continue
                    target = _resolve_import("", module, fpath)
                    if target:
                        key = (fpath, target)
                        if key not in seen:
                            seen.add(key)
                            triples.append(Triple(
                                fpath, "imports", target,
                                line,  # original import statement
                            ))

    # Also store external imports (condition="external") for LLM context
    for fpath, code in file_contents.items():
        for line in code.split("\n"):
            line = line.strip()
            m = re.match(r'^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))', line)
            if m:
                module = (m.group(1) or m.group(2) or "").split(".")[0]
                if module and len(module) > 2:
                    # Check if this is external (not resolved to our files)
                    is_internal = any(module in f.replace("/", ".") for f in clean_files)
                    if not is_internal:
                        triples.append(Triple(fpath, "imports_external", module, line))

    # Cross-file call detection: if file A calls a name defined in file B
    defined_names: dict[str, set[str]] = {}
    for fpath, code in file_contents.items():
        names = set()
        for line in code.split("\n"):
            m = re.match(r'^\s*(?:def|class)\s+(\w+)', line.strip())
            if m:
                names.add(m.group(1))
        defined_names[fpath] = names

    for fpath, code in file_contents.items():
        for other_fpath, other_names in defined_names.items():
            if other_fpath == fpath:
                continue
            other_base = other_fpath.split("/")[-1].replace(".py", "")
            for name in other_names:
                # Check for module.name pattern (e.g. views.handle_request)
                if f"{other_base}.{name}" in code:
                    edge = (fpath, other_fpath)
                    if edge not in seen:
                        seen.add(edge)
                        triples.append(Triple(
                            fpath, "calls", other_fpath,
                            f"{fpath.split('/')[-1]}→{other_fpath.split('/')[-1]} [callsite]",
                        ))
                    break

    return triples


def process_question(qi, item, v2_reader, v3_reader, mgr):
    """Process one DependEval question with both V2 and V3."""
    files_raw, file_contents, gt_raw = parse_dependeval_content(item)
    files = [f.strip("'\"") for f in files_raw]
    gt = [f.strip("'\"") for f in gt_raw]
    gt_basenames = [f.split("/")[-1] for f in gt]

    # Extract and encode
    triples = extract_imports_from_content(files, file_contents)

    with tempfile.TemporaryDirectory() as td:
        local_mgr = ShardManager(td, shard_size=64)
        if triples:
            for i in range(0, len(triples), 5):
                local_mgr.encode("", triples=triples[i:i+5])
            local_mgr.save()
            local_mgr._rebuild_entity_index()

        file_list = ", ".join([f.split("/")[-1] for f in files])

        # V2 retrieval
        t0 = time.perf_counter()
        v2_local = V2ReadPipeline(local_mgr)
        p2 = v2_local.query(f"dependency order of {file_list}", budget_tokens=1000)
        v2_ctx = p2.to_text("compact")
        v2_route_ms = (time.perf_counter() - t0) * 1000
        v2_tok = p2.estimate_tokens()

        # V3 retrieval
        t1 = time.perf_counter()
        builder = SummaryBuilder(local_mgr)
        store = builder.build_all()
        v3_local = V3ReadPipeline(local_mgr, store)
        b3 = v3_local.query(f"dependency order of {file_list}", budget_tokens=1000)
        v3_ctx = b3.answer_context(800)
        v3_route_ms = (time.perf_counter() - t1) * 1000
        v3_tok = b3.estimate_tokens()

        # Also try topo_sort directly
        try:
            topo = ra.topo_sort(local_mgr, relation="imports")
        except Exception:
            topo = []

        # Build path→basename mapping for the LLM
        path_map = "\n".join(f"  {f} = {f.split('/')[-1]}" for f in files)

        # LLM answers in parallel
        def answer_method(ctx, method_name):
            prompt = (
                f"Given these dependency facts (A imports B means A depends on B):\n{ctx[:2000]}\n\n"
                f"Files (full path = basename):\n{path_map}\n\n"
                f"Order ALL these files by dependency (base files first, files that depend on others last).\n"
                f"Use the FULL PATHS in your answer.\n"
                f"Return ONLY a JSON array of the full file paths."
            )
            raw = ask_cli(prompt)
            try:
                m = re.search(r'\[.*\]', raw, re.DOTALL)
                if m:
                    pred_raw = json.loads(m.group(0))
                    return [f.strip("'\" ") for f in pred_raw]
            except: pass
            return []

        with ThreadPoolExecutor(max_workers=2) as ex:
            fv2 = ex.submit(answer_method, v2_ctx, "v2")
            fv3 = ex.submit(answer_method, v3_ctx, "v3")
            v2_pred_full = fv2.result()
            v3_pred_full = fv3.result()

        # Compare using full paths (gt is already full paths)
        # Also try basename comparison as fallback
        v2_pred = [f.split("/")[-1] for f in v2_pred_full]
        v3_pred = [f.split("/")[-1] for f in v3_pred_full]
        v2_exact = v2_pred_full == gt or v2_pred == gt_basenames
        v3_exact = v3_pred_full == gt or v3_pred == gt_basenames
        topo_exact = False
        if topo:
            topo_exact = topo == gt
            if not topo_exact:
                topo_basenames = [t.split("/")[-1] for t in topo]
                topo_exact = topo_basenames == gt_basenames

        s2 = "✓" if v2_exact else "✗"
        s3 = "✓" if v3_exact else "✗"
        st = "✓" if topo_exact else "✗"
        print(f"  [{qi+1:2d}] V2={s2} V3={s3} topo={st} "
              f"V2tok={v2_tok:>4d} V3tok={v3_tok:>4d} "
              f"triples={len(triples):>3d} files={len(files)} "
              f"| exp={gt_basenames}")

        return {
            "qi": qi, "files": [f.split("/")[-1] for f in files],
            "expected": gt_basenames, "n_triples": len(triples),
            "v2": {"pred": v2_pred, "exact": v2_exact, "tokens": v2_tok + len(v2_ctx)//4,
                   "route_ms": round(v2_route_ms, 1)},
            "v3": {"pred": v3_pred, "exact": v3_exact, "tokens": v3_tok + len(v3_ctx)//4,
                   "route_ms": round(v3_route_ms, 1)},
            "topo_sort": {"pred": topo, "exact": topo_exact},
        }


def main():
    with open("/tmp/DependEval/data/python/task2_python_final.json") as f:
        data = json.load(f)

    # All Python Task 2 questions with 3-5 files
    picks = [d for d in data if 3 <= len(d["files"]) <= 5]

    print("=" * 70)
    print(f"EXP 13: V2 vs V3 on DependEval Task 2 ({len(picks)} questions)")
    print("=" * 70)

    results = []

    # Batch of 10
    for bs in range(0, len(picks), 10):
        batch = picks[bs:bs+10]
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {
                ex.submit(process_question, bs+i, item, None, None, None): i
                for i, item in enumerate(batch)
            }
            for f in as_completed(futures):
                try: results.append(f.result())
                except Exception as e: print(f"  ERROR: {e}")
        print(f"  --- batch done: {min(bs+10, len(picks))}/{len(picks)} ---")

    results.sort(key=lambda r: r["qi"])

    # Summary
    n = len(results)
    print(f"\n{'=' * 70}")
    print(f"DEPENDEVAL RESULTS ({n} questions)")
    print(f"{'=' * 70}")

    for method, label in [("v2", "V2"), ("v3", "V3"), ("topo_sort", "TOPO")]:
        exact = sum(r[method]["exact"] for r in results)
        avg_tok = sum(r[method].get("tokens", 0) for r in results) / n if method != "topo_sort" else 0
        total_tok = sum(r[method].get("tokens", 0) for r in results) if method != "topo_sort" else 0
        print(f"  {label:6s}: exact={exact}/{n} ({exact/n:.0%})  avg_tok={avg_tok:.0f}  total_tok={total_tok}")

    # Token breakdown
    print(f"\n--- Token Usage ---")
    v2_total = sum(r["v2"]["tokens"] for r in results)
    v3_total = sum(r["v3"]["tokens"] for r in results)
    print(f"  V2 total: {v2_total:,} tokens ({v2_total/n:.0f} avg)")
    print(f"  V3 total: {v3_total:,} tokens ({v3_total/n:.0f} avg)")
    saving = (1 - v3_total / max(1, v2_total)) * 100
    print(f"  V3 vs V2: {'saves' if saving > 0 else 'costs'} {abs(saving):.0f}%")

    # Failure analysis
    print(f"\n--- V3 Failure Analysis ---")
    v3_failures = [r for r in results if not r["v3"]["exact"]]
    n_fail = len(v3_failures)
    print(f"  Total failures: {n_fail}/{n}")

    # Categorize failures
    cat_dup_names = 0
    cat_insufficient_edges = 0
    cat_wrong_order = 0
    cat_wrong_files = 0
    cat_empty_pred = 0

    for r in v3_failures:
        pred = r["v3"]["pred"]
        exp = r["expected"]
        n_files = len(r["files"])
        n_edges = r["n_triples"]

        if not pred:
            cat_empty_pred += 1
            r["v3"]["failure_reason"] = "empty_prediction"
        elif len(set(r["files"])) < len(r["files"]):
            cat_dup_names += 1
            r["v3"]["failure_reason"] = "duplicate_filenames"
        elif n_edges < n_files - 1:
            cat_insufficient_edges += 1
            r["v3"]["failure_reason"] = "insufficient_edges"
        elif set(pred) != set([f.split("/")[-1] for f in exp]):
            cat_wrong_files += 1
            r["v3"]["failure_reason"] = "wrong_files"
        else:
            cat_wrong_order += 1
            r["v3"]["failure_reason"] = "wrong_order"

    print(f"  Empty prediction:     {cat_empty_pred}")
    print(f"  Duplicate filenames:  {cat_dup_names}")
    print(f"  Insufficient edges:   {cat_insufficient_edges}")
    print(f"  Wrong files:          {cat_wrong_files}")
    print(f"  Wrong order:          {cat_wrong_order}")

    # By file count
    print(f"\n--- Accuracy by File Count ---")
    for nf in sorted(set(len(r["files"]) for r in results)):
        subset = [r for r in results if len(r["files"]) == nf]
        ns = len(subset)
        v2_e = sum(r["v2"]["exact"] for r in subset)
        v3_e = sum(r["v3"]["exact"] for r in subset)
        print(f"  {nf} files: V2={v2_e}/{ns} ({v2_e/ns:.0%})  V3={v3_e}/{ns} ({v3_e/ns:.0%})")

    Path("results").mkdir(exist_ok=True)
    with open("results/exp13_dependeval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/exp13_dependeval.json")


if __name__ == "__main__":
    main()
