"""Language-Adaptive Import Extraction — handles ANY file type.

Pipeline:
  1. Detect language from file extension (known or unknown)
  2. If known language → use built-in regex extractor
  3. If unknown language → ask LLM to generate import regex patterns,
     test them on sample code, cache if they work
  4. If all else fails → LLM extracts imports directly from code

No hardcoded language limit. New languages are learned on the fly.
"""

from __future__ import annotations

import re
from typing import Optional

from pixelmem.triple_extractor import Triple


# ═══════════════════════════════════════════════
# Language detection
# ═══════════════════════════════════════════════

def detect_language(files: list[str]) -> str:
    """Detect dominant language from file extensions."""
    ext_counts: dict[str, int] = {}
    for f in files:
        ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    ext_to_lang = {
        "py": "python", "java": "java",
        "js": "javascript", "jsx": "javascript", "mjs": "javascript",
        "ts": "typescript", "tsx": "typescript",
        "c": "c", "h": "c",
        "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "hh": "cpp",
        "cs": "csharp",
        "php": "php",
    }

    lang_counts: dict[str, int] = {}
    for ext, cnt in ext_counts.items():
        lang = ext_to_lang.get(ext, "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + cnt

    if not lang_counts:
        return "unknown"
    return max(lang_counts, key=lang_counts.get)


# ═══════════════════════════════════════════════
# Per-language import extractors (regex-based)
# ═══════════════════════════════════════════════

def _extract_python_imports(code: str) -> list[str]:
    """Extract import targets from Python code."""
    targets = []
    for line in code.split("\n"):
        line = line.strip()
        # from X import Y
        m = re.match(r'^from\s+(\.{0,3}[\w.]*)\s+import\s+([\w, *]+)', line)
        if m:
            from_mod = m.group(1).lstrip(".")
            names = [n.strip() for n in m.group(2).split(",")]
            for name in names:
                name = name.strip()
                if name and name != "*":
                    targets.append(f"{from_mod}.{name}" if from_mod else name)
                elif from_mod:
                    targets.append(from_mod)
            continue
        # import X, Y, Z
        m = re.match(r'^import\s+([\w., ]+)', line)
        if m:
            for mod in m.group(1).split(","):
                mod = mod.strip()
                if mod and re.match(r'^[\w.]+$', mod):
                    targets.append(mod)
    return targets


def _extract_java_imports(code: str) -> list[str]:
    """Extract import targets from Java code."""
    targets = []
    for line in code.split("\n"):
        # import com.example.package.ClassName;
        m = re.match(r'^\s*import\s+(?:static\s+)?([\w.]+)\s*;', line)
        if m:
            targets.append(m.group(1))
    return targets


def _extract_js_ts_imports(code: str) -> list[str]:
    """Extract import targets from JavaScript/TypeScript code."""
    targets = []
    # import X from 'path'
    for m in re.finditer(r"""(?:import|require)\s*\(?['"]([^'"]+)['"]\)?""", code):
        targets.append(m.group(1))
    # import { X } from 'path'
    for m in re.finditer(r"""from\s+['"]([^'"]+)['"]""", code):
        if m.group(1) not in targets:
            targets.append(m.group(1))
    return targets


def _extract_c_cpp_includes(code: str) -> list[str]:
    """Extract #include targets from C/C++ code."""
    targets = []
    for m in re.finditer(r'#include\s*[<"]([^>"]+)[>"]', code):
        targets.append(m.group(1))
    return targets


def _extract_csharp_usings(code: str) -> list[str]:
    """Extract using targets from C# code."""
    targets = []
    for m in re.finditer(r'^\s*using\s+([\w.]+)\s*;', code, re.MULTILINE):
        targets.append(m.group(1))
    return targets


def _extract_php_imports(code: str) -> list[str]:
    """Extract use/require/include targets from PHP code."""
    targets = []
    # use Namespace\Class;
    for m in re.finditer(r'^\s*use\s+([\w\\]+)', code, re.MULTILINE):
        targets.append(m.group(1).replace("\\", "."))
    # require/include 'file.php'
    for m in re.finditer(r"""(?:require|include)(?:_once)?\s*\(?['"]([^'"]+)['"]""", code):
        targets.append(m.group(1))
    return targets


_LANG_EXTRACTORS = {
    "python": _extract_python_imports,
    "java": _extract_java_imports,
    "javascript": _extract_js_ts_imports,
    "typescript": _extract_js_ts_imports,
    "c": _extract_c_cpp_includes,
    "cpp": _extract_c_cpp_includes,
    "csharp": _extract_csharp_usings,
    "php": _extract_php_imports,
}

# Cache for dynamically learned extraction rules
_LEARNED_PATTERNS: dict[str, list[str]] = {}  # language → [regex patterns]


def _extract_universal(code: str, patterns: list[str]) -> list[str]:
    """Extract import targets using a list of regex patterns."""
    targets = []
    for pattern in patterns:
        try:
            for m in re.finditer(pattern, code, re.MULTILINE):
                target = m.group(1) if m.lastindex else m.group(0)
                if target and len(target) > 1:
                    targets.append(target.strip())
        except re.error:
            continue
    return targets


def learn_extraction_rules(
    language: str,
    sample_code: str,
    ask_fn,
) -> list[str]:
    """Ask LLM to generate import extraction regex patterns for an unknown language.

    Returns list of regex patterns with capture group 1 = import target.
    Tests patterns on sample code before caching.
    """
    if language in _LEARNED_PATTERNS:
        return _LEARNED_PATTERNS[language]

    prompt = (
        f"I need regex patterns to extract import/include/require/use statements "
        f"from {language} source code.\n\n"
        f"Sample code:\n```\n{sample_code[:1500]}\n```\n\n"
        f"Return a JSON array of Python regex patterns. "
        f"Each pattern must have exactly ONE capture group that captures "
        f"the imported module/file name.\n\n"
        f"Example for Python: [\"^import\\\\s+([\\\\w.]+)\", \"^from\\\\s+([\\\\w.]+)\\\\s+import\"]\n"
        f"Example for C: [\"#include\\\\s*[<\\\"]([^>\\\"]+)[>\\\"]\"]\n\n"
        f"Return ONLY the JSON array."
    )

    answer = ask_fn(prompt)
    if isinstance(answer, tuple):
        answer = answer[0]

    patterns = []
    try:
        m = re.search(r'\[.*\]', answer, re.DOTALL)
        if m:
            import json
            raw_patterns = json.loads(m.group(0))
            for p in raw_patterns:
                if isinstance(p, str):
                    # Validate: must compile and have 1 group
                    try:
                        compiled = re.compile(p, re.MULTILINE)
                        if compiled.groups >= 1:
                            # Test on sample code
                            matches = compiled.findall(sample_code)
                            if matches:
                                patterns.append(p)
                    except re.error:
                        continue
    except Exception:
        pass

    # Fallback: generic patterns that work for most languages
    if not patterns:
        patterns = [
            r'^\s*import\s+([\w.]+)',
            r'^\s*from\s+([\w.]+)\s+import',
            r'#include\s*[<"]([^>"]+)[>"]',
            r'^\s*use\s+([\w\\:.]+)',
            r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
        ]

    _LEARNED_PATTERNS[language] = patterns
    return patterns


def get_extractor(language: str, sample_code: str = "", ask_fn=None):
    """Get or build an import extractor for any language.

    Returns a function(code) → list[str] of import targets.
    """
    # Known language
    if language in _LANG_EXTRACTORS:
        return _LANG_EXTRACTORS[language]

    # Already learned
    if language in _LEARNED_PATTERNS:
        patterns = _LEARNED_PATTERNS[language]
        return lambda code: _extract_universal(code, patterns)

    # Learn from LLM
    if ask_fn and sample_code:
        patterns = learn_extraction_rules(language, sample_code, ask_fn)
        return lambda code: _extract_universal(code, patterns)

    # Ultimate fallback: broad patterns covering most languages
    all_patterns = [
        r'^\s*import\s+([\w.]+)',                        # Python, Java, Kotlin, Go
        r'^\s*import\s+["\']([^"\']+)["\']',             # Go: import "fmt"
        r'^\s*from\s+([\w.]+)\s+import',                 # Python
        r'#include\s*[<"]([^>"]+)[>"]',                  # C, C++, ObjC
        r'^\s*use\s+([\w\\:.]+)',                         # PHP, Rust, Perl
        r'''require\s*\(?\s*['"]([^'"]+)['"]\s*\)?''',   # JS, Ruby, Lua
        r'''require_relative\s+['"]([^'"]+)['"]''',      # Ruby
        r'''from\s+['"]([^'"]+)['"]''',                  # JS/TS ESM
        r'^\s*@import\s+["\']([^"\']+)["\']',            # CSS, SCSS
        r'^\s*load\s+["\']([^"\']+)["\']',               # Ruby, Tcl
        r'^\s*extern\s+crate\s+(\w+)',                   # Rust
        r'^\s*open\s+(\w+)',                             # OCaml
        r'^\s*using\s+([\w.]+)\s*;',                     # C#
    ]
    return lambda code: _extract_universal(code, all_patterns)


# ═══════════════════════════════════════════════
# Module → file resolution (language-aware)
# ═══════════════════════════════════════════════

def _build_module_map(files: list[str], language: str) -> dict[str, str]:
    """Build module/path → file lookup based on language conventions."""
    module_to_file: dict[str, str] = {}

    for fpath in files:
        basename = fpath.rsplit("/", 1)[-1]
        name_no_ext = basename.rsplit(".", 1)[0]

        # Register basename without extension
        module_to_file[name_no_ext] = fpath
        module_to_file[basename] = fpath

        if language == "python":
            full_module = fpath.replace("/", ".").replace(".py", "")
            parts = full_module.split(".")
            for i in range(len(parts)):
                module_to_file[".".join(parts[i:])] = fpath

        elif language == "java":
            # com/example/Foo.java → com.example.Foo
            full_module = fpath.replace("/", ".").replace(".java", "")
            parts = full_module.split(".")
            for i in range(len(parts)):
                module_to_file[".".join(parts[i:])] = fpath
            # Also register just the class name
            module_to_file[parts[-1]] = fpath

        elif language in ("javascript", "typescript"):
            # Relative paths: ./foo, ../bar, @scope/pkg
            # Register with and without extension
            module_to_file[fpath] = fpath
            module_to_file["./" + fpath] = fpath
            module_to_file[fpath.rsplit(".", 1)[0]] = fpath
            module_to_file["./" + fpath.rsplit(".", 1)[0]] = fpath

        elif language in ("c", "cpp"):
            # #include "foo.h" matches by filename
            module_to_file[basename] = fpath
            # Also partial path matching
            parts = fpath.split("/")
            for i in range(len(parts)):
                module_to_file["/".join(parts[i:])] = fpath

        elif language == "csharp":
            # using Namespace.Class → match by class name or namespace
            full_ns = fpath.replace("/", ".").replace(".cs", "")
            parts = full_ns.split(".")
            for i in range(len(parts)):
                module_to_file[".".join(parts[i:])] = fpath

        elif language == "php":
            # use Namespace\Class → match
            module_to_file[fpath] = fpath
            module_to_file[name_no_ext] = fpath

    return module_to_file


# ═══════════════════════════════════════════════
# Main extraction pipeline
# ═══════════════════════════════════════════════

def extract_multilang(
    files: list[str],
    file_contents: dict[str, str],
    language: Optional[str] = None,
    ask_fn=None,
) -> tuple[list[Triple], dict]:
    """Language-adaptive import extraction.

    1. Detect language
    2. Extract imports with language-specific parser
    3. Resolve to file edges
    4. Add definitions and external imports for context
    5. If insufficient edges and ask_fn provided, LLM fallback

    Returns (triples, stats).
    """
    if language is None:
        language = detect_language(files)

    # Get extractor — known language uses built-in, unknown learns dynamically
    sample_code = next(iter(file_contents.values()), "") if file_contents else ""
    extractor = get_extractor(language, sample_code, ask_fn)
    module_map = _build_module_map(files, language)

    triples: list[Triple] = []
    seen_edges: set[tuple[str, str]] = set()
    total_import_lines = 0
    resolved_count = 0

    for fpath, code in file_contents.items():
        # Extract import targets
        targets = extractor(code)
        total_import_lines += len(targets)

        for target in targets:
            # Try to resolve to one of our files
            resolved_file = None
            # Try exact match
            resolved_file = module_map.get(target)
            if not resolved_file:
                # Try each component
                for part in target.replace("\\", ".").split("."):
                    if part and len(part) > 1:
                        r = module_map.get(part)
                        if r and r != fpath:
                            resolved_file = r
                            break

            if resolved_file and resolved_file != fpath:
                edge = (fpath, resolved_file)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    resolved_count += 1
                    triples.append(Triple(
                        fpath, "imports", resolved_file,
                        f"{fpath.rsplit('/', 1)[-1]}→{resolved_file.rsplit('/', 1)[-1]}",
                    ))
            else:
                # Store as external import for context
                ext_name = target.split(".")[0] if "." in target else target
                if ext_name and len(ext_name) > 1:
                    triples.append(Triple(fpath, "imports_external", ext_name, "external"))

        # Store definitions
        for line in code.split("\n"):
            stripped = line.strip()
            # Language-specific definition patterns
            if language == "python":
                m = re.match(r'^(?:def|class)\s+(\w+)', stripped)
            elif language == "java":
                m = re.match(r'(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface|enum)\s+(\w+)', stripped)
                if not m:
                    m = re.match(r'(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(', stripped)
            elif language in ("javascript", "typescript"):
                m = re.match(r'(?:export\s+)?(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)', stripped)
            elif language in ("c", "cpp"):
                m = re.match(r'(?:[\w:*&]+\s+)+(\w+)\s*\(', stripped)
            elif language == "csharp":
                m = re.match(r'(?:public|private|internal)?\s*(?:static\s+)?(?:class|struct|interface)\s+(\w+)', stripped)
            elif language == "php":
                m = re.match(r'(?:public|private|protected)?\s*(?:static\s+)?function\s+(\w+)', stripped)
                if not m:
                    m = re.match(r'class\s+(\w+)', stripped)
            else:
                m = None
            if m:
                triples.append(Triple(fpath, "defines", m.group(1), "definition"))

    # Cross-file call detection
    defined_names: dict[str, set[str]] = {}
    for fpath, code in file_contents.items():
        names = set()
        for t in triples:
            if t.subject == fpath and t.relation == "defines":
                names.add(t.object)
        defined_names[fpath] = names

    for fpath, code in file_contents.items():
        for other, names in defined_names.items():
            if other == fpath:
                continue
            other_base = other.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            for name in names:
                if f"{other_base}.{name}" in code:
                    edge = (fpath, other)
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        triples.append(Triple(
                            fpath, "calls", other,
                            f"{fpath.rsplit('/', 1)[-1]}→{other.rsplit('/', 1)[-1]} [callsite]",
                        ))
                    break

    stats = {
        "language": language,
        "total_import_lines": total_import_lines,
        "resolved_edges": resolved_count,
        "total_triples": len(triples),
        "method": "deterministic",
    }

    # LLM fallback if insufficient edges
    n_internal = sum(1 for t in triples if t.relation in ("imports", "calls"))
    needed = len(files) - 1

    if ask_fn and n_internal < needed:
        llm_triples = _llm_fallback(files, file_contents, language, triples, ask_fn)
        triples.extend(llm_triples)
        stats["llm_edges"] = len(llm_triples)
        stats["method"] = "hybrid"

    stats["total_triples"] = len(triples)
    return triples, stats


def _llm_fallback(
    files: list[str],
    file_contents: dict[str, str],
    language: str,
    existing_triples: list[Triple],
    ask_fn,
) -> list[Triple]:
    """LLM resolves remaining edges from import lines + definitions."""
    import json

    # Build compact summary per file
    lines = []
    for fpath in files:
        code = file_contents.get(fpath, "")
        fname = fpath.rsplit("/", 1)[-1]
        defs = [t.object for t in existing_triples if t.subject == fpath and t.relation == "defines"]
        imports = [l.strip() for l in code.split("\n") if _is_import_line(l.strip(), language)][:10]

        lines.append(f"=== {fname} ({fpath}) ===")
        if imports:
            for imp in imports:
                lines.append(f"  {imp}")
        if defs:
            lines.append(f"  defines: {', '.join(defs[:10])}")
        lines.append("")

    compact = "\n".join(lines)
    file_list = "\n".join(f"  {f}" for f in files)

    prompt = (
        f"These {language} files have dependencies. "
        f"Determine which file depends on which.\n\n"
        f"{compact[:3000]}\n"
        f"Files:\n{file_list}\n\n"
        f'Return a JSON array of [source, target] pairs: '
        f'[["file_that_depends", "file_it_depends_on"]].\n'
        f"Use FULL file paths."
    )

    answer = ask_fn(prompt)
    if isinstance(answer, tuple):
        answer = answer[0]

    seen = {(t.subject, t.object) for t in existing_triples if t.relation in ("imports", "calls")}
    new_triples = []
    try:
        m = re.search(r'\[.*\]', answer, re.DOTALL)
        if m:
            edges = json.loads(m.group(0))
            for edge in edges:
                if isinstance(edge, list) and len(edge) == 2:
                    src, tgt = edge[0].strip("'\" "), edge[1].strip("'\" ")
                    # Match to our files
                    src_match = _find_file(src, files)
                    tgt_match = _find_file(tgt, files)
                    if src_match and tgt_match and src_match != tgt_match:
                        if (src_match, tgt_match) not in seen:
                            seen.add((src_match, tgt_match))
                            new_triples.append(Triple(
                                src_match, "imports", tgt_match,
                                f"{src_match.rsplit('/', 1)[-1]}→{tgt_match.rsplit('/', 1)[-1]} [llm]",
                            ))
    except Exception:
        pass
    return new_triples


def _is_import_line(line: str, language: str) -> bool:
    if language == "python":
        return bool(re.match(r'^(?:from|import)\s', line))
    elif language == "java":
        return bool(re.match(r'^import\s', line))
    elif language in ("javascript", "typescript"):
        return bool(re.match(r'^(?:import|const.*require|let.*require|var.*require)', line))
    elif language in ("c", "cpp"):
        return bool(re.match(r'^#include\s', line))
    elif language == "csharp":
        return bool(re.match(r'^using\s', line))
    elif language == "php":
        return bool(re.match(r'^(?:use|require|include)', line))
    return False


def _find_file(name: str, files: list[str]) -> Optional[str]:
    name = name.strip("'\" ")
    if name in files:
        return name
    bn = name.rsplit("/", 1)[-1]
    matches = [f for f in files if f.rsplit("/", 1)[-1] == bn]
    if len(matches) == 1:
        return matches[0]
    for f in files:
        if f.endswith(name):
            return f
    return None
