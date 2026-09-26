"""Deterministic code structure extraction — AST + regex, no LLM.

Extracts from Python files:
  - File-level: imports, module docstring
  - Class-level: class name, base classes, methods
  - Function-level: function name, parameters, decorators, calls
  - Config-level: module-level assignments (constants, settings)
  - Dependency-level: import chains, relative imports

Also supports lightweight extraction for non-Python files via regex:
  - JS/TS: import/export, function/class declarations
  - JSON/YAML: top-level keys
  - Markdown: headers

Returns list[Triple] ready for PixelMem encoding.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Optional

from pixelmem.triple_extractor import Triple


# ── Python AST Extraction ───────────────────────────────────────

def extract_file(
    file_path: str,
    repo_root: str = "",
    include_calls: bool = True,
    include_assignments: bool = True,
) -> list[Triple]:
    """Extract KG triples from a single file."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".py":
        return _extract_python(file_path, repo_root, include_calls, include_assignments)
    elif suffix in (".js", ".ts", ".jsx", ".tsx"):
        return _extract_javascript(file_path, repo_root)
    elif suffix in (".json", ".yaml", ".yml", ".toml"):
        return _extract_config(file_path, repo_root)
    elif suffix == ".md":
        return _extract_markdown(file_path, repo_root)
    else:
        return _extract_generic(file_path, repo_root)


def _rel_path(file_path: str, repo_root: str) -> str:
    """Get clean relative path as entity name."""
    if repo_root:
        try:
            return str(Path(file_path).relative_to(repo_root))
        except ValueError:
            pass
    return str(Path(file_path).name)


def _extract_python(
    file_path: str,
    repo_root: str,
    include_calls: bool = True,
    include_assignments: bool = True,
) -> list[Triple]:
    """Full AST-based extraction for Python files."""
    triples = []
    rel = _rel_path(file_path, repo_root)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)
    except (SyntaxError, UnicodeDecodeError):
        return [Triple(rel, "type", "python_file", "parse_error")]

    lines = source.split("\n")
    n_lines = len(lines)
    triples.append(Triple(rel, "type", "python_file", f"{n_lines} lines"))

    # Module docstring
    docstring = ast.get_docstring(tree)
    if docstring:
        short_doc = docstring.split("\n")[0][:100]
        triples.append(Triple(rel, "docstring", short_doc, ""))

    for node in ast.walk(tree):
        # Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                triples.append(Triple(rel, "imports", name, f"line {node.lineno}"))

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                full = f"{module}.{alias.name}" if module else alias.name
                triples.append(Triple(rel, "imports", full, f"line {node.lineno}"))

        # Classes
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            line_range = f"lines {node.lineno}-{node.end_lineno or '?'}"
            triples.append(Triple(rel, "contains_class", class_name, line_range))

            # Base classes
            for base in node.bases:
                base_name = _get_name(base)
                if base_name:
                    triples.append(Triple(class_name, "extends", base_name, ""))

            # Class docstring
            class_doc = ast.get_docstring(node)
            if class_doc:
                short = class_doc.split("\n")[0][:80]
                triples.append(Triple(class_name, "docstring", short, ""))

            # Methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = f"{class_name}.{item.name}"
                    m_range = f"lines {item.lineno}-{item.end_lineno or '?'}"
                    triples.append(Triple(rel, "contains_method", method_name, m_range))

                    # Decorators
                    for dec in item.decorator_list:
                        dec_name = _get_name(dec)
                        if dec_name:
                            triples.append(Triple(method_name, "decorated_by", dec_name, ""))

        # Top-level functions
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Only top-level (not inside a class)
            if _is_top_level(node, tree):
                func_name = node.name
                line_range = f"lines {node.lineno}-{node.end_lineno or '?'}"
                triples.append(Triple(rel, "contains_function", func_name, line_range))

                # Parameters
                params = [a.arg for a in node.args.args if a.arg != "self"]
                if params:
                    triples.append(Triple(func_name, "parameters", ", ".join(params[:10]), ""))

                # Decorators
                for dec in node.decorator_list:
                    dec_name = _get_name(dec)
                    if dec_name:
                        triples.append(Triple(func_name, "decorated_by", dec_name, ""))

                # Function docstring
                func_doc = ast.get_docstring(node)
                if func_doc:
                    short = func_doc.split("\n")[0][:80]
                    triples.append(Triple(func_name, "docstring", short, ""))

                # Function calls (what this function calls)
                if include_calls:
                    calls = _extract_calls(node)
                    for call_name in calls[:20]:
                        triples.append(Triple(func_name, "calls", call_name, ""))

        # Module-level assignments (constants, config)
        if include_assignments and isinstance(node, ast.Assign):
            if _is_top_level(node, tree):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        value = _get_assign_value(node.value)
                        if value:
                            triples.append(Triple(
                                rel, "defines_constant", target.id,
                                f"= {value[:60]}"
                            ))

    return triples


def _is_top_level(node, tree) -> bool:
    """Check if a node is at module level (not nested in class/function)."""
    for top_node in tree.body:
        if top_node is node:
            return True
    return False


def _get_name(node) -> Optional[str]:
    """Extract name from various AST node types."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        value = _get_name(node.value)
        if value:
            return f"{value}.{node.attr}"
        return node.attr
    elif isinstance(node, ast.Call):
        return _get_name(node.func)
    return None


def _get_assign_value(node) -> Optional[str]:
    """Extract simple assignment value as string."""
    if isinstance(node, ast.Constant):
        return repr(node.value)[:60]
    elif isinstance(node, ast.List):
        return f"[...] ({len(node.elts)} items)"
    elif isinstance(node, ast.Dict):
        return f"{{...}} ({len(node.keys)} keys)"
    elif isinstance(node, ast.Call):
        name = _get_name(node.func)
        return f"{name}(...)" if name else None
    return None


def _extract_calls(func_node) -> list[str]:
    """Extract function/method calls from a function body."""
    calls = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            name = _get_name(node.func)
            if name and not name.startswith("_") and len(name) > 1:
                calls.add(name)
    return sorted(calls)


# ── JavaScript/TypeScript Extraction ────────────────────────────

def _extract_javascript(file_path: str, repo_root: str) -> list[Triple]:
    """Regex-based extraction for JS/TS files."""
    triples = []
    rel = _rel_path(file_path, repo_root)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except Exception:
        return []

    lines = source.split("\n")
    triples.append(Triple(rel, "type", "javascript_file", f"{len(lines)} lines"))

    # Imports
    for m in re.finditer(r"import\s+(?:{[^}]+}|[\w*]+)\s+from\s+['\"]([^'\"]+)['\"]", source):
        triples.append(Triple(rel, "imports", m.group(1), ""))
    for m in re.finditer(r"require\(['\"]([^'\"]+)['\"]\)", source):
        triples.append(Triple(rel, "imports", m.group(1), ""))

    # Exports
    for m in re.finditer(r"export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)", source):
        triples.append(Triple(rel, "exports", m.group(1), ""))

    # Functions/classes
    for i, line in enumerate(lines, 1):
        m = re.match(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", line)
        if m:
            triples.append(Triple(rel, "contains_function", m.group(1), f"line {i}"))
        m = re.match(r"(?:export\s+)?class\s+(\w+)", line)
        if m:
            triples.append(Triple(rel, "contains_class", m.group(1), f"line {i}"))

    return triples


# ── Config File Extraction ──────────────────────────────────────

def _extract_config(file_path: str, repo_root: str) -> list[Triple]:
    """Extract top-level keys from config files."""
    triples = []
    rel = _rel_path(file_path, repo_root)
    suffix = Path(file_path).suffix.lower()

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return []

    triples.append(Triple(rel, "type", "config_file", suffix))

    if suffix == ".json":
        import json
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                for key in list(data.keys())[:30]:
                    val = data[key]
                    val_str = repr(val)[:50] if not isinstance(val, (dict, list)) else f"({type(val).__name__})"
                    triples.append(Triple(rel, "configures", key, val_str))
        except json.JSONDecodeError:
            pass

    elif suffix in (".yaml", ".yml"):
        # Simple regex for top-level keys
        for m in re.finditer(r"^(\w[\w_-]*)\s*:", content, re.MULTILINE):
            triples.append(Triple(rel, "configures", m.group(1), ""))

    elif suffix == ".toml":
        for m in re.finditer(r"^\[([^\]]+)\]", content, re.MULTILINE):
            triples.append(Triple(rel, "configures_section", m.group(1), ""))

    return triples


# ── Markdown Extraction ─────────────────────────────────────────

def _extract_markdown(file_path: str, repo_root: str) -> list[Triple]:
    """Extract headers and structure from markdown."""
    triples = []
    rel = _rel_path(file_path, repo_root)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return []

    triples.append(Triple(rel, "type", "markdown_file", ""))

    for m in re.finditer(r"^(#{1,4})\s+(.+)$", content, re.MULTILINE):
        level = len(m.group(1))
        heading = m.group(2).strip()
        triples.append(Triple(rel, f"h{level}", heading, ""))

    return triples


# ── Generic File Extraction ─────────────────────────────────────

def _extract_generic(file_path: str, repo_root: str) -> list[Triple]:
    """Minimal extraction for unknown file types."""
    rel = _rel_path(file_path, repo_root)
    suffix = Path(file_path).suffix.lower()
    try:
        size = os.path.getsize(file_path)
    except OSError:
        size = 0
    return [Triple(rel, "type", suffix.lstrip(".") + "_file" if suffix else "file", f"{size} bytes")]


# ── Repository-Level Extraction ─────────────────────────────────

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".tox", ".mypy_cache",
    ".pytest_cache", "venv", ".venv", "env", ".env", "dist", "build",
    ".eggs", "*.egg-info", ".idea", ".vscode",
}

_SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll",
    ".exe", ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".pdf", ".doc", ".docx",
}


def extract_repo(
    repo_path: str,
    max_files: int = 1000,
    include_calls: bool = True,
    include_assignments: bool = True,
    file_extensions: Optional[set[str]] = None,
) -> list[Triple]:
    """Extract KG triples from an entire repository.

    Walks the directory tree, extracts from each file, and adds
    directory structure relationships.

    Args:
        repo_path: Path to repository root.
        max_files: Maximum files to process.
        include_calls: Extract function call relationships.
        include_assignments: Extract module-level constants.
        file_extensions: If set, only process these extensions.

    Returns:
        List of triples describing the repository structure.
    """
    triples = []
    repo = Path(repo_path)
    repo_name = repo.name

    triples.append(Triple(repo_name, "type", "repository", str(repo)))

    file_count = 0
    for root, dirs, files in os.walk(repo):
        # Skip hidden/build directories
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]

        rel_dir = str(Path(root).relative_to(repo))
        if rel_dir != ".":
            parent = str(Path(rel_dir).parent)
            if parent == ".":
                parent = repo_name
            triples.append(Triple(parent, "contains_dir", rel_dir, ""))

        for fname in sorted(files):
            if file_count >= max_files:
                break

            fpath = os.path.join(root, fname)
            suffix = Path(fname).suffix.lower()

            # Skip binary/unwanted files
            if suffix in _SKIP_EXTENSIONS:
                continue
            if file_extensions and suffix not in file_extensions:
                continue

            rel_file = str(Path(fpath).relative_to(repo))

            # Directory containment
            if rel_dir == ".":
                triples.append(Triple(repo_name, "contains_file", rel_file, ""))
            else:
                triples.append(Triple(rel_dir, "contains_file", rel_file, ""))

            # File-level extraction
            file_triples = extract_file(
                fpath, str(repo), include_calls, include_assignments
            )
            triples.extend(file_triples)
            file_count += 1

    triples.append(Triple(repo_name, "total_files", str(file_count), ""))
    return triples
