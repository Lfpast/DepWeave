"""localMem extraction and task memory over globalMem's MCP evidence graph."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from benchmarks.dependeval.python_deps import PythonDependencyEngine, PythonDependencyExtractor
from benchmarks.repoqa import RepoQAFunctionExtractor, _score_function_against_desc, _tokens
from core.cache import PixelMemCache
from core.types import TaskSpec


class DepWeaveRunner:
    """One MCP session, many repository snapshots and task queries.

    ``llm`` must implement ``count_tokens(prompt)`` using its actual chat
    tokenizer and ``__call__(prompt) -> (text, input_tokens, output_tokens)``.
    This prevents the approximate global graph token estimate from silently
    overrunning the Qwen input limit.
    """

    def __init__(self, llm, *, max_input_tokens: int = 4096,
                 cache_dir: str | None = None, dependency_cache_dir: str | None = None,
                 reuse_local_functions: bool = True, server_path: Path | None = None):
        if max_input_tokens < 256:
            raise ValueError("max_input_tokens must be at least 256")
        if not callable(getattr(llm, "count_tokens", None)):
            raise TypeError("llm must implement count_tokens with the model's chat tokenizer")
        self.llm = llm
        self.max_input_tokens = max_input_tokens
        self.cache = PixelMemCache(cache_dir) if cache_dir else None
        self.dependency_cache = PixelMemCache(dependency_cache_dir) if dependency_cache_dir else None
        self.reuse_local_functions = reuse_local_functions
        self.server_path = server_path or ROOT / "globalMem/code_graph_mcp_server.py"
        self.client = None
        self.prepared: dict[tuple[str, str], dict] = {}
        self.sessions: dict[str, dict] = {}

    async def __aenter__(self):
        from fastmcp import Client
        self.client = Client(self.server_path)
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self.client.__aexit__(*exc)
        self.client = None

    async def _call(self, tool: str, args: dict) -> dict:
        if self.client is None:
            raise RuntimeError("use 'async with DepWeaveRunner(...)' to open globalMem MCP")
        result = await self.client.call_tool(tool, args)
        payload = (result.structured_content or {}).get("result")
        if payload is None:
            payload = result.content[0].text
        return json.loads(payload)

    async def prepare(self, repo_id: str, documents: dict[str, str]) -> dict:
        digest = _digest(documents)
        key = (repo_id, digest)
        if key in self.prepared:
            prepared = self.prepared[key]
            previous = self.sessions.get(repo_id)
            if previous and previous["snapshot_id"] != prepared["snapshot_id"]:
                del self.sessions[repo_id]
            if not self.reuse_local_functions:
                local_functions, local_cache_hit, local_extract_time = self._extract_local_functions(documents)
                return {**prepared, "local_functions": local_functions,
                        "local_cache_hit": local_cache_hit, "prepare_reused": True,
                        "global_index_time_s": 0.0, "local_extract_time_s": local_extract_time}
            return {**prepared, "prepare_reused": True,
                    "global_index_time_s": 0.0, "local_extract_time_s": 0.0}
        started = time.perf_counter()
        index = await self._call("code_index_documents", {
            "repo_id": repo_id, "documents_json": json.dumps(documents),
        })
        global_index_time = time.perf_counter() - started
        previous = self.sessions.get(repo_id)
        if previous and previous["snapshot_id"] != index["snapshot_id"]:
            del self.sessions[repo_id]
        local_functions, local_cache_hit, local_extract_time = self._extract_local_functions(documents)
        prepared = {"snapshot_id": index["snapshot_id"], "coverage": index["coverage"],
                    "global_stats": index["stats"], "local_functions": local_functions,
                    "local_cache_hit": local_cache_hit,
                    "prepare_reused": False, "global_index_time_s": global_index_time,
                    "local_extract_time_s": local_extract_time}
        self.prepared[key] = prepared
        return prepared

    def _extract_local_functions(self, documents: dict[str, str]) -> tuple[list, bool, float]:
        extractor = RepoQAFunctionExtractor()
        probe = self.cache.wrap(extractor) if self.cache else extractor
        started = time.perf_counter()
        functions = probe.extract(documents)
        return functions, bool(getattr(probe, "last_hit", False)), time.perf_counter() - started

    async def repoqa(self, repo_id: str, documents: dict[str, str], description: str) -> dict:
        prepared = await self.prepare(repo_id, documents)
        query_started = time.perf_counter()
        desc_tokens = set(_tokens(description))
        scored = []
        for p in prepared["local_functions"]:
            prov = p.provenance or {}
            score = _score_function_against_desc(p.object, prov.get("docstring", ""),
                                                 prov.get("snippet", ""), desc_tokens)
            scored.append((score, p))
        scored.sort(key=lambda item: (-item[0], item[1].subject,
                                      (item[1].provenance or {}).get("lineno", 0)))
        previous = self.sessions.get(repo_id)
        refer_back = bool(re.search(r"\b(previous|earlier|above|that function)\b|之前|刚才|上一轮", description, re.I))
        prior_seeds = []
        if previous and refer_back and previous["snapshot_id"] == prepared["snapshot_id"]:
            prior_seeds = sorted(previous.get("candidates", []),
                                 key=lambda c: c["id"] != previous.get("last_prediction"))[:4]
        seeds = [{"path": c["path"], "line": c["line"], "name": c["name"]}
                 for c in prior_seeds]
        seeds += [{"path": p.subject, "line": (p.provenance or {}).get("lineno"),
                   "name": p.object} for _, p in scored[:12]]
        packet = await self._call("code_evidence", {
            "snapshot_id": prepared["snapshot_id"], "query": description,
            "seeds_json": json.dumps(seeds), "max_candidates": 16,
            "max_edges": 80, "hops": 1,
        })
        _validate_packet(packet, prepared["snapshot_id"], documents)
        local = {(p.subject, (p.provenance or {}).get("lineno"), p.object): p
                 for p in prepared["local_functions"]}
        base = (
            "Find the Python function matching the description. Choose one candidate label "
            "or answer UNKNOWN if evidence is insufficient. Source locations identify "
            "definitions; unresolved calls are not proven dependencies.\n"
            f"Description: {description[:1600]}\n"
            f"Repository snapshot: {packet['snapshot_id']}\n"
            "Candidates and their source evidence:\n"
        )
        blocks: list[tuple[str, str]] = []
        for number, c in enumerate(packet["candidates"], 1):
            p = local.get((c["path"], c["line"], c["name"]))
            prov = p.provenance or {} if p else {}
            source = prov.get("snippet") or _source_excerpt(documents.get(c["path"], ""), c["line"])
            doc = (prov.get("docstring") or "").splitlines()[:2]
            touching = [e for e in packet["edges"] if c["id"] in (e["from_id"], e["to_id"])]
            rels = "\n".join("  " + _edge_text(e) for e in touching[:3])
            block = (f"C{number} ID {c['id']}\n  {c['kind']} {c['name']}{c['signature']} at "
                     f"{c['path']}:{c['line']}\n  doc: {' '.join(doc)[:220]}\n"
                     f"  source: {source[:420]}\n" + (rels + "\n" if rels else ""))
            blocks.append((c["id"], block))
        memory = self._memory_context(repo_id, description, packet["snapshot_id"])
        tail = ("\nEvidence gaps:\n" + "\n".join(_gap_text(g) for g in _prompt_gaps(packet["gaps"]))
                + ("\n" + memory if memory else "")
                + "\nReturn exactly: ANSWER: <candidate label, such as C1, or UNKNOWN>\n")
        prompt, used_ids, omitted = self._pack(base, blocks, tail)
        llm_started = time.perf_counter()
        completion, tokens_in, tokens_out = self.llm(prompt)
        llm_time = time.perf_counter() - llm_started
        if tokens_in > self.max_input_tokens:
            raise ValueError("model input exceeded max_input_tokens despite preflight count")
        match = re.search(r"ANSWER\s*:\s*(\S+)", completion or "")
        response = match.group(1).strip("`.,") if match else ((completion or "").strip().splitlines() or [""])[0]
        labels = {f"C{i}": c for i, c in enumerate(packet["candidates"], 1) if c["id"] in used_ids}
        chosen = labels.get(response.upper())
        self._remember(repo_id, packet, used_ids, omitted)
        self.sessions[repo_id]["last_prediction"] = chosen["id"] if chosen else None
        return {"predicted": chosen["name"] if chosen else "", "predicted_id": chosen["id"] if chosen else None,
                "snapshot_id": packet["snapshot_id"], "coverage": packet["coverage"],
                "source_scope": "repoqa_repository_payload",
                "n_local_functions": len(prepared["local_functions"]),
                "n_global_edges": len(packet["edges"]), "used_evidence_ids": used_ids,
                "gaps": packet["gaps"] + ([{"reason": "budget_omitted", "count": omitted}] if omitted else []),
                "local_cache_hit": prepared["local_cache_hit"],
                "prepare_reused": prepared["prepare_reused"],
                "global_index_time_s": prepared["global_index_time_s"],
                "local_extract_time_s": prepared["local_extract_time_s"],
                "extraction_time": prepared["global_index_time_s"] + prepared["local_extract_time_s"],
                "query_time": time.perf_counter() - query_started,
                "cache_hit": prepared["local_cache_hit"],
                "llm_time_s": llm_time,
                "input_tokens": tokens_in, "output_tokens": tokens_out}

    async def dependeval(self, repo_id: str, documents: dict[str, str], files: list[str]) -> dict:
        prepared = await self.prepare(repo_id, documents)
        extractor = PythonDependencyExtractor()
        probe = self.dependency_cache.wrap(extractor) if self.dependency_cache else extractor
        extract_started = time.perf_counter()
        raw = probe.extract(documents)
        dependency_cache_hit = bool(getattr(probe, "last_hit", False))
        if dependency_cache_hit:
            extractor.extract(documents)  # Rebuild the namespace used by derivation.
        dependency_extract_time = time.perf_counter() - extract_started
        query_started = time.perf_counter()
        spec = TaskSpec(domain="python_dependency_ordering", description="Order Python files",
                        input_schema={}, query={"kind": "ordering"})
        local = PythonDependencyEngine(extractor).derive(raw, [], spec)
        packet = await self._call("code_evidence", {
            "snapshot_id": prepared["snapshot_id"], "query": " ".join(Path(f).stem for f in files),
            "seeds_json": json.dumps([{"path": f} for f in files]),
            "max_candidates": 24, "max_edges": 120, "hops": 1,
        })
        _validate_packet(packet, prepared["snapshot_id"], documents)
        basename = {p: Path(p).name for p in files}
        base = ("Order these Python files so dependencies come before their users. "
                "Use the source-backed global relations to check the local extraction hints. "
                "An unresolved relation is not a constraint.\n"
                f"Files: {json.dumps(files)}\nSnapshot: {packet['snapshot_id']}\n"
                "Local file structure:\n")
        for path in files:
            imports = [line.strip() for line in documents.get(path, "").splitlines()
                       if line.strip().startswith(("import ", "from "))][:6]
            base += f"  {path}: imports={imports}\n"
        base += f"Local derived order hint: {json.dumps(local.ordering_hint)}\n"
        base += "Local derived relation hints (check against source-backed relations):\n"
        ns = extractor.last_namespace
        for edge in local.strong[:12]:
            source = ns.original_path_for(edge.subject) if ns else edge.subject
            target = ns.original_path_for(edge.object) if ns else edge.object
            base += f"  {source} -> {target}; {edge.provenance.get('edge_type', 'derived')}\n"
        base += f"Ambiguous local pairs: {len(local.ambiguous)}\n"
        base += "Source-backed global relations (dependent -> dependency):\n"
        blocks = []
        for i, edge in enumerate(packet["edges"]):
            source_path = edge["source"]["path"]
            target_path = edge["to_id"].split("|")[1]
            if source_path != target_path and source_path in basename and target_path in basename:
                blocks.append((str(i), f"  {source_path} -> {target_path}; {_edge_text(edge)}\n"))
        tail = ("\nUnresolved or absent evidence: " + "; ".join(_gap_text(g) for g in _prompt_gaps(packet["gaps"]))
                + "\nReturn only a JSON array of the input file paths in dependency order.\n")
        prompt, used_edges, omitted = self._pack(base, blocks, tail)
        llm_started = time.perf_counter()
        completion, tokens_in, tokens_out = self.llm(prompt)
        llm_time = time.perf_counter() - llm_started
        if tokens_in > self.max_input_tokens:
            raise ValueError("model input exceeded max_input_tokens despite preflight count")
        match = re.search(r"\[[\s\S]*?\]", completion or "")
        try:
            predicted = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            predicted = []
        if not isinstance(predicted, list):
            predicted = []
        predicted = [str(p) for p in predicted]
        self._remember(repo_id, packet, [packet["edges"][int(i)]["from_id"] for i in used_edges], omitted)
        return {"predicted_order": predicted, "snapshot_id": packet["snapshot_id"],
                "coverage": packet["coverage"], "n_local_primitives": len(raw),
                "source_scope": "dependeval_task_slice",
                "n_local_strong": len(local.strong), "n_global_edges": len(packet["edges"]),
                "used_global_edges": len(used_edges),
                "gaps": packet["gaps"] + ([{"reason": "budget_omitted", "count": omitted}] if omitted else []),
                "prepare_reused": prepared["prepare_reused"],
                "global_index_time_s": prepared["global_index_time_s"],
                "local_extract_time_s": prepared["local_extract_time_s"],
                "dependency_extract_time_s": dependency_extract_time,
                "extraction_time": (prepared["global_index_time_s"]
                                    + prepared["local_extract_time_s"] + dependency_extract_time),
                "query_time": time.perf_counter() - query_started,
                "cache_hit": dependency_cache_hit,
                "llm_time_s": llm_time,
                "input_tokens": tokens_in, "output_tokens": tokens_out}

    def _pack(self, base: str, blocks: list[tuple[str, str]], tail: str) -> tuple[str, list[str], int]:
        if self.llm.count_tokens(base + tail) > self.max_input_tokens:
            raise ValueError("fixed prompt exceeds max_input_tokens")
        chosen: list[str] = []
        parts: list[str] = []
        for key, block in blocks:
            trial = base + "".join(parts) + block + tail
            if self.llm.count_tokens(trial) <= self.max_input_tokens:
                parts.append(block)
                chosen.append(key)
        omitted = len(blocks) - len(chosen)
        if omitted:
            while True:
                note = f"\nBudget omitted {omitted} evidence groups; no conclusion follows from those omissions.\n"
                if self.llm.count_tokens(base + "".join(parts) + note + tail) <= self.max_input_tokens:
                    break
                if not parts:
                    raise ValueError("fixed prompt plus budget gap exceeds max_input_tokens")
                parts.pop()
                chosen.pop()
                omitted += 1
            tail = note + tail
        prompt = base + "".join(parts) + tail
        if self.llm.count_tokens(prompt) > self.max_input_tokens:
            raise AssertionError("prompt budget was not enforced")
        return prompt, chosen, omitted

    def _memory_context(self, repo_id: str, query: str, sid: str) -> str:
        record = self.sessions.get(repo_id)
        if not record or record["snapshot_id"] != sid:
            return ""
        if not re.search(r"\b(previous|earlier|above|that function)\b|之前|刚才|上一轮", query, re.I):
            return ""
        return ("Previous model selection (unverified): "
                + str(record.get("last_prediction")) + ". Prior verified evidence IDs: "
                + ", ".join(record["evidence_ids"][-4:])
                + ". Sources: "
                + "; ".join(f"{c['name']} at {c['path']}:{c['line']}" for c in record.get("candidates", [])[-4:])
                + ". Relations: "
                + "; ".join(_edge_text(e) for e in record.get("edges", [])[-4:])
                + ". Recheck their source before relying on a relation.")

    def _remember(self, repo_id: str, packet: dict, used_ids: list[str], omitted: int) -> None:
        old = self.sessions.get(repo_id, {})
        ids = old.get("evidence_ids", []) if old.get("snapshot_id") == packet["snapshot_id"] else []
        old_candidates = old.get("candidates", []) if old.get("snapshot_id") == packet["snapshot_id"] else []
        candidates = [c for c in packet["candidates"] if c["id"] in used_ids]
        old_edges = old.get("edges", []) if old.get("snapshot_id") == packet["snapshot_id"] else []
        edges = [e for e in packet["edges"] if e["from_id"] in used_ids or e["to_id"] in used_ids]
        self.sessions[repo_id] = {"snapshot_id": packet["snapshot_id"],
                                  "evidence_ids": list(dict.fromkeys([*ids, *used_ids]))[-64:],
                                  "candidates": list({c["id"]: c for c in [*old_candidates, *candidates]}.values())[-32:],
                                  "edges": (old_edges + edges)[-64:],
                                  "gaps": packet["gaps"] + ([{"reason": "budget_omitted", "count": omitted}] if omitted else [])}


def _digest(documents: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, source in sorted(documents.items()):
        for value in (path, str(source)):
            raw = value.encode()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _source_excerpt(source: str, line: int, max_lines: int = 9) -> str:
    lines = source.splitlines()
    return "\n".join(lines[max(0, line - 1):line - 1 + max_lines])[:420]


def _edge_text(edge: dict) -> str:
    src = edge["source"]
    return (f"{edge['relation']} {edge['from_id'].split('|')[2]} -> "
            f"{edge['to_id'].split('|')[2]} at {src['path']}:{src['line']}")


def _gap_text(gap: dict) -> str:
    if "source" in gap:
        src = gap["source"]
        return f"  {gap['reason']} {gap.get('target_text', '')} at {src['path']}:{src['line']}"
    if gap["reason"] == "seed_not_in_graph":
        return f"  seed_not_in_graph {gap.get('name')} at {gap.get('path')}:{gap.get('line')}"
    if gap["reason"] == "duplicate_definition":
        return f"  duplicate_definition {gap.get('qual')}"
    return f"  {gap['reason']} {gap.get('path', gap.get('count', gap.get('omitted_count', '')))}"


def _prompt_gaps(gaps: list[dict], limit: int = 12) -> list[dict]:
    reasons = ("edge_limit", "gap_limit", "not_indexed", "seed_not_in_graph", "duplicate_definition")
    summary = [g for g in gaps if g["reason"] in reasons]
    details = [g for g in gaps if g["reason"] not in reasons]
    return (summary + details)[:limit]


def _validate_packet(packet: dict, snapshot_id: str, documents: dict[str, str]) -> None:
    """Reject stale or untraceable cross-layer evidence before prompting."""
    if packet.get("snapshot_id") != snapshot_id:
        raise ValueError("global evidence snapshot does not match local documents")
    if not isinstance(packet.get("coverage"), dict) or not isinstance(packet.get("gaps"), list):
        raise ValueError("global evidence packet is incomplete")
    for candidate in packet.get("candidates", []):
        if not candidate["id"].startswith(snapshot_id + "|"):
            raise ValueError("candidate belongs to another snapshot")
        if candidate["path"] not in documents or candidate["line"] < 1:
            raise ValueError("candidate has no source location")
    for edge in packet.get("edges", []):
        source = edge["source"]
        if not edge["from_id"].startswith(snapshot_id + "|") or not edge["to_id"].startswith(snapshot_id + "|"):
            raise ValueError("relation belongs to another snapshot")
        if (source["path"] not in documents or source["line"] < 1 or
                source["line"] > len(documents[source["path"]].splitlines())):
            raise ValueError("relation source site cannot be checked")
