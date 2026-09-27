"""Exp 62 — One end-to-end case: code_graph vs CodeGraph (CodexGraph / RepoGraph).

Task (a real agent/dev question that NEEDS a code graph):
    "Which function computes the retry backoff delay, and what breaks if I
     change its signature?"  -> localization + blast-radius.

We run it for real through our code_graph plugin and measure the prompt we would
hand the LLM, then compare — honestly, not strawmanned — against the two ways the
recent code-graph systems answer the same question:

  * RepoGraph  : ego-graph retrieval returns the RAW SOURCE LINES of the
                 neighbourhood, appended to the prompt; the LLM must still read
                 the code and derive the blast radius itself. (1 turn, big.)
  * CodexGraph : the LLM AUTHORS Cypher against a Neo4j graph over MULTIPLE
                 agent turns (find fn -> find callers -> callers-of-callers ...),
                 each turn round-tripping schema + query + results, and needs the
                 DB stood up. (N turns, infra, brittle queries.)
  * full-source: dump every file. (1 turn, biggest, LLM derives everything.)

Our path: extract -> derive (calls->file_depends_on, reverse call chain) ->
ONE compact packet. The answer structure is pre-derived and handed over.

Offline, no network. Run: python experiments/exp62_vs_codegraph_case.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelmem.v5.plugins.code_graph import CodeGraphExtractor, CodeGraphIndex
from pixelmem.v5.plugins.code_graph.index import approx_tokens


# --------------------------------------------------------------------------
# A small but realistic repo. Retry logic is spread across files; the answer
# to "what breaks if I change compute_delay" is a transitive caller chain.
# --------------------------------------------------------------------------

REPO: dict[str, str] = {
    "backoff.py": '''
"""Backoff helpers for the HTTP client."""
import random


def compute_delay(attempt, base=0.5, cap=30.0):
    """Exponential backoff with jitter for retry attempt `attempt`."""
    raw = base * (2 ** attempt)
    return min(cap, raw) * (0.5 + random.random() / 2)


def jitter(value):
    """Add +/-10% jitter to a value."""
    return value * (0.9 + random.random() * 0.2)
''',
    "client.py": '''
"""A small HTTP client with retry-on-429 logic."""
import time

from backoff import compute_delay


class HttpClient:
    """Minimal retrying HTTP client."""

    def __init__(self, session, max_attempts=5):
        self.session = session
        self.max_attempts = max_attempts

    def _send_with_retry(self, method, url, body=None):
        """Send a request, retrying with backoff on HTTP 429."""
        for attempt in range(self.max_attempts):
            resp = self.session.send(method, url, body)
            if resp.status != 429:
                return resp
            delay = compute_delay(attempt)
            time.sleep(delay)
        return resp

    def request(self, method, url, body=None):
        """Public entry point — delegates to the retrying sender."""
        return self._send_with_retry(method, url, body)

    def get(self, url):
        """HTTP GET."""
        return self.request("GET", url)

    def post(self, url, body):
        """HTTP POST."""
        return self.request("POST", url, body)
''',
    # --- noise files: realistic repo bulk a naive approach would dump ---
    "models.py": '''
"""Domain models — unrelated to retry logic."""
from dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str
    email: str

    def display(self):
        return f"{self.name} <{self.email}>"


@dataclass
class Account:
    owner: User
    balance: float

    def deposit(self, amount):
        self.balance += amount
        return self.balance
''',
    "serializers.py": '''
"""JSON (de)serialization helpers — unrelated to retry logic."""
import json


def to_json(obj):
    """Serialize a dataclass-like object to JSON."""
    return json.dumps(obj.__dict__)


def from_json(blob):
    """Parse a JSON blob into a dict."""
    return json.loads(blob)
''',
}

QUERY = ("Which function computes the retry backoff delay, and what breaks if I "
         "change its signature?")


def func_source(name: str) -> str:
    """The raw source segment of the first function named `name` (for the
    RepoGraph raw-ego estimate)."""
    for src in REPO.values():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return ast.get_source_segment(src, node) or ""
    return ""


def main() -> None:
    ex = CodeGraphExtractor()
    prims = ex.extract(REPO)
    idx = CodeGraphIndex.from_primitives(prims)

    # ---- OUR PATH: one shot, pre-derived ---------------------------------
    hit = idx.search_symbol("retry backoff delay", k=1)[0]
    qual = hit["qual"]
    blast = idx.trace_call_chain(qual, direction="in", max_depth=4)
    usage = idx.locate_usage(qual)
    file_impact = idx.find_dependencies(qual.split("::")[0], reverse=True)

    impacted: list[str] = []
    for ch in blast["chains"]:
        for sym in ch.split(" -> ")[1:]:
            if sym not in impacted:
                impacted.append(sym)
    example = blast["chains"][0].replace(" -> ", " <- ") if blast["chains"] else "(none)"

    packet = [
        f"match: {qual}{hit['signature']}  [{hit['kind']}, {hit['file']}]",
        "impacts (transitive callers): " + (", ".join(impacted) or "(none)"),
        "example chain: " + example,
        f"call sites: {usage['n_sites']} -> " +
        ", ".join(f"{s['by']} ({s['file']}:{s.get('line','?')})" for s in usage["sites"]),
        "file impact: " + (", ".join(
            f"{d['file']} depends_on {file_impact['file']} (via {', '.join(d['via'])})"
            for d in file_impact.get("depended_on_by", [])) or "(none cross-file)"),
    ]
    our_packet = "\n".join(packet)
    our_prompt = (f"Question: {QUERY}\n\n"
                  f"Code-graph tool result (derived, one shot):\n{our_packet}\n\n"
                  "Answer using the tool result.")

    print("=" * 74)
    print("CASE:", QUERY)
    print("=" * 74)
    print("\n--- OUR code_graph: the ONE derived packet handed to the LLM ---\n")
    print(our_packet)
    print(f"\n  -> our full prompt: ~{approx_tokens(our_prompt)} tokens, 1 LLM turn, "
          f"no DB, blast radius PRE-DERIVED")

    # ---- RepoGraph-style: raw-line ego-graph of the neighbourhood --------
    neighbourhood = ["compute_delay", "_send_with_retry", "request", "get", "post"]
    raw_ego = "\n\n".join(func_source(n) for n in neighbourhood)
    repo_prompt = (f"Question: {QUERY}\n\nRelevant code (ego-graph):\n{raw_ego}\n\n"
                   "Answer.")

    # ---- full-source dump ------------------------------------------------
    full = "\n\n".join(f"# {p}\n{s}" for p, s in REPO.items())

    print("\n--- contrast (same correct answer; cost to GET there) ---\n")
    rows = [
        ("our code_graph (derived packet)", approx_tokens(our_prompt), "1",
         "no", "tool (pre-derived)"),
        ("RepoGraph (raw ego-graph lines)", approx_tokens(repo_prompt), "1",
         "no", "LLM reads code"),
        ("CodexGraph (LLM writes Cypher)", _codexgraph_est(), "3-4",
         "Neo4j", "LLM, across turns"),
        ("full-source dump", approx_tokens(full + QUERY), "1",
         "no", "LLM reads everything"),
    ]
    print(f"  {'approach':34} {'~tokens':>8} {'turns':>6} {'infra':>7}  who derives blast radius")
    print("  " + "-" * 86)
    for name, tok, turns, infra, who in rows:
        print(f"  {name:34} {tok:>8} {turns:>6} {infra:>7}  {who}")

    base = approx_tokens(our_prompt)
    print(f"\n  vs RepoGraph raw-ego : {approx_tokens(repo_prompt)/base:.1f}x more tokens, "
          f"and the LLM still has to derive the blast radius from code")
    print(f"  vs full-source dump  : {approx_tokens(full)/base:.1f}x more tokens")
    print("  vs CodexGraph        : 3-4 LLM turns + a running Neo4j + the LLM has to")
    print("                         author correct Cypher each turn; we answer in one.")

    # sanity: we actually found the right thing
    assert qual.endswith("compute_delay"), qual
    assert any("post" in c or "get" in c for c in blast["chains"]), blast["chains"]
    print("\n[verified] tool localized compute_delay and traced the caller chain to get/post")


def _codexgraph_est() -> str:
    """Rough lower bound: 3 Cypher turns, each ~schema(120) + query(40) +
    results(80) in, returned to the model. Reported as a range, not false
    precision — the point is it is multi-turn, not the exact count."""
    per_turn = 120 + 40 + 80
    return f"~{per_turn*3}+"


if __name__ == "__main__":
    main()
