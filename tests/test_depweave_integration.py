"""Real MCP boundary checks with a deterministic stand-in for Qwen generation."""

from __future__ import annotations

import importlib.util
import json
import re
import unittest
from pathlib import Path

from depweave import DepWeaveRunner
from experiments.exp56_repoqa_python_full import run_cases
from experiments.exp58_v5_depeval_full import run_items


class FakeModel:
    def __init__(self):
        self.prompts = []

    def count_tokens(self, prompt):
        return len(prompt.split()) + 8

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if "Candidates and their source evidence" in prompt:
            candidate = re.search(r"^(C\d+) ID ", prompt, re.M)
            answer = f"ANSWER: {candidate.group(1)}" if candidate else "ANSWER: UNKNOWN"
        else:
            files = json.loads(re.search(r"^Files: (\[.*\])$", prompt, re.M).group(1))
            answer = json.dumps(list(reversed(files)))
        return answer, self.count_tokens(prompt), len(answer.split())


@unittest.skipUnless(importlib.util.find_spec("fastmcp"), "fastmcp is required for MCP integration")
class DepWeaveIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_graph_preserves_sites_and_refuses_unbound_calls(self):
        docs = {
            "pkg/__init__.py": "from .impl import target\n",
            "pkg/impl.py": "def target(): pass\n",
            "other.py": "def target(): pass\ndef lonely(): pass\ndef m(): pass\n",
            "client.py": "from pkg import target as alias\ndef run(x):\n    alias()\n    alias()\n    x.m()\n",
            "unbound.py": "def check():\n    lonely()\n",
            "same.py": "def repeat(): pass\ndef repeat(): pass\n",
            "bad.py": "def broken(:\n",
        }
        async with DepWeaveRunner(FakeModel()) as runner:
            prep = await runner.prepare("test-graph", docs)
            packet = await runner._call("code_evidence", {
                "snapshot_id": prep["snapshot_id"], "query": "run",
                "seeds_json": json.dumps([{"path": "client.py", "line": 2, "name": "run"},
                                          {"path": "unbound.py", "line": 1, "name": "check"}]),
            })
        alias_calls = [e for e in packet["edges"] if e["relation"] == "calls"
                       and e["source"]["path"] == "client.py"
                       and "pkg/impl.py::target" in e["to_id"]]
        self.assertEqual([e["source"]["line"] for e in alias_calls], [3, 4])
        self.assertTrue(any(g.get("target_text") == "m" for g in packet["gaps"]))
        self.assertTrue(any(g.get("target_text") == "lonely" for g in packet["gaps"]))
        self.assertEqual(packet["coverage"]["parse_error_files"], ["bad.py"])
        self.assertIn("same.py::repeat", packet["coverage"]["duplicate_qualnames"])
        self.assertTrue(any(g["reason"] == "duplicate_definition" for g in packet["gaps"]))

    async def test_official_entrypoints_use_both_layers(self):
        repo_docs = {"a.py": "def target():\n    \"\"\"Find special value.\"\"\"\n    return 1\n",
                     "b.py": "from a import target\ndef use():\n    return target()\n"}
        repo_case = {"repo": "toy/repo", "content": repo_docs, "name": "target",
                     "description": "Find special value", "path": "a.py", "line": 1}
        model = FakeModel()
        repo_records = await run_cases([repo_case], model)
        self.assertTrue(repo_records[0]["exact"])
        self.assertTrue(repo_records[0]["entity_exact"])
        self.assertGreater(repo_records[0]["n_local_functions"], 0)
        self.assertGreater(repo_records[0]["n_global_edges"], 0)
        self.assertIn("Repository snapshot:", model.prompts[0])

        item = {"files": ["base.py", "mid.py", "app.py"],
                "gt": ["app.py", "mid.py", "base.py"],
                "content": "'base.py'\n:def target(): pass\n'mid.py'\n:from base import target\ndef use(): target()\n'app.py'\n:from mid import use\ndef main(): use()"}
        dep_records = await run_items([item], model)
        self.assertGreater(dep_records[0]["n_local_primitives"], 0)
        self.assertGreater(dep_records[0]["n_global_edges"], 0)
        self.assertIn("Source-backed global relations", model.prompts[1])

    async def test_budget_and_snapshot_invalidation(self):
        docs = {"many.py": "\n".join(f"def f{i}():\n    \"\"\"function {i} with detail detail detail.\"\"\"\n    return {i}"
                                      for i in range(20))}
        model = FakeModel()
        async with DepWeaveRunner(model, max_input_tokens=256) as runner:
            first = await runner.repoqa("changing", docs, "function with detail")
            self.assertLessEqual(first["input_tokens"], 256)
            self.assertTrue(any(g["reason"] == "budget_omitted" for g in first["gaps"]))
            self.assertIn("Budget omitted", model.prompts[0])
            self.assertIn("changing", runner.sessions)
            changed = {"many.py": docs["many.py"] + "\ndef added(): pass\n"}
            second = await runner.prepare("changing", changed)
            self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
            self.assertNotIn("changing", runner.sessions)

    async def test_followup_reuses_source_but_marks_model_choice_unverified(self):
        docs = {"a.py": "def target():\n    return 1\n",
                "b.py": "from a import target\ndef use(): return target()\n"}
        model = FakeModel()
        async with DepWeaveRunner(model) as runner:
            first = await runner.repoqa("followup", docs, "target function")
            second = await runner.repoqa("followup", docs, "previous function")
        self.assertEqual(first["snapshot_id"], second["snapshot_id"])
        self.assertIn("Previous model selection (unverified)", model.prompts[1])
        self.assertIn("Sources:", model.prompts[1])

    async def test_name_match_does_not_hide_wrong_entity(self):
        docs = {"a.py": "def target():\n    return 'first'\n",
                "b.py": "def target():\n    return 'second'\n"}
        case = {"repo": "duplicate", "content": docs, "name": "target",
                "description": "target", "path": "b.py", "line": 1}
        result = (await run_cases([case], FakeModel()))[0]
        self.assertTrue(result["exact"])
        self.assertFalse(result["entity_exact"])


if __name__ == "__main__":
    unittest.main()
