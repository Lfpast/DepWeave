"""Offline checks for the local caller and benchmark pipeline boundary."""

import contextlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from benchmarks.local_model import LocalQwen3
from experiments.exp56_repoqa_python_full import _run_cardmem
from experiments.exp58_v5_depeval_full import process_q
from core.cache import PixelMemCache


class LocalModelTests(unittest.TestCase):
    def test_local_checkpoint_and_token_contract(self):
        calls = {}

        class Inputs(dict):
            def to(self, device):
                calls["device"] = device
                return self

        class Tokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                calls["messages"] = messages
                calls["template"] = kwargs
                return Inputs(input_ids=types.SimpleNamespace(shape=(1, 3)))

            def decode(self, ids, **kwargs):
                calls["decode"] = kwargs
                return "compute_area"

        class Model:
            device = "cpu"

            def eval(self):
                return self

            def generate(self, **kwargs):
                calls["generate"] = kwargs
                return [[1, 2, 3, 4, 5]]

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls["tokenizer_load"] = kwargs
                return Tokenizer()

        class AutoModel:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls["model_load"] = kwargs
                return Model()

        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModel
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"torch": fake_torch,
                                         "transformers": fake_transformers}):
                llm = LocalQwen3(directory, max_new_tokens=10)
                result = llm("calculate area")

        self.assertEqual(result, ("compute_area", 3, 2))
        self.assertEqual(calls["tokenizer_load"], {"local_files_only": True})
        self.assertTrue(calls["model_load"]["local_files_only"])
        self.assertFalse(calls["template"]["enable_thinking"])
        self.assertEqual(calls["messages"][0]["content"], "calculate area")
        self.assertEqual(calls["generate"]["max_new_tokens"], 10)

    def test_requires_existing_local_path(self):
        with self.assertRaises(ValueError):
            LocalQwen3(None)
        with self.assertRaises(FileNotFoundError):
            LocalQwen3("/nonexistent/qwen3-4b")


class BenchmarkBoundaryTests(unittest.TestCase):
    def test_dependeval_uses_caller_and_counts_tokens(self):
        item = {
            "files": ["a.py", "b.py", "c.py"],
            "gt": ["a.py", "b.py", "c.py"],
            "content": "'a.py':\nX=1\n'b.py':\nimport a\n'c.py':\nimport b",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = process_q(
                0, item, PixelMemCache(directory),
                lambda prompt: ('["a.py", "b.py", "c.py"]', 30, 12),
            )
        self.assertTrue(result["exact"])
        self.assertEqual((result["input_tokens"], result["output_tokens"]),
                         (30, 12))

    def test_repoqa_uses_caller_and_counts_tokens(self):
        case = {
            "repo": "org/test",
            "content": {"a.py": "def compute_area(width, height):\n"
                        "    \"\"\"Calculate rectangle area.\"\"\"\n"
                        "    return width * height\n"},
            "needle_name": "compute_area",
            "needle_description": "Calculate rectangle area given width and height",
            "needle_path": "a.py",
        }
        result = _run_cardmem(case, lambda prompt: ("compute_area", 20, 5), None)
        self.assertTrue(result["exact"])
        self.assertEqual((result["input_tokens"], result["output_tokens"]),
                         (20, 5))


if __name__ == "__main__":
    unittest.main()
