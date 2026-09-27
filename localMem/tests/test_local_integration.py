"""Offline checks for the local caller and benchmark pipeline boundary."""

import contextlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from benchmarks.local_model import LocalQwen3


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
                if not kwargs.get("return_dict"):
                    return [1, 2, 3]
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
                counted = llm.count_tokens("calculate area")

        self.assertEqual(result, ("compute_area", 3, 2))
        self.assertEqual(counted, 3)
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


if __name__ == "__main__":
    unittest.main()
