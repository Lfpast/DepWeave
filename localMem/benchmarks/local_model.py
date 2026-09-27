"""Offline Qwen3-4B caller shared by all benchmark drivers."""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path


MODEL_NAME = "Qwen3-4B"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "qwen3-4b"


def add_model_arguments(parser: argparse.ArgumentParser, *, max_new_tokens: int) -> None:
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN3_4B_PATH"),
        help="Local Qwen3-4B checkpoint directory (or set QWEN3_4B_PATH).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)


class LocalQwen3:
    """Single local model instance matching the pipeline's LLMCaller protocol."""

    def __init__(self, model_path: str | None, *, max_new_tokens: int = 512) -> None:
        if not model_path:
            raise ValueError("Set --model-path or QWEN3_4B_PATH to a local Qwen3-4B directory")
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Local Qwen3-4B directory does not exist: {path}")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        # Import at construction time so --help and offline interface tests do
        # not need the model environment. local_files_only prevents downloads.
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install shortMem/requirements.txt for local Qwen3 inference") from exc

        self.path = path
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype="auto", device_map="auto", local_files_only=True
        )
        self._model.eval()
        self._lock = threading.Lock()

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        # A shared Transformers model is serialized even if a driver runs
        # extraction tasks in multiple threads.
        with self._lock:
            inputs = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_tokens = inputs["input_ids"].shape[-1]
            if input_tokens + self.max_new_tokens > 32768:
                raise ValueError(
                    f"Prompt has {input_tokens} tokens; with {self.max_new_tokens} output tokens "
                    "it exceeds Qwen3-4B's native 32768-token context"
                )
            inputs = inputs.to(self._model.device)
            with self._torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_ids = output[0][input_tokens:]
            completion = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            return completion, input_tokens, len(new_ids)
