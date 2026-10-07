"""Hugging Face text generation wrapper.

Default model is ``Qwen/Qwen3-0.6B`` (small, runs on CPU). To switch to a
larger model edit ``GENERATION_MODEL_NAME`` in ``src/config.py``.

The class auto-detects CUDA / MPS / CPU at construction time. It accepts
either a plain prompt string or a list of chat messages.
"""
from __future__ import annotations

import re
from typing import Iterable

from .config import (
    GENERATION_MODEL_NAME,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
)

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by Qwen3 reasoning mode."""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # If the model started thinking but never closed the tag, drop everything
    # up to the last </think> if present, otherwise drop the dangling block.
    if "<think>" in cleaned.lower():
        idx = cleaned.lower().rfind("</think>")
        cleaned = cleaned[idx + len("</think>"):] if idx >= 0 else ""
    return cleaned.strip()


def _resolve_device() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


class Generator:
    """Lazy-loaded HF causal-LM generator."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.model_name = model_name or GENERATION_MODEL_NAME
        self.device = device or _resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Pick a sensible dtype: fp16 on GPU, fp32 on CPU.
        import torch  # type: ignore

        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    # --- Generation -----------------------------------------------------
    def generate_from_messages(
        self,
        messages: Iterable[dict],
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Tokenizer doesn't support enable_thinking (non-Qwen3 model).
            prompt = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        return self._generate(prompt, max_new_tokens, temperature, top_p)

    def generate_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> str:
        return self._generate(prompt, max_new_tokens, temperature, top_p)

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        import torch  # type: ignore

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Strip the prompt tokens from the output.
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return _strip_thinking(text)
