"""Groq hosted-API generator.

Drop-in replacement for ``src.generator.Generator`` that calls the Groq
cloud API instead of running a local Hugging Face model. Used when
``LLM_BACKEND == "groq"`` in ``src/config.py``.

Setup
-----
1. Get a free API key from https://console.groq.com/keys
2. Set the environment variable ``GROQ_API_KEY`` (PowerShell:
   ``$env:GROQ_API_KEY = "gsk_..."``) before launching Streamlit.

The free tier is rate-limited (~30 requests/min) but plenty for a demo.
"""
from __future__ import annotations

import os
from typing import Iterable

from .config import (
    GROQ_MODEL_NAME,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
)


class GroqGenerator:
    """Mirrors ``Generator`` so the rest of the pipeline doesn't care."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None) -> None:
        from groq import Groq  # type: ignore

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and set it before launching:\n"
                '    $env:GROQ_API_KEY = "gsk_..."'
            )
        self.model_name = model_name or GROQ_MODEL_NAME
        self.client = Groq(api_key=key)

    def generate_from_messages(
        self,
        messages: Iterable[dict],
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()

    def generate_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> str:
        # Wrap the bare prompt as a single user message.
        return self.generate_from_messages(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
