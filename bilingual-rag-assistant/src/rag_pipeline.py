"""End-to-end RAG pipeline used by both the FastAPI backend and Streamlit UI.

The pipeline is a thin coordinator:

    question → detect_language → retrieve top-k → build prompt → generate

Heavy components (embedding model, generation model, FAISS index) are
loaded lazily and cached so that the very first call may be slow but
subsequent calls reuse them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import LLM_BACKEND, TOP_K
from .language_detector import detect_language
from .prompt_builder import build_chat_messages, build_prompt
from .retriever import Retriever


@dataclass
class RAGResponse:
    question: str
    detected_language: str
    retrieved_chunks: list[dict]
    answer: str
    prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "detected_language": self.detected_language,
            "retrieved_chunks": self.retrieved_chunks,
            "answer": self.answer,
        }


@dataclass
class RAGPipeline:
    top_k: int = TOP_K
    retriever: Optional[Retriever] = None
    _generator: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.retriever is None:
            self.retriever = Retriever()

    # --- Lazy singletons -----------------------------------------------
    def _ensure_generator(self):
        if self._generator is None:
            if LLM_BACKEND == "groq":
                from .groq_generator import GroqGenerator
                self._generator = GroqGenerator()
            else:
                from .generator import Generator
                self._generator = Generator()
        return self._generator

    # --- Public --------------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: int | None = None,
        language_override: str | None = None,
    ) -> RAGResponse:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        k = top_k or self.top_k

        language = language_override or detect_language(question)
        chunks = self.retriever.retrieve(question, top_k=k)

        try:
            generator = self._ensure_generator()
        except Exception as exc:  # pragma: no cover - depends on env
            return RAGResponse(
                question=question,
                detected_language=language,
                retrieved_chunks=chunks,
                answer=(
                    f"[Generator unavailable: {exc!s}]\n"
                    "Top retrieved context is shown above; you can read the "
                    "chunks directly while you set up the generation model."
                ),
            )

        # Prefer the chat template if the tokenizer supports it.
        try:
            messages = build_chat_messages(question, chunks, language)
            answer_text = generator.generate_from_messages(messages)
            prompt_for_display = "\n\n".join(
                f"[{m['role']}]\n{m['content']}" for m in messages
            )
        except Exception:
            prompt_for_display = build_prompt(question, chunks, language)
            answer_text = generator.generate_from_prompt(prompt_for_display)

        return RAGResponse(
            question=question,
            detected_language=language,
            retrieved_chunks=chunks,
            answer=answer_text,
            prompt=prompt_for_display,
        )


# Lightweight module-level helper so callers don't always have to manage
# a long-lived pipeline object themselves.
_default_pipeline: RAGPipeline | None = None


def answer_question(question: str, top_k: int | None = None) -> RAGResponse:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    return _default_pipeline.answer(question, top_k=top_k)
