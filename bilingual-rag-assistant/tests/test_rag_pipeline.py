"""End-to-end pipeline test using fakes for the heavy components.

We swap in a stub Retriever and a stub Generator so the test runs
without downloading any models or building a real FAISS index.
"""
from __future__ import annotations

from src.prompt_builder import build_chat_messages, build_prompt
from src.rag_pipeline import RAGPipeline


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, question, top_k=5):
        return list(self._chunks[:top_k])


class _FakeGenerator:
    def __init__(self, reply: str = "fake answer"):
        self.reply = reply
        self.last_messages = None
        self.last_prompt = None

    def generate_from_messages(self, messages, **_):
        self.last_messages = list(messages)
        return self.reply

    def generate_from_prompt(self, prompt, **_):
        self.last_prompt = prompt
        return self.reply


def _chunk(idx: int, lang: str = "en") -> dict:
    return {
        "chunk_id": f"c{idx}",
        "document_id": f"doc{idx}",
        "language": lang,
        "title": f"Title {idx}",
        "text": f"Context body {idx}.",
        "source": "test",
        "score": 1.0 - idx * 0.1,
    }


def test_pipeline_returns_question_language_chunks_and_answer():
    chunks = [_chunk(0, "en"), _chunk(1, "en")]
    pipeline = RAGPipeline(retriever=_FakeRetriever(chunks))
    pipeline._generator = _FakeGenerator("hello")
    result = pipeline.answer("What sections should a graduation project report include?")

    assert result.question.startswith("What sections")
    assert result.detected_language == "en"
    assert [c["chunk_id"] for c in result.retrieved_chunks] == ["c0", "c1"]
    assert result.answer == "hello"


def test_pipeline_detects_turkish():
    chunks = [_chunk(0, "tr")]
    pipeline = RAGPipeline(retriever=_FakeRetriever(chunks))
    pipeline._generator = _FakeGenerator("merhaba")
    result = pipeline.answer("Mezuniyet projesi raporunda hangi bölümler olmalı?")
    assert result.detected_language == "tr"
    assert result.answer == "merhaba"


def test_prompt_builder_includes_context_and_question():
    chunks = [_chunk(0)]
    prompt = build_prompt("Where is the cafeteria?", chunks, "en")
    assert "Where is the cafeteria?" in prompt
    assert "Context body 0" in prompt
    assert "Answer" in prompt  # English section header


def test_prompt_builder_turkish_uses_turkish_headers():
    chunks = [_chunk(0, "tr")]
    prompt = build_prompt("Kütüphane nerede?", chunks, "tr")
    assert "Cevap" in prompt
    assert "Kütüphane nerede?" in prompt


def test_chat_messages_have_system_and_user_roles():
    msgs = build_chat_messages("Hi", [_chunk(0)], "en")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Hi" in msgs[1]["content"]
