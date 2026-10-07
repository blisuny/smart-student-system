"""Tests for retriever + vector store using a stub embedder.

These tests avoid the heavy ``sentence_transformers`` dependency by
injecting a deterministic in-memory embedder.
"""
from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # noqa: F841

from src.retriever import Retriever
from src.vector_store import VectorStore


class _StubEmbedder:
    """Maps a small vocabulary of phrases to fixed unit vectors."""

    def __init__(self) -> None:
        self.dim = 4
        self.table = {
            "library hours":  np.array([1, 0, 0, 0], dtype="float32"),
            "course registration": np.array([0, 1, 0, 0], dtype="float32"),
            "graduation report":   np.array([0, 0, 1, 0], dtype="float32"),
            "cafeteria":           np.array([0, 0, 0, 1], dtype="float32"),
        }

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            tl = t.lower()
            best_key, best = None, -1.0
            for k in self.table:
                # crude longest-common-keyword match
                if k in tl:
                    score = len(k)
                    if score > best:
                        best, best_key = score, k
            out[i] = self.table[best_key] if best_key else np.array([1, 1, 1, 1], dtype="float32") / 2.0
        # normalise
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype("float32")


def _make_chunks() -> list[dict]:
    base = [
        ("c1", "library hours weekdays 07:30 to 21:00"),
        ("c2", "course registration is performed through OBS"),
        ("c3", "graduation project report sections"),
        ("c4", "cafeteria opens at 8 in the morning"),
    ]
    chunks = []
    for cid, text in base:
        chunks.append({
            "chunk_id": cid,
            "document_id": cid + "_doc",
            "language": "en",
            "title": cid,
            "text": text,
            "source": "test",
        })
    return chunks


def test_search_returns_top_match():
    chunks = _make_chunks()
    embedder = _StubEmbedder()
    embeddings = embedder.encode([c["text"] for c in chunks])

    store = VectorStore()
    store.build(chunks, embeddings)

    retriever = Retriever(store=store, embedder=embedder)
    hits = retriever.retrieve("library hours", top_k=1)
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["score"] > 0


def test_search_top_k_orders_by_score():
    chunks = _make_chunks()
    embedder = _StubEmbedder()
    embeddings = embedder.encode([c["text"] for c in chunks])

    store = VectorStore()
    store.build(chunks, embeddings)

    retriever = Retriever(store=store, embedder=embedder)
    hits = retriever.retrieve("graduation report", top_k=3)
    assert len(hits) == 3
    assert hits[0]["chunk_id"] == "c3"
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_save_and_load_round_trip(tmp_path):
    chunks = _make_chunks()
    embedder = _StubEmbedder()
    embeddings = embedder.encode([c["text"] for c in chunks])

    store = VectorStore()
    store.build(chunks, embeddings)

    index_path = tmp_path / "x.index"
    chunks_path = tmp_path / "x.chunks.json"
    store.save(index_path=index_path, chunks_path=chunks_path)

    reloaded = VectorStore.load(index_path=index_path, chunks_path=chunks_path)
    assert len(reloaded.chunks) == len(chunks)
    hits = reloaded.search(embedder.encode("cafeteria"), top_k=1)
    assert hits[0]["chunk_id"] == "c4"
