"""FAISS-backed vector store with side-car JSON metadata.

We use a normalised cosine-similarity index (``IndexFlatIP`` over
unit-norm vectors) to keep the implementation simple and exact — fine
for a graduation-project corpus of a few thousand chunks.

Files written
-------------
* ``faiss.index`` — the FAISS index
* ``chunks.json`` — the chunk metadata in the same order as the index

Both paths are configurable but default to ``data/processed/`` from
``config.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import CHUNKS_PATH, INDEX_PATH


class VectorStore:
    def __init__(self) -> None:
        self.index = None  # type: ignore[assignment]
        self.chunks: list[dict] = []
        self._dim: int | None = None

    # --- Build ----------------------------------------------------------
    def build(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings have different lengths")

        import faiss  # type: ignore

        self._dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(self._dim)
        index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        self.index = index
        self.chunks = list(chunks)

    # --- Persistence ----------------------------------------------------
    def save(
        self,
        index_path: Path | str = INDEX_PATH,
        chunks_path: Path | str = CHUNKS_PATH,
    ) -> None:
        if self.index is None:
            raise RuntimeError("Vector store has no index built; call build() first.")

        import faiss  # type: ignore

        index_path = Path(index_path)
        chunks_path = Path(chunks_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        chunks_path.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        index_path: Path | str = INDEX_PATH,
        chunks_path: Path | str = CHUNKS_PATH,
    ) -> "VectorStore":
        import faiss  # type: ignore

        store = cls()
        store.index = faiss.read_index(str(index_path))
        store.chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        store._dim = store.index.d
        return store

    # --- Search ---------------------------------------------------------
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Vector store has no index loaded.")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        scores, indices = self.index.search(
            np.ascontiguousarray(query_embedding, dtype="float32"),
            top_k,
        )
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results


# --- CLI: build the index from data/raw/ ----------------------------------
def build_index_from_raw(
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> tuple[VectorStore, dict]:
    """End-to-end: load raw -> chunk -> embed -> save FAISS index."""
    from .chunker import chunk_documents
    from .config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS
    from .data_loader import load_documents
    from .embeddings import EmbeddingModel

    documents = load_documents()
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size or CHUNK_SIZE_WORDS,
        overlap=overlap or CHUNK_OVERLAP_WORDS,
    )

    print(f"Embedding {len(chunks)} chunks with the embedding model ...")
    model = EmbeddingModel()
    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        show_progress_bar=True,
    )

    store = VectorStore()
    store.build(chunks, embeddings)
    store.save()

    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_dim": store._dim,
        "embedding_model": model.model_name,
    }
    return store, stats


if __name__ == "__main__":  # pragma: no cover
    _, stats = build_index_from_raw()
    print("Index built:", stats)
    print(f"  index -> {INDEX_PATH}")
    print(f"  chunks -> {CHUNKS_PATH}")
