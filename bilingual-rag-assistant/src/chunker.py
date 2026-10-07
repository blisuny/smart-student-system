"""Word-based document chunker.

Each input document (see ``data_loader.load_documents``) is split into
overlapping word windows. Metadata is preserved on every chunk.
"""
from __future__ import annotations

import re
from typing import Iterable

from .config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS

_WORD_SPLIT = re.compile(r"\s+")


def _split_words(text: str) -> list[str]:
    return [w for w in _WORD_SPLIT.split(text.strip()) if w]


def chunk_document(
    document: dict,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    """Split one document into ``{chunk_id, document_id, language, ...}`` chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    words = _split_words(document["text"])
    if not words:
        return []

    chunks: list[dict] = []
    step = chunk_size - overlap
    for start, idx in zip(range(0, len(words), step), range(10**9)):
        end = min(start + chunk_size, len(words))
        piece = " ".join(words[start:end])
        chunks.append({
            "chunk_id": f"{document['id']}__c{idx:03d}",
            "document_id": document["id"],
            "language": document["language"],
            "title": document.get("title", ""),
            "text": piece,
            "source": document.get("source", ""),
        })
        if end == len(words):
            break
    return chunks


def chunk_documents(
    documents: Iterable[dict],
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    out: list[dict] = []
    for doc in documents:
        out.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return out


if __name__ == "__main__":  # pragma: no cover
    from .data_loader import load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    if chunks:
        print("First chunk preview:")
        print(chunks[0]["chunk_id"], chunks[0]["language"], chunks[0]["text"][:120], "...")
