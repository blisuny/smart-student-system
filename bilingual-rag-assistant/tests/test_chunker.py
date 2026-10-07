"""Tests for the word-based chunker."""
import pytest

from src.chunker import chunk_document, chunk_documents


def _doc(text: str, doc_id: str = "doc1", language: str = "en") -> dict:
    return {
        "id": doc_id,
        "language": language,
        "title": "Test doc",
        "text": text,
        "source": "test",
    }


def test_short_doc_produces_single_chunk():
    chunks = chunk_document(_doc("hello world from test"), chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world from test"
    assert chunks[0]["chunk_id"].startswith("doc1__c")
    assert chunks[0]["document_id"] == "doc1"
    assert chunks[0]["language"] == "en"
    assert chunks[0]["source"] == "test"


def test_long_doc_splits_with_overlap():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_document(_doc(" ".join(words)), chunk_size=10, overlap=2)
    # step = 10 - 2 = 8 → starts at 0, 8, 16; loop breaks when end hits 25.
    assert len(chunks) == 3
    assert chunks[0]["text"].split()[:3] == ["w0", "w1", "w2"]
    assert chunks[1]["text"].split()[0] == "w8"
    # Overlap: last 2 words of chunk 0 == first 2 words of chunk 1
    chunk0_tail = chunks[0]["text"].split()[-2:]
    chunk1_head = chunks[1]["text"].split()[:2]
    assert chunk0_tail == chunk1_head
    # Last chunk ends at the final word.
    assert chunks[-1]["text"].split()[-1] == "w24"


def test_metadata_is_preserved():
    chunks = chunk_document(_doc("a b c d e f g h", "abc", "tr"), chunk_size=4, overlap=1)
    for c in chunks:
        assert c["document_id"] == "abc"
        assert c["language"] == "tr"
        assert c["title"] == "Test doc"
        assert c["source"] == "test"


def test_empty_text_yields_no_chunks():
    chunks = chunk_document(_doc("   "), chunk_size=5, overlap=1)
    assert chunks == []


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_document(_doc("a b c"), chunk_size=2, overlap=2)
    with pytest.raises(ValueError):
        chunk_document(_doc("a b c"), chunk_size=2, overlap=-1)


def test_chunk_documents_concatenates():
    docs = [_doc("a b c d", "d1"), _doc("e f g", "d2")]
    chunks = chunk_documents(docs, chunk_size=3, overlap=1)
    doc_ids = {c["document_id"] for c in chunks}
    assert doc_ids == {"d1", "d2"}
