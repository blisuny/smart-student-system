"""Hybrid retriever: FAISS embedding search + IDF-weighted lexical fallback.

Why hybrid: the multilingual sentence-transformer is weak at matching
proper nouns (e.g. "Rowanda") because the name is mostly OOV. A pure
dense search ranks fluent-but-irrelevant chunks above the right one.

Why IDF-weighted lexical: a simple "+0.3 per matching token" lexical
boost over-rewards common words. The query "rowanda kimdir" matches
"kimdir" in ~200 chunks but "rowanda" in only ~2 — IDF weighting makes
the rare token dominate, which is exactly what we want for name lookups.
"""
from __future__ import annotations

import math
import re

from .config import TOP_K
from .embeddings import EmbeddingModel
from .vector_store import VectorStore

_WORD_RE = re.compile(r"[A-Za-zÇÖŞİĞÜçöşığü0-9]+", re.UNICODE)

_STOPWORDS = {
    # English
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "those", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "but", "not", "do", "does", "did", "have", "has", "had",
    "what", "where", "when", "who", "why", "how", "which", "any", "all",
    "there", "their", "his", "her", "its", "my", "your",
    # Turkish
    "bir", "bi", "iki", "var", "yok", "mi", "mı", "mu", "mü", "ne", "neyi",
    "ait", "icin", "için", "ile", "ya", "ve", "veya", "ama", "fakat",
    "ben", "sen", "biz", "siz", "onlar", "bu", "şu", "su", "o",
    "da", "de", "ki", "gibi", "kadar", "olarak",
}

# Scale factor: a single match on a very rare token (IDF ≈ log(N) ≈ 7)
# becomes ~0.7, comparable to a strong dense cosine score.
_LEXICAL_SCALE = 0.10


def _query_tokens(question: str) -> set[str]:
    tokens = _WORD_RE.findall(question)
    return {t.lower() for t in tokens if len(t) > 2 and t.lower() not in _STOPWORDS}


def _chunk_tokens(chunk: dict) -> set[str]:
    haystack = (chunk.get("title") or "") + " " + (chunk.get("text") or "")
    return {t.lower() for t in _WORD_RE.findall(haystack)}


class Retriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self.store = store or VectorStore.load()
        self.embedder = embedder or EmbeddingModel()
        self._token_index, self._idf = self._build_lexical_index()

    def _build_lexical_index(self) -> tuple[list[set[str]], dict[str, float]]:
        """Pre-tokenize every chunk once and build an IDF table over the corpus."""
        token_index: list[set[str]] = []
        df: dict[str, int] = {}
        for chunk in self.store.chunks:
            toks = _chunk_tokens(chunk)
            token_index.append(toks)
            for t in toks:
                df[t] = df.get(t, 0) + 1
        n = max(len(self.store.chunks), 1)
        # +1 smoothing so a token in every chunk doesn't get IDF=0.
        idf = {t: math.log((n + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}
        return token_index, idf

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        # 1. Dense search, over-fetch so the lexical re-rank has room.
        query_vec = self.embedder.encode(question)
        over_k = max(top_k * 4, 20)
        dense = self.store.search(query_vec, top_k=over_k)

        q_tokens = _query_tokens(question)
        if not q_tokens:
            return dense[:top_k]

        # 2. Lexical pass: sum IDF of query tokens present in each chunk.
        #    Rare tokens (e.g. proper nouns) dominate; common ones contribute little.
        lexical: dict[str, tuple[float, dict]] = {}
        for chunk, toks in zip(self.store.chunks, self._token_index):
            score = sum(self._idf.get(tok, 0.0) for tok in q_tokens if tok in toks)
            if score > 0:
                lexical[chunk["chunk_id"]] = (score, chunk)

        # 3. Merge: dense score + scaled lexical IDF; include lexical-only hits.
        merged: dict[str, dict] = {}
        for chunk in dense:
            cid = chunk["chunk_id"]
            idf_score = lexical.get(cid, (0.0, None))[0]
            new_chunk = dict(chunk)
            new_chunk["score"] = float(chunk["score"]) + _LEXICAL_SCALE * idf_score
            merged[cid] = new_chunk

        for cid, (idf_score, chunk) in lexical.items():
            if cid in merged:
                continue
            new_chunk = dict(chunk)
            new_chunk["score"] = _LEXICAL_SCALE * idf_score
            merged[cid] = new_chunk

        ranked = sorted(merged.values(), key=lambda c: c["score"], reverse=True)
        return ranked[:top_k]
