"""Wrapper around a sentence-transformers embedding model.

The wrapper is intentionally thin: it loads the model lazily, normalises
embeddings (so cosine similarity = inner product), and exposes a single
``encode`` method that handles single strings or lists of strings.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        # ``sentence_transformers`` is a heavy dependency; import lazily so
        # the rest of the pipeline (e.g. tests) can be imported without it.
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.model = SentenceTransformer(self.model_name)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: str | Iterable[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype="float32")
