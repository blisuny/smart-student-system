"""Load student-knowledge documents from ``data/raw/``.

Each file in ``data/raw/`` is a JSON list of records with the schema::

    {
        "id":       str,           # globally unique
        "language": "tr" | "en",
        "title":    str,
        "text":     str,
        "source":   str            # short tag, e.g. "qa_main", "uskudar.edu.tr"
    }

The loader is intentionally simple: it walks the directory, parses every
``*.json`` file, validates each record, and returns one flat list.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import RAW_DIR, SUPPORTED_LANGUAGES

REQUIRED_FIELDS = ("id", "language", "title", "text", "source")


class InvalidDocumentError(ValueError):
    """Raised when a document does not satisfy the required schema."""


def _validate_document(doc: dict, file_path: Path, index: int) -> None:
    if not isinstance(doc, dict):
        raise InvalidDocumentError(
            f"{file_path.name} record #{index}: not an object"
        )
    missing = [f for f in REQUIRED_FIELDS if f not in doc]
    if missing:
        raise InvalidDocumentError(
            f"{file_path.name} record #{index}: missing fields {missing}"
        )
    if doc["language"] not in SUPPORTED_LANGUAGES:
        raise InvalidDocumentError(
            f"{file_path.name} record #{index}: language "
            f"{doc['language']!r} not in {SUPPORTED_LANGUAGES}"
        )
    if not isinstance(doc["text"], str) or not doc["text"].strip():
        raise InvalidDocumentError(
            f"{file_path.name} record #{index}: empty text"
        )


def load_documents(raw_dir: Path | None = None) -> list[dict]:
    """Read every ``*.json`` file under ``raw_dir`` and return all records.

    Parameters
    ----------
    raw_dir:
        Directory holding raw JSON files. Defaults to ``config.RAW_DIR``.
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    documents: list[dict] = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise InvalidDocumentError(f"{path.name}: {exc}") from exc

        if not isinstance(payload, list):
            raise InvalidDocumentError(
                f"{path.name}: top-level JSON must be a list of records"
            )

        for i, doc in enumerate(payload):
            _validate_document(doc, path, i)
            documents.append(dict(doc))

    return documents


def iter_documents(raw_dir: Path | None = None) -> Iterable[dict]:
    """Generator variant for very large corpora."""
    yield from load_documents(raw_dir)


if __name__ == "__main__":  # pragma: no cover
    docs = load_documents()
    by_lang: dict[str, int] = {}
    for d in docs:
        by_lang[d["language"]] = by_lang.get(d["language"], 0) + 1
    print(f"Loaded {len(docs)} documents from {RAW_DIR}")
    for lang, count in sorted(by_lang.items()):
        print(f"  {lang}: {count}")
