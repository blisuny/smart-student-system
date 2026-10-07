"""Detect whether a piece of text is Turkish (`tr`) or English (`en`).

Strategy:
  1. Cheap heuristic on Turkish-only characters / common Turkish stopwords.
     This is robust for short student questions and avoids loading langdetect.
  2. Fall back to `langdetect` if available and the heuristic is uncertain.
  3. Fall back to the configured default language if nothing matches.
"""
from __future__ import annotations

import re

from .config import DEFAULT_LANGUAGE

_TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
_TURKISH_STOPWORDS = {
    "ve", "bir", "için", "ile", "ne", "nasıl", "nedir", "kaç", "hangi", "var",
    "mı", "mi", "mu", "mü", "ama", "şu", "bu", "şöyle", "böyle", "değil",
    "olur", "olmaz", "olarak", "üzerine", "üzerinden", "üzerinde", "kadar",
    "ders", "ders kaydı", "öğrenci", "üniversite", "kütüphane", "fakülte",
}

_ENGLISH_STOPWORDS = {
    "the", "and", "is", "are", "was", "were", "do", "does", "did", "what",
    "how", "when", "where", "which", "who", "why", "of", "to", "in", "on",
    "at", "for", "a", "an", "this", "that", "these", "those", "with", "from",
    "student", "university", "library", "course", "exam", "registration",
}

_WORD_RE = re.compile(r"[\wıİğĞüÜşŞçÇöÖ]+", re.UNICODE)


def _tokenise(text: str) -> list[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text)]


def detect_language(text: str) -> str:
    """Return ``"tr"`` for Turkish, ``"en"`` for English.

    On uncertainty, returns ``DEFAULT_LANGUAGE`` (English by default).
    """
    if not text or not text.strip():
        return DEFAULT_LANGUAGE

    if any(ch in _TURKISH_CHARS for ch in text):
        return "tr"

    tokens = set(_tokenise(text))
    tr_hits = len(tokens & _TURKISH_STOPWORDS)
    en_hits = len(tokens & _ENGLISH_STOPWORDS)
    if tr_hits > en_hits and tr_hits > 0:
        return "tr"
    if en_hits > tr_hits and en_hits > 0:
        return "en"

    try:
        from langdetect import detect, DetectorFactory  # type: ignore

        DetectorFactory.seed = 0
        guess = detect(text)
        if guess == "tr":
            return "tr"
        if guess == "en":
            return "en"
    except Exception:
        pass

    return DEFAULT_LANGUAGE


if __name__ == "__main__":  # pragma: no cover
    samples = [
        "Mezuniyet projesi raporunda hangi bölümler olmalı?",
        "What sections should a graduation project report include?",
        "Kütüphane saat kaçta kapanıyor?",
        "Where is the cafeteria?",
    ]
    for s in samples:
        print(f"{detect_language(s)} -> {s}")
