"""Tests for the language detector."""
from src.language_detector import detect_language


def test_turkish_with_special_chars():
    assert detect_language("Mezuniyet projesi raporunda hangi bölümler olmalı?") == "tr"


def test_turkish_without_special_chars():
    # No accented characters but Turkish stopwords ("hangi", "var") present.
    assert detect_language("Hangi binada cafe var") == "tr"


def test_english_question():
    assert detect_language(
        "What sections should a graduation project report include?"
    ) == "en"


def test_short_english_phrase():
    assert detect_language("Where is the cafeteria?") == "en"


def test_empty_falls_back_to_default():
    # Empty input should not crash and should return a supported language.
    assert detect_language("") in {"tr", "en"}


def test_ambiguous_falls_back_to_default():
    # A single non-language word should not blow up.
    assert detect_language("xyz") in {"tr", "en"}
