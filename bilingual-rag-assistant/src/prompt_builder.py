"""Build bilingual RAG prompts.

The output is a single string (or chat-message list) that:

1. Tells the model to answer in the same language as the question.
2. Lists numbered context items.
3. Forbids inventing facts not in the context.
4. Demands a single direct answer (no headings or extra sections).

Two helpers are exposed: ``build_prompt`` (string for plain text models)
and ``build_chat_messages`` (list of dicts for chat-template models such as
Qwen).
"""
from __future__ import annotations

from typing import Iterable

# --- Localised header strings -------------------------------------------
_INSUFFICIENT = {
    "en": (
        "I could not find this information in the dataset. For up-to-date and "
        "official details, please check the Üsküdar University website "
        "(https://uskudar.edu.tr) or contact the relevant department or the "
        "Student Affairs Office."
    ),
    "tr": (
        "Bu bilgiyi veri setinde bulamadım. Güncel ve resmi bilgi için lütfen "
        "Üsküdar Üniversitesi web sitesini (https://uskudar.edu.tr) ziyaret edin "
        "veya ilgili bölüm ya da Öğrenci İşleri ile iletişime geçin."
    ),
}

_SYSTEM_TEMPLATES = {
    "en": (
        "You are an Üsküdar University student-guidance assistant. "
        "Read the user's question carefully and ANSWER IT directly using only the "
        "numbered context. Do NOT paste chunks verbatim; synthesize the relevant facts "
        "into a real answer. If the context does not contain the answer, write exactly: "
        "\"{insufficient}\" then stop.\n"
        "Answer ONLY in English. Reply with a single direct answer in 1–4 sentences. "
        "Do NOT use markdown headings, do NOT add a reasoning section, a next-step section, "
        "or a confidence rating. Output only the answer text itself."
    ),
    "tr": (
        "Sen Üsküdar Üniversitesi öğrenci rehber asistanısın. "
        "Kullanıcının sorusunu dikkatle oku ve SADECE numaralı bağlamı kullanarak "
        "soruya DOĞRUDAN cevap ver. Bağlam parçalarını olduğu gibi yapıştırma; ilgili "
        "bilgileri gerçek bir cevaba dönüştür. Bağlam soruyu cevaplamıyorsa şunu yaz: "
        "\"{insufficient}\" ve dur.\n"
        "SADECE Türkçe yanıt ver. Tek ve doğrudan bir cevap ver (1–4 cümle). "
        "Markdown başlığı KULLANMA, gerekçe bölümü, sonraki adım bölümü veya güven "
        "değerlendirmesi EKLEME. Yalnızca cevap metninin kendisini yaz."
    ),
}


def _normalise_lang(lang: str) -> str:
    return "tr" if str(lang).lower().startswith("tr") else "en"


def _format_context(retrieved_chunks: Iterable[dict]) -> str:
    lines: list[str] = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        title = chunk.get("title") or "(untitled)"
        source = chunk.get("source") or "unknown source"
        text = chunk.get("text") or ""
        lines.append(
            f"[{i}] {title}  —  source: {source}\n{text.strip()}"
        )
    return "\n\n".join(lines) if lines else "(no relevant context retrieved)"


def _system_prompt(language: str) -> str:
    lang = _normalise_lang(language)
    return _SYSTEM_TEMPLATES[lang].format(insufficient=_INSUFFICIENT[lang])


def build_prompt(
    question: str,
    retrieved_chunks: list[dict],
    language: str,
) -> str:
    """Return a single prompt string (system + context + question)."""
    lang = _normalise_lang(language)
    system = _system_prompt(lang)
    context_block = _format_context(retrieved_chunks)
    question_label = "Soru" if lang == "tr" else "Question"
    answer_label = "Cevap" if lang == "tr" else "Answer"
    return (
        f"{system}\n\n"
        f"=== Context ===\n{context_block}\n\n"
        f"=== {question_label} ===\n{question.strip()}\n\n"
        f"=== {answer_label} ===\n"
    )


def build_chat_messages(
    question: str,
    retrieved_chunks: list[dict],
    language: str,
) -> list[dict]:
    """Chat-format equivalent of ``build_prompt`` for chat-template models."""
    lang = _normalise_lang(language)
    context_block = _format_context(retrieved_chunks)
    user_label = "Soru" if lang == "tr" else "Question"
    user_content = (
        f"=== Context ===\n{context_block}\n\n"
        f"=== {user_label} ===\n{question.strip()}"
    )
    return [
        {"role": "system", "content": _system_prompt(lang)},
        {"role": "user", "content": user_content},
    ]
