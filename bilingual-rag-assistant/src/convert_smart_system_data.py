"""Converter for the SmartSystemStudent Backend/Data JSON files.

Reads the 5 files under
``SmartSystemStudent final/SmartSystemStudent/Backend/Data/``
(Professors, clubs, erasmus, events, research) and writes them into
``data/raw/`` matching the RAG schema ``{id, language, title, text, source}``.

Usage
-----
    python -m src.convert_smart_system_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RAW_DIR
from .language_detector import detect_language

DEFAULT_INPUT_DIR = Path("SmartSystemStudent final") / "SmartSystemStudent" / "Backend" / "Data"


def _clean(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v)
    s = str(val).strip()
    return "" if s.lower() in {"not available", "n/a", "none", ""} else s


def _join_lines(parts: list[str]) -> str:
    return "\n".join(p for p in parts if p)


def _keywords_line(item: dict) -> str:
    kws = item.get("keywords") or []
    return f"Keywords: {', '.join(kws)}" if kws else ""


def professors_to_docs(items: list[dict]) -> list[dict]:
    docs = []
    for it in items:
        name = _clean(it.get("name"))
        title = _clean(it.get("title"))
        faculty = _clean(it.get("faculty"))
        department = _clean(it.get("department"))
        email = _clean(it.get("email"))
        office = _clean(it.get("office"))
        hours = _clean(it.get("office_hours"))
        url = _clean(it.get("profile_url"))
        desc = _clean(it.get("description"))

        full_title = " ".join(filter(None, [title, name])) or f"Professor {it.get('id')}"
        text = _join_lines([
            f"{title} {name}".strip() if name else "",
            f"Faculty: {faculty}" if faculty else "",
            f"Department: {department}" if department else "",
            f"Office: {office}" if office else "",
            f"Office hours: {hours}" if hours else "",
            f"Email: {email}" if email else "",
            f"Profile: {url}" if url else "",
            desc,
            _keywords_line(it),
        ])
        lang = detect_language(text) or "en"
        docs.append({
            "id": f"prof_{it.get('id')}",
            "language": lang,
            "title": full_title,
            "text": text,
            "source": "uskudar.edu.tr/akademik-personel",
        })
    return docs


def clubs_to_docs(items: list[dict]) -> list[dict]:
    docs = []
    for it in items:
        name = _clean(it.get("name"))
        category = _clean(it.get("category"))
        desc = _clean(it.get("description"))
        activities = _clean(it.get("activities"))
        contact = _clean(it.get("contact"))

        text = _join_lines([
            desc,
            f"Category: {category}" if category else "",
            f"Activities: {activities}" if activities else "",
            f"Contact: {contact}" if contact else "",
            _keywords_line(it),
        ])
        lang = detect_language(text) or "en"
        docs.append({
            "id": f"club_{it.get('id')}",
            "language": lang,
            "title": name or f"Club {it.get('id')}",
            "text": text,
            "source": "sks.uskudar.edu.tr/student-clubs",
        })
    return docs


def erasmus_to_docs(items: list[dict]) -> list[dict]:
    docs = []
    for it in items:
        title = _clean(it.get("title"))
        category = _clean(it.get("category"))
        desc = _clean(it.get("description"))
        extra_parts = []
        for key, val in it.items():
            if key in {"id", "title", "category", "description", "keywords"}:
                continue
            cleaned = _clean(val)
            if cleaned:
                extra_parts.append(f"{key.replace('_', ' ').title()}: {cleaned}")

        text = _join_lines([
            desc,
            f"Category: {category}" if category else "",
            *extra_parts,
            _keywords_line(it),
        ])
        lang = detect_language(text) or "tr"
        docs.append({
            "id": f"erasmus_{it.get('id')}",
            "language": lang,
            "title": title or f"Erasmus {it.get('id')}",
            "text": text,
            "source": "uskudar.edu.tr/erasmus",
        })
    return docs


def events_to_docs(items: list[dict]) -> list[dict]:
    docs = []
    for it in items:
        title = _clean(it.get("title"))
        date = _clean(it.get("date"))
        location = _clean(it.get("location"))
        category = _clean(it.get("category"))
        desc = _clean(it.get("description"))
        url = _clean(it.get("source_url"))

        text = _join_lines([
            desc,
            f"Date: {date}" if date else "",
            f"Location: {location}" if location else "",
            f"Category: {category}" if category else "",
            f"More info: {url}" if url else "",
            _keywords_line(it),
        ])
        lang = detect_language(text) or "tr"
        docs.append({
            "id": f"event_{it.get('id')}",
            "language": lang,
            "title": title or f"Event {it.get('id')}",
            "text": text,
            "source": "uskudar.edu.tr/etkinlik",
        })
    return docs


def research_to_docs(items: list[dict]) -> list[dict]:
    docs = []
    for it in items:
        title = _clean(it.get("title"))
        category = _clean(it.get("category"))
        desc = _clean(it.get("description"))
        url = _clean(it.get("source_url"))

        text = _join_lines([
            desc,
            f"Category: {category}" if category else "",
            f"More info: {url}" if url else "",
            _keywords_line(it),
        ])
        lang = detect_language(text) or "en"
        docs.append({
            "id": f"research_{it.get('id')}",
            "language": lang,
            "title": title or f"Research {it.get('id')}",
            "text": text,
            "source": "uskudar.edu.tr/research",
        })
    return docs


CONVERTERS = {
    "Professors.json":  ("uskudar_professors.json",  professors_to_docs),
    "clubs.json":       ("uskudar_clubs.json",       clubs_to_docs),
    "erasmus.json":     ("uskudar_erasmus.json",     erasmus_to_docs),
    "events.json":      ("uskudar_events.json",      events_to_docs),
    "research.json":    ("uskudar_research.json",    research_to_docs),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing the 5 SmartSystemStudent JSON files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for src_name, (dst_name, fn) in CONVERTERS.items():
        src_path = input_dir / src_name
        if not src_path.exists():
            print(f"  skip (missing): {src_path}")
            continue
        items = json.loads(src_path.read_text(encoding="utf-8"))
        docs = fn(items)
        dst_path = RAW_DIR / dst_name
        dst_path.write_text(
            json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  {src_name:22s} -> {dst_path.name}  ({len(docs)} docs)")
        total += len(docs)

    print(f"\nWrote {total} documents into {RAW_DIR}")


if __name__ == "__main__":
    main()
