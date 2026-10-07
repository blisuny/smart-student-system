"""One-off converter for the legacy ``USKUDAR_UNIVERSITY_DATASET.json``.

Produces JSON files in ``data/raw/`` matching the schema expected by
``data_loader.load_documents``: ``{id, language, title, text, source}``.

Usage
-----
    python -m src.convert_legacy_data \
        --input "../Graduation Proj/USKUDAR_UNIVERSITY_DATASET.json"

If ``--input`` is omitted, the script tries the default path
``../Graduation Proj/USKUDAR_UNIVERSITY_DATASET.json`` relative to the
project root (i.e. the legacy folder placed alongside this project).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .config import RAW_DIR
from .language_detector import detect_language

DEFAULT_LEGACY_PATH = Path("..") / "Graduation Proj" / "USKUDAR_UNIVERSITY_DATASET.json"


def _qa_pair_to_doc(prefix: str, source: str, default_lang: str, item: dict) -> dict:
    qid = item.get("id")
    question = (item.get("question") or "").strip()
    answer = (item.get("answer") or "").strip()
    title = item.get("topic") or item.get("category") or question[:80]
    extra = []
    for key in ("category", "subcategory", "topic", "information"):
        val = item.get(key)
        if val:
            extra.append(f"{key}: {val}")
    body = "\n".join(filter(None, [
        f"Soru / Question: {question}" if question else "",
        f"Cevap / Answer: {answer}" if answer else "",
        " | ".join(extra) if extra else "",
    ]))
    lang = detect_language(question or answer) or default_lang
    return {
        "id": f"{prefix}_{qid}",
        "language": lang,
        "title": title or f"{prefix} #{qid}",
        "text": body.strip(),
        "source": source,
    }


def _bilingual_announcement_to_docs(item: dict) -> list[dict]:
    url = item.get("url", "")
    docs: list[dict] = []
    slug = re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_")[-60:] or "announcement"
    if item.get("text_tr"):
        docs.append({
            "id": f"ann_bi_tr_{slug}",
            "language": "tr",
            "title": item.get("title_tr") or item.get("title_en") or "Announcement",
            "text": item["text_tr"].strip(),
            "source": url or "uskudar.edu.tr/announcements",
        })
    if item.get("text_en"):
        text_en = item["text_en"].strip()
        lang = detect_language(text_en[:400])
        docs.append({
            "id": f"ann_bi_{lang}_{slug}_en",
            "language": lang,
            "title": item.get("title_en") or item.get("title_tr") or "Announcement",
            "text": text_en,
            "source": url or "uskudar.edu.tr/announcements",
        })
    return docs


def _scraped_announcement_to_doc(item: dict, idx: int) -> dict | None:
    text = (item.get("clean_text") or item.get("text") or "").strip()
    if not text:
        return None
    title = item.get("title") or "Announcement"
    return {
        "id": f"ann_scraped_{idx:04d}",
        "language": detect_language(text[:400]),
        "title": title,
        "text": text,
        "source": item.get("url") or item.get("source") or "uskudar.edu.tr",
    }


def _professor_records_to_docs(records: list[dict]) -> list[dict]:
    """Group schedule rows by professor and emit one document per professor."""
    by_prof: dict[str, list[dict]] = {}
    for row in records:
        key = f"{row.get('professor', 'unknown')}|{row.get('academic_year', '')}"
        by_prof.setdefault(key, []).append(row)

    docs: list[dict] = []
    for key, rows in by_prof.items():
        first = rows[0]
        prof = first.get("professor") or "Unknown professor"
        dept = first.get("department") or ""
        year = first.get("academic_year") or ""

        lines: list[str] = [
            f"Professor: {prof}",
            f"Department: {dept}",
            f"Academic year: {year}",
            "",
        ]
        for row in rows:
            block = row.get("block_type") or ""
            day = row.get("day") or ""
            time_ = row.get("time") or ""
            mode = row.get("mode") or ""
            activity = row.get("activity") or ""
            notes = row.get("notes") or ""
            descriptors = " | ".join(
                bit for bit in [day, time_, mode, activity, notes]
                if bit and bit != "nan"
            )
            lines.append(f"- {block}: {descriptors}")

        text = "\n".join(lines).strip()
        slug = re.sub(r"[^a-z0-9]+", "_", prof.lower()).strip("_")[:60]
        docs.append({
            "id": f"prof_{slug}",
            "language": "en",
            "title": f"{prof} — schedule ({year})",
            "text": text,
            "source": "professor_schedule",
        })
    return docs


def _write(out_dir: Path, name: str, docs: list[dict]) -> Path:
    out = out_dir / name
    out.write_text(
        json.dumps(docs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def convert(legacy_path: Path, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    sections = payload.get("sections", {})
    summary: dict[str, int] = {}

    if "qa_main" in sections:
        docs = [
            _qa_pair_to_doc("qa_main", "uskudar.edu.tr", "tr", item)
            for item in sections["qa_main"]["data"]
        ]
        summary["qa_main"] = len(docs)
        _write(out_dir, "uskudar_qa_main.json", docs)

    if "qa_json" in sections:
        docs = [
            _qa_pair_to_doc("qa_json", "uskudar.edu.tr", "tr", item)
            for item in sections["qa_json"]["data"]
        ]
        summary["qa_json"] = len(docs)
        _write(out_dir, "uskudar_qa_json.json", docs)

    if "qa_student" in sections:
        docs = [
            _qa_pair_to_doc("qa_student", "student_campus", "tr", item)
            for item in sections["qa_student"]["data"]
        ]
        summary["qa_student"] = len(docs)
        _write(out_dir, "uskudar_qa_student.json", docs)

    if "qa_university" in sections:
        docs = [
            _qa_pair_to_doc("qa_university", "obs", "tr", item)
            for item in sections["qa_university"]["data"]
        ]
        summary["qa_university"] = len(docs)
        _write(out_dir, "uskudar_qa_university.json", docs)

    if "announcements_bilingual" in sections:
        all_docs: list[dict] = []
        for item in sections["announcements_bilingual"]["data"]:
            all_docs.extend(_bilingual_announcement_to_docs(item))
        summary["announcements_bilingual"] = len(all_docs)
        _write(out_dir, "uskudar_announcements_bilingual.json", all_docs)

    if "announcements_scraped" in sections:
        docs = []
        for idx, item in enumerate(sections["announcements_scraped"]["data"]):
            doc = _scraped_announcement_to_doc(item, idx)
            if doc is not None:
                docs.append(doc)
        summary["announcements_scraped"] = len(docs)
        _write(out_dir, "uskudar_announcements_scraped.json", docs)

    if "professor_schedule" in sections:
        docs = _professor_records_to_docs(sections["professor_schedule"]["data"])
        summary["professor_schedule"] = len(docs)
        _write(out_dir, "uskudar_professor_schedule.json", docs)

    return summary


def _resolve_legacy_path(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    candidate = (RAW_DIR.parent.parent / DEFAULT_LEGACY_PATH).resolve()
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to USKUDAR_UNIVERSITY_DATASET.json")
    parser.add_argument(
        "--output",
        default=str(RAW_DIR),
        help="Directory to write converted JSON files",
    )
    args = parser.parse_args()

    legacy_path = _resolve_legacy_path(args.input)
    out_dir = Path(args.output).expanduser().resolve()
    if not legacy_path.exists():
        raise SystemExit(
            f"Legacy dataset not found: {legacy_path}\n"
            "Pass --input <path> if the file is somewhere else."
        )

    print(f"Reading legacy dataset: {legacy_path}")
    summary = convert(legacy_path, out_dir)
    print(f"Wrote converted files to: {out_dir}")
    for name, count in summary.items():
        print(f"  {name}: {count} records")


if __name__ == "__main__":  # pragma: no cover
    main()
