"""5-fold evaluation for the bilingual RAG pipeline.

Reads ``data/evaluation/questions.json`` (a list of evaluation examples),
splits them into 5 folds, runs the RAG pipeline on each fold, and writes
predictions plus simple metrics to ``data/evaluation/results.json``.

Each example must look like::

    {
      "id": "q_tr_001",
      "question": "...",
      "language": "tr",
      "expected_answer": "comma,separated,keywords",
      "expected_source": "project_rules"
    }

Metrics computed per example
----------------------------
* ``language_match``                 — detected language equals expected
* ``source_retrieved``               — expected source seen in retrieved chunks
* ``answer_contains_expected_keywords`` — fraction of expected keywords found
                                          in the generated answer (0.0–1.0)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .config import EVAL_DIR, TOP_K
from .language_detector import detect_language
from .rag_pipeline import RAGPipeline

QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_PATH = EVAL_DIR / "results.json"


@dataclass
class FoldResult:
    fold: int
    metrics: dict
    predictions: list[dict]


def _load_examples(path: Path = QUESTIONS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}. "
            "Create one by following the schema in src/evaluation.py docstring."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _split_into_folds(examples: list[dict], n_folds: int = 5) -> list[list[dict]]:
    if not examples:
        return [[] for _ in range(n_folds)]
    fold_size = math.ceil(len(examples) / n_folds)
    return [examples[i:i + fold_size] for i in range(0, len(examples), fold_size)]


def _keyword_recall(answer: str, expected_csv: str) -> float:
    keywords = [k.strip().lower() for k in expected_csv.split(",") if k.strip()]
    if not keywords:
        return 0.0
    answer_lower = (answer or "").lower()
    hits = sum(1 for k in keywords if k in answer_lower)
    return hits / len(keywords)


def _evaluate_fold(
    fold: int,
    examples: list[dict],
    pipeline: RAGPipeline,
    top_k: int,
) -> FoldResult:
    predictions: list[dict] = []
    lang_hits = src_hits = 0
    keyword_scores: list[float] = []

    for example in examples:
        result = pipeline.answer(example["question"], top_k=top_k)
        detected = result.detected_language
        retrieved_sources = {c.get("source", "") for c in result.retrieved_chunks}
        expected_source = example.get("expected_source", "")
        source_hit = bool(expected_source) and expected_source in retrieved_sources
        lang_match = detected == example.get("language")
        kw_score = _keyword_recall(result.answer, example.get("expected_answer", ""))

        lang_hits += int(lang_match)
        src_hits += int(source_hit)
        keyword_scores.append(kw_score)

        predictions.append({
            "id": example.get("id"),
            "question": example["question"],
            "expected_language": example.get("language"),
            "detected_language": detected,
            "expected_source": expected_source,
            "retrieved_sources": sorted(retrieved_sources),
            "language_match": lang_match,
            "source_retrieved": source_hit,
            "answer_contains_expected_keywords": kw_score,
            "answer": result.answer,
        })

    n = max(1, len(examples))
    metrics = {
        "n_examples": len(examples),
        "language_match": lang_hits / n,
        "source_retrieved": src_hits / n,
        "answer_contains_expected_keywords":
            sum(keyword_scores) / n if keyword_scores else 0.0,
    }
    return FoldResult(fold=fold, metrics=metrics, predictions=predictions)


def run_evaluation(
    questions_path: Path = QUESTIONS_PATH,
    results_path: Path = RESULTS_PATH,
    top_k: int = TOP_K,
    n_folds: int = 5,
) -> dict:
    examples = _load_examples(questions_path)
    folds = _split_into_folds(examples, n_folds=n_folds)

    pipeline = RAGPipeline()
    fold_results: list[FoldResult] = []
    for i, fold_examples in enumerate(folds, start=1):
        print(f"--- Fold {i}/{n_folds} ({len(fold_examples)} examples) ---")
        fold_results.append(_evaluate_fold(i, fold_examples, pipeline, top_k))

    averaged = _average_metrics([fr.metrics for fr in fold_results])
    payload = {
        "n_folds": n_folds,
        "top_k": top_k,
        "average": averaged,
        "folds": [
            {
                "fold": fr.fold,
                "metrics": fr.metrics,
                "predictions": fr.predictions,
            }
            for fr in fold_results
        ],
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote results to {results_path}")
    print("Average metrics across folds:")
    for k, v in averaged.items():
        print(f"  {k}: {v:.3f}")
    return payload


def _average_metrics(metrics_per_fold: list[dict]) -> dict:
    if not metrics_per_fold:
        return {}
    keys = [k for k in metrics_per_fold[0].keys() if k != "n_examples"]
    averaged: dict = {}
    for k in keys:
        averaged[k] = sum(m[k] for m in metrics_per_fold) / len(metrics_per_fold)
    return averaged


if __name__ == "__main__":  # pragma: no cover
    run_evaluation()
