"""FastAPI backend for the bilingual RAG assistant.

Run with:
    uvicorn app.api:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import TOP_K
from src.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Bilingual TR/EN RAG Student Assistant",
    description=(
        "Retrieval-Augmented Generation over a small bilingual dataset of "
        "Üsküdar University student-guidance content."
    ),
    version="0.1.0",
)

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(..., description="Student question in Turkish or English.")
    top_k: int = Field(TOP_K, ge=1, le=20)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    language: str
    title: str
    text: str
    source: str
    score: float


class AskResponse(BaseModel):
    question: str
    detected_language: str
    retrieved_chunks: list[Chunk]
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    pipeline = _get_pipeline()
    result = pipeline.answer(req.question, top_k=req.top_k)
    return AskResponse(
        question=result.question,
        detected_language=result.detected_language,
        retrieved_chunks=[Chunk(**c) for c in result.retrieved_chunks],
        answer=result.answer,
    )
