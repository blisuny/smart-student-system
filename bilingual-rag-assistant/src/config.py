"""Central configuration for the bilingual RAG assistant.

All paths are derived from PROJECT_ROOT so the project is portable.
Models and hyperparameters can be changed in one place.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

INDEX_PATH = PROCESSED_DIR / "faiss.index"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"

# --- Models --------------------------------------------------------------
# To swap the embedding model, change this string and rebuild the index.
EMBEDDING_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v1"

# --- LLM backend ----------------------------------------------------------
# "hf"   = run a local Hugging Face model (free, slow, small).
# "groq" = call the Groq hosted API (free tier, fast, large model).
#          Requires GROQ_API_KEY env var. See src/groq_generator.py.
LLM_BACKEND = "groq"

# Local HF model used when LLM_BACKEND == "hf".
GENERATION_MODEL_NAME = "Qwen/Qwen3-0.6B"

# Groq model used when LLM_BACKEND == "groq".
# llama-3.3-70b-versatile is the strongest free-tier model with great Turkish.
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 80

# --- Retrieval -----------------------------------------------------------
TOP_K = 3

# --- Generation ----------------------------------------------------------
MAX_NEW_TOKENS = 160
TEMPERATURE = 0.2
TOP_P = 0.9

# --- Misc ----------------------------------------------------------------
DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ("tr", "en")
