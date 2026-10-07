# Bilingual Turkish-English RAG-Based Small Language Model for Student Academic Guidance

A graduation thesis project: a small, local, bilingual (Turkish + English)
RAG chatbot that answers Üsküdar University student questions using a
custom dataset and a small Hugging Face language model.

## Problem definition

Students need quick answers about university procedures (course
registration, library hours, Erasmus, professor office hours, campus
announcements). General-purpose chatbots either hallucinate or miss the
local Üsküdar-specific context. Training a small LLM from scratch is
unrealistic at the graduation-project scale.

This project answers that problem with **Retrieval-Augmented Generation
(RAG)** over a bilingual local dataset. The system:

1. Detects the question language (Turkish or English).
2. Retrieves the most relevant chunks from the local dataset using a
   multilingual sentence-transformer + FAISS index.
3. Generates an answer with a small Hugging Face model (Qwen3-0.6B by
   default), constrained to the retrieved context, and **answers in the
   same language as the question**.
4. Refuses confidently when the context is insufficient.

## System architecture

```
                ┌──────────────────────────┐
                │ Student question (TR/EN) │
                └─────────────┬────────────┘
                              ▼
                  ┌────────────────────────┐
                  │ language_detector      │
                  └─────────────┬──────────┘
                                ▼
       ┌────────────────────────────────────────┐
       │ retriever:                             │
       │   embeddings + FAISS over data/raw/    │
       └────────────────────────┬───────────────┘
                                ▼
       ┌────────────────────────────────────────┐
       │ prompt_builder (TR or EN system msg)   │
       └────────────────────────┬───────────────┘
                                ▼
       ┌────────────────────────────────────────┐
       │ generator (HF Qwen3-0.6B by default)   │
       └────────────────────────┬───────────────┘
                                ▼
            ┌─────────────────────────────────┐
            │ Answer / Cevap                  │
            │ Reasoning / Gerekçe             │
            │ What you should do next /       │
            │     Sonraki adım                │
            │ Confidence / Güven              │
            └─────────────────────────────────┘
```

The pipeline is split into small, testable Python modules:

| Module | Role |
|---|---|
| [`src/data_loader.py`](src/data_loader.py) | Load `*.json` files from `data/raw/` |
| [`src/chunker.py`](src/chunker.py) | Word-window chunking (default 400/80) |
| [`src/embeddings.py`](src/embeddings.py) | sentence-transformers wrapper |
| [`src/vector_store.py`](src/vector_store.py) | FAISS index + side-car JSON |
| [`src/retriever.py`](src/retriever.py) | Embed → FAISS top-k |
| [`src/language_detector.py`](src/language_detector.py) | TR/EN detection |
| [`src/prompt_builder.py`](src/prompt_builder.py) | Bilingual RAG prompt |
| [`src/generator.py`](src/generator.py) | HF causal-LM generation |
| [`src/rag_pipeline.py`](src/rag_pipeline.py) | End-to-end coordinator |
| [`src/evaluation.py`](src/evaluation.py) | 5-fold evaluation |
| [`app/api.py`](app/api.py) | FastAPI backend |
| [`app/streamlit_app.py`](app/streamlit_app.py) | Streamlit demo UI |

## Why RAG (not "training a model from scratch")

* Training a Turkish/English LLM from scratch needs millions of dollars
  of compute, terabytes of curated bilingual data, and a research team.
  None of those are available at the graduation-project scale.
* Fine-tuning a mid-size LLM on a 1,400-record dataset would simply
  memorize the dataset — and break instantly when the dataset changes.
* RAG separates *knowledge* (the editable dataset under `data/raw/`)
  from *reasoning* (the small frozen LLM). Updating the chatbot means
  editing JSON files, not retraining anything.
* A small frozen model (Qwen3-0.6B) is cheap, runs on a CPU, and is
  good enough at extracting answers from short, clearly-cited context.

## Dataset format

Every file under `data/raw/*.json` is a JSON list of records:

```json
[
  {
    "id": "tr_001",
    "language": "tr",
    "title": "Mezuniyet Projesi Raporu",
    "text": "Mezuniyet projesi raporunda problem tanımı, ...",
    "source": "project_rules"
  }
]
```

| field | meaning |
|---|---|
| `id` | globally unique string |
| `language` | `"tr"` or `"en"` |
| `title` | short human-readable label, shown in the UI |
| `text` | the actual content; can be one paragraph or many |
| `source` | short tag (e.g. `obs`, `uskudar.edu.tr`) — used by the evaluator |

The repository ships with [`data/raw/example_documents.json`](data/raw/example_documents.json)
plus 7 converted Üsküdar files produced from the legacy unified dataset
by [`src/convert_legacy_data.py`](src/convert_legacy_data.py).

## Quick start (5 minutes)

> **Requires Python 3.12.** Python 3.14 does NOT have prebuilt wheels for `torch`/`faiss`
> and the install will fail. Install Python 3.12 from python.org if needed.

```powershell
# 1. Create a virtual environment using Python 3.12
py -3.12 -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the FAISS vector index (one-time, ~2 minutes — downloads embedding model)
python -m src.vector_store

# 4. (See "LLM backend" below) — set GROQ_API_KEY if using the hosted backend.

# 5. Run the chatbot
streamlit run app/streamlit_app.py
```

Streamlit opens on <http://localhost:8501>.

## LLM backend

This project supports two generation backends, switched in [`src/config.py`](src/config.py)
via the `LLM_BACKEND` setting:

### `"groq"` (default — fast, free, requires API key)

Uses Groq's hosted `llama-3.3-70b-versatile` model. Excellent Turkish, ~2-second
responses. Free tier with rate limits (~30 req/min) is plenty for a demo.

**Setup:**
1. Get a free API key at <https://console.groq.com/keys> (sign in with Google/GitHub).
2. Set the environment variable before launching Streamlit:

   ```powershell
   # Windows PowerShell (one session):
   $env:GROQ_API_KEY = "gsk_your_key_here"

   # Or set permanently for your user (no need to repeat):
   [Environment]::SetEnvironmentVariable("GROQ_API_KEY", "gsk_your_key_here", "User")
   # Then restart your terminal.
   ```
   ```bash
   # macOS / Linux:
   export GROQ_API_KEY="gsk_your_key_here"
   ```

### `"hf"` (fully local — slow on CPU)

Runs Qwen3-0.6B locally via Hugging Face transformers. No API key needed but
each answer takes ~30–60 seconds on CPU and Turkish quality is weaker.

To use it, edit [`src/config.py`](src/config.py): change `LLM_BACKEND = "groq"` to `LLM_BACKEND = "hf"`.

## Running the FastAPI backend (optional)

```bash
uvicorn app.api:app --reload --port 8000
# POST http://localhost:8000/ask  with body {"question": "...", "top_k": 5}
```

## Running tests

```bash
pytest -q
```

Tests cover: chunker, language detector, prompt builder, pipeline
(with stubbed retriever + generator), and FAISS round-trip.

## Running 5-fold evaluation

```bash
python -m src.evaluation
```

Reads [`data/evaluation/questions.json`](data/evaluation/questions.json),
splits into 5 folds, runs the full pipeline on each fold, computes per-fold
and averaged metrics (`language_match`, `source_retrieved`,
`answer_contains_expected_keywords`), and writes
`data/evaluation/results.json`.

## Example questions

**Turkish**
> Mezuniyet projesi raporunda hangi bölümler olmalı?

**English**
> What sections should a graduation project report include?

The assistant should produce something like:

```
### Answer
A graduation project report should include problem definition,
methodology, experiments, results, and evaluation sections.

### Reasoning
Context [1] (project_rules) lists exactly these five sections.

### What you should do next
Open the latest version of the project report template and confirm
that all five sections are present before submitting.

### Confidence
High
```

## Limitations

* The dataset is small (~1,400 records, mostly Turkish) and biased
  toward Üsküdar University. Out-of-domain questions get a "context
  insufficient" answer.
* The default generator (Qwen3-0.6B) is a small CPU-friendly model.
  Answer fluency improves notably if you switch to Qwen3-1.7B (or
  larger) on a GPU.
* Language detection uses a simple heuristic plus optional `langdetect`
  fallback. Code-switched questions ("Erasmus için ne yapmalıyım?")
  are routed to whichever language has the stronger signal.
* The 5-fold evaluation uses keyword-based metrics, which under-credit
  paraphrased correct answers. A small LLM-as-judge layer would be a
  natural extension.

## Future work

* **Better dataset.** More English-language records (the legacy
  bilingual announcements field is mislabeled — both halves contain
  Turkish text), and a curated FAQ from advisors.
* **LoRA fine-tuning** for *answer style* only (not for memorization),
  so the model reliably produces the four-section format.
* **Better evaluation.** Add LLM-as-judge for groundedness and
  answer-correctness, and a manual rubric for helpfulness.
* **Web deployment.** Host the FastAPI backend behind a simple Vercel
  / Render frontend.

## Repository layout

```
bilingual-rag-assistant/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── example_documents.json
│   │   └── uskudar_*.json (after running convert_legacy_data)
│   ├── processed/   (FAISS index lives here)
│   └── evaluation/
│       ├── questions.json
│       └── results.json (after running evaluation)
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── language_detector.py
│   ├── prompt_builder.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   ├── evaluation.py
│   └── convert_legacy_data.py
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── tests/
│   ├── test_language_detector.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_rag_pipeline.py
└── notebooks/
    └── demo.ipynb
```
