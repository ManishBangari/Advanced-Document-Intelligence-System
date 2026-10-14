# Advanced Document Intelligence System

This repository is a local-first document intelligence application built around retrieval-augmented generation, document comparison, and structured information extraction. It ingests supported local files, splits them into chunks, embeds the chunks with a sentence-transformer model, stores them in a local FAISS index, and retrieves relevant passages with optional BM25 and hybrid search.

## What the project does

- Ingests documents from the local `data/` directory
- Preserves source metadata such as document ID, file hash, file type, source label, and page number when available
- Builds a local FAISS index and stores canonical metadata in SQLite
- Supports vector-only, BM25-only, and hybrid retrieval
- Answers questions using retrieved evidence and explicit abstention when evidence is insufficient
- Compares two selected documents with conservative, evidence-backed difference reporting
- Extracts structured fields with Pydantic validation and source references
- Provides a Streamlit interface for documents, search, comparison, and extraction
- Includes offline evaluation helpers and a synthetic benchmark corpus for reproducible testing

## Architecture and data flow

The app is intentionally layered so that the UI stays thin and the business logic remains in modules under `src/`:

- `src/data_loader.py` loads supported files and normalizes metadata
- `src/embedding.py` handles chunking and embedding configuration
- `src/vectorstore.py` builds and persists the FAISS index plus SQLite metadata tables
- `src/search.py` wraps vector/BM25/hybrid retrieval and optional LLM-backed answering
- `src/comparison.py` compares document evidence and flags uncertainty
- `src/information_extractor.py` validates structured fields and keeps evidence chains
- `src/evaluation.py` provides retrieval metrics and runtime measurement utilities
- `app.py` exposes the Streamlit interface

All generated runtime artifacts are local to the repository, including the FAISS index and SQLite metadata database.

## Supported file types

- PDF
- TXT
- CSV
- XLS/XLSX
- DOCX
- JSON

## Installation

### Option 1: Local Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: uv

```bash
uv sync
uv run streamlit run app.py
```

### Option 3: Docker

```bash
docker build -t rag-app .
docker run --rm -p 8501:8501 --env-file .env rag-app
```

## Configuration

Copy the example environment file and fill in any optional provider keys:

```bash
cp .env.example .env
```

Available settings:

- `GROQ_API_KEY` for optional Groq-backed generation
- `OPENAI_API_KEY` for optional OpenAI-backed generation if configured in the codebase later
- `STREAMLIT_SERVER_PORT` for local/in-container port configuration
- `APP_DATA_DIR` and `INDEX_DIR` for local storage paths

The application can run in retrieval-only mode without any LLM provider configured.

## Local model setup

The default embedding model is `all-MiniLM-L6-v2`, which is downloaded locally by `sentence-transformers`. The first run may take a few moments as the model downloads.

If an LLM generation backend is enabled, it remains optional and must be configured explicitly. The app does not require paid APIs for ingestion, indexing, retrieval, comparison, extraction, or evaluation.

## Data storage and privacy behavior

The project stores uploaded and processed files under the local `data/` directory and writes index metadata and FAISS artifacts under `faiss_store/`.

This app is intended for local-first, privacy-conscious use. It does not require a hosted vector database or cloud storage for the core workflows. Sensitive documents should remain on your local machine or in a private environment; do not commit `.env` or private file uploads to version control.

## Evaluation and reliability

The project includes a lightweight evaluation layer at `src/evaluation.py` and a version-controlled synthetic dataset at `data/evaluation/synthetic_corpus.json`.

```python
from src.evaluation import evaluate_retrieval

metrics = evaluate_retrieval(
    ["doc-3", "doc-1", "doc-9"],
    ["doc-1", "doc-2"],
    top_k=3,
)
print(metrics)
```

The evaluation module computes:

- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- nDCG@K
- callable latency measurements

These metrics are intended for reproducible offline benchmarking of retrieval behavior. They do not prove factual correctness of generated answers.

## Testing

```bash
pytest -q
```

The default suite is offline and does not depend on paid APIs.

## Troubleshooting

- If the app cannot find files: ensure the `data/` directory exists and contains supported file types.
- If indexing or retrieval fails: delete stale generated artifacts in `faiss_store/` and rerun the application.
- If the embedding model download is slow: the first run may take time while the model is fetched to the local cache.
- If a provider-backed answer fails: the app falls back to retrieval evidence rather than fabricating a result.
- If Docker is used: ensure `.env` is provided and that port 8501 is open on the host machine.

## Notes and known limitations

- The app is designed to avoid hardcoded content and to keep ingestion metadata available through retrieval.
- FAISS remains a derived search index; canonical metadata is preserved in the ingest layer and SQLite tables.
- Empty or malformed files are skipped with warnings rather than crashing the app.
- Local LLM generation is optional; retrieval and comparison workflows remain usable without a model provider.
- This project is local-first and does not require a hosted vector database for core usage.
