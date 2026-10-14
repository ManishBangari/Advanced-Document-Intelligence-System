# Advanced Document Intelligence System

This project is a local-first document RAG prototype for ingesting documents, splitting them into chunks, embedding them with a local sentence-transformer model, and retrieving relevant passages with FAISS.

## Current status

The repository currently supports a local ingestion and vector-search flow with local files under the `data/` directory. Phase 1 adds reliable ingestion metadata, file hashing, duplicate detection, and deterministic chunk IDs. Phase 2 adds safe FAISS lifecycle management with SQLite-backed canonical metadata. Phase 3 adds hybrid retrieval with BM25 + vector fusion. Phase 4 adds evidence-grounded QA and explicit abstention when relevant passages are absent. Phase 5 adds structured document comparison that flags identical passages, value changes, not-found cases, and uncertain comparisons with real source citations. Phase 6 adds schema-validated structured extraction from document text with source references and explicit missing-field handling. Phase 7 adds a Streamlit UI with document management, document Q&A, side-by-side comparison, and structured extraction workflows built over the tested service modules. Phase 8 adds a repeatable evaluation layer for retrieval quality, deterministic scoring, latency measurement, and a small synthetic corpus for offline QA benchmarking.

## Supported file types

- PDF
- TXT
- CSV
- XLS/XLSX
- DOCX
- JSON

## Ingestion metadata

Each loaded document keeps source metadata such as:

- `document_id`
- `source`
- `file_type`
- `file_hash`
- `page` (when available)
- `chunk_index`
- `chunk_id` after chunking

The project uses SHA-256 hashing to detect duplicate or changed files. Empty files and empty chunks are skipped safely.

## Hybrid retrieval

The retrieval layer supports three modes:

- `vector` for semantic retrieval only
- `bm25` for keyword retrieval only
- `hybrid` for Reciprocal Rank Fusion (RRF) across semantic and BM25 result lists

This keeps retrieval independent from generation and allows deterministic, metadata-rich result fusion without requiring a hosted vector database.

## Document comparison

The comparison layer supports evidence-backed comparisons across two or more selected documents. It reports:

- identical passages with no substantive change
- value changes such as date or numeric differences
- missing sections as `not_found_in_source` rather than proof of absence
- uncertain comparisons when evidence is ambiguous or scope differs
- real citations for both sides, including source and chunk metadata when available

The comparison module is intentionally conservative and does not infer claims that are not directly supported by the selected passages.

## Structured extraction

The extraction layer supports schema-driven field extraction from document text using Pydantic validation. It preserves source evidence for each extracted value and leaves missing or unsupported fields as `null`/missing instead of guessing. The default schema covers common document facts such as contract IDs, monetary amounts, effective dates, and status values; custom schema definitions can be supplied for other document types.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The UI exposes tabs for dashboard, document management, question answering, document comparison, and structured extraction. The app starts with no uploaded files and creates the required local folders automatically.

## Evaluation and reliability

The project includes a lightweight, deterministic evaluation module at `src/evaluation.py` and a version-controlled synthetic corpus at `data/evaluation/synthetic_corpus.json`.

```python
from src.evaluation import evaluate_retrieval

metrics = evaluate_retrieval(
    ["doc-3", "doc-1", "doc-9"],
    ["doc-1", "doc-2"],
    top_k=3,
)
print(metrics)
```

The evaluation layer computes:

- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- nDCG@K
- latency for measured callables

The scores are intended for reproducible comparison across retrieval settings, not as proof of factual correctness in the generated answer.

## Testing

```bash
pytest -q
```

## Notes

- The app is designed to avoid hardcoded content and to keep ingestion metadata available through retrieval.
- FAISS remains a derived search index; canonical metadata should be preserved in the document ingest layer.
- This project is local-first and does not require a hosted vector database for core usage.
