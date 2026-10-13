# Advanced Document Intelligence System

This project is a local-first document RAG prototype for ingesting documents, splitting them into chunks, embedding them with a local sentence-transformer model, and retrieving relevant passages with FAISS.

## Current status

The repository currently supports a basic ingestion and vector-search flow with local files under the `data/` directory. Phase 1 adds reliable ingestion metadata, file hashing, duplicate detection, and deterministic chunk IDs.

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

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Testing

```bash
pytest -q
```

## Notes

- The app is designed to avoid hardcoded content and to keep ingestion metadata available through retrieval.
- FAISS remains a derived search index; canonical metadata should be preserved in the document ingest layer.
- This project is local-first and does not require a hosted vector database for core usage.
