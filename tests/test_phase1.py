from unittest.mock import patch

import numpy as np
from langchain_core.documents import Document

from src.data_loader import detect_duplicate_hashes, load_all_documents
from src.embedding import EmbeddingPipeline


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, show_progress_bar=False):
        return np.array([[float(index % 3), float(index % 2)] for index in range(len(texts))], dtype="float32")

    def get_sentence_embedding_dimension(self):
        return 2


def test_load_all_documents_preserves_metadata_and_detects_duplicates(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    first = docs_dir / "first.txt"
    second = docs_dir / "second.txt"
    first.write_text("hello world\nthis is the same content\n", encoding="utf-8")
    second.write_text("hello world\nthis is the same content\n", encoding="utf-8")

    documents = load_all_documents(str(docs_dir))

    assert len(documents) == 2
    for document in documents:
        metadata = document.metadata
        assert metadata["document_id"]
        assert metadata["source"] in {"first.txt", "second.txt"}
        assert metadata["file_type"] == "txt"
        assert metadata["file_hash"]
        assert metadata["duplicate_hash"] is False

    duplicates = detect_duplicate_hashes(documents)
    assert len(duplicates) == 1
    assert next(iter(duplicates.values()))[0].metadata["file_hash"]


def test_empty_files_are_skipped(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    documents = load_all_documents(str(tmp_path))
    assert documents == []


def test_embedding_pipeline_creates_deterministic_chunk_ids(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Alpha beta gamma delta epsilon zeta\n", encoding="utf-8")

    document = Document(
        page_content=text_file.read_text(encoding="utf-8"),
        metadata={"document_id": "doc-123", "source": "sample.txt", "file_type": "txt", "page": 0},
    )

    with patch("src.embedding.SentenceTransformer", DummySentenceTransformer):
        pipeline = EmbeddingPipeline(chunk_size=12, chunk_overlap=0)
        chunks = pipeline.chunk_documents([document])
        embeddings = pipeline.embed_chunks(chunks)

    assert chunks
    assert len(chunks) == len({chunk.metadata["chunk_id"] for chunk in chunks})
    assert all("text" in chunk.metadata for chunk in chunks)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 2
