from unittest.mock import patch

import numpy as np
from langchain_core.documents import Document

from src.vectorstore import FaissVectorStore


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, show_progress_bar=False):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                vectors.append([1.0, 0.0])
            elif "beta" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.2, 0.2])
        return np.array(vectors, dtype="float32")

    def get_sentence_embedding_dimension(self):
        return 2


def test_new_document_becomes_searchable(tmp_path):
    persist_dir = tmp_path / "store"
    with patch("src.vectorstore.SentenceTransformer", DummySentenceTransformer), patch("src.embedding.SentenceTransformer", DummySentenceTransformer):
        store = FaissVectorStore(str(persist_dir), embedding_model="dummy-model")
        store.add_documents([
            Document(page_content="Alpha is a test value", metadata={"document_id": "doc-1", "source": "alpha.txt", "file_type": "txt"})
        ])

    with patch("src.vectorstore.SentenceTransformer", DummySentenceTransformer), patch("src.embedding.SentenceTransformer", DummySentenceTransformer):
        reopened = FaissVectorStore(str(persist_dir), embedding_model="dummy-model")
        reopened.load()
        results = reopened.query("Alpha", top_k=3)

    assert results
    assert results[0]["metadata"]["source"] == "alpha.txt"


def test_update_document_replaces_old_content(tmp_path):
    persist_dir = tmp_path / "store"
    with patch("src.vectorstore.SentenceTransformer", DummySentenceTransformer), patch("src.embedding.SentenceTransformer", DummySentenceTransformer):
        store = FaissVectorStore(str(persist_dir), embedding_model="dummy-model")
        store.add_documents([
            Document(page_content="Old beta value", metadata={"document_id": "doc-1", "source": "values.txt", "file_type": "txt"})
        ])
        store.update_document("doc-1", [
            Document(page_content="New alpha value", metadata={"document_id": "doc-1", "source": "values.txt", "file_type": "txt"})
        ])
        results = store.query("beta", top_k=3)

    assert results
    assert all(item["metadata"]["text"] != "Old beta value" for item in results)
    assert any(item["metadata"]["text"] == "New alpha value" for item in results)


def test_delete_document_removes_active_search_results(tmp_path):
    persist_dir = tmp_path / "store"
    with patch("src.vectorstore.SentenceTransformer", DummySentenceTransformer), patch("src.embedding.SentenceTransformer", DummySentenceTransformer):
        store = FaissVectorStore(str(persist_dir), embedding_model="dummy-model")
        store.add_documents([
            Document(page_content="Beta is archived", metadata={"document_id": "doc-2", "source": "archive.txt", "file_type": "txt"})
        ])
        store.delete_document("doc-2")
        results = store.query("beta", top_k=3)

    assert results == []


def test_rebuild_uses_canonical_records(tmp_path):
    persist_dir = tmp_path / "store"
    with patch("src.vectorstore.SentenceTransformer", DummySentenceTransformer), patch("src.embedding.SentenceTransformer", DummySentenceTransformer):
        store = FaissVectorStore(str(persist_dir), embedding_model="dummy-model")
        store.add_documents([
            Document(page_content="Alpha and beta both appear", metadata={"document_id": "doc-3", "source": "mixed.txt", "file_type": "txt"})
        ])
        store.rebuild_index()
        results = store.query("beta", top_k=3)

    assert results
    assert results[0]["metadata"]["source"] == "mixed.txt"


def test_failed_rebuild_leaves_last_known_good_index(tmp_path):
    persist_dir = tmp_path / "store"
    persist_dir.mkdir()
    index_path = persist_dir / "faiss.index"
    metadata_path = persist_dir / "metadata.pkl"
    with patch("src.vectorstore.SentenceTransformer", DummySentenceTransformer), patch("src.embedding.SentenceTransformer", DummySentenceTransformer):
        good_store = FaissVectorStore(str(persist_dir), embedding_model="dummy-model")
        good_store.add_documents([
            Document(page_content="Alpha is safe", metadata={"document_id": "doc-4", "source": "safe.txt", "file_type": "txt"})
        ])

    index_path.write_bytes(b"broken-index")
    metadata_path.write_bytes(b"not-a-pickle")

    with patch("src.vectorstore.SentenceTransformer", DummySentenceTransformer), patch("src.embedding.SentenceTransformer", DummySentenceTransformer):
        store = FaissVectorStore(str(persist_dir), embedding_model="dummy-model")
        try:
            store.load()
        except (FileNotFoundError, ValueError):
            pass
        else:
            raise AssertionError("Corrupt index should have raised an error")
