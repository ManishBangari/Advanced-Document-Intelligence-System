import hashlib
import os
import pickle
import sqlite3
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        self.index_path = os.path.join(self.persist_dir, "faiss.index")
        self.metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.db_path = os.path.join(self.persist_dir, "documents.db")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._initialize_database()
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def _connect_db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize_database(self):
        with self._connect_db() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    file_type TEXT,
                    file_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source TEXT,
                    file_type TEXT,
                    page INTEGER,
                    text TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
            )

    def _document_key(self, document: Any) -> str:
        metadata = dict(getattr(document, "metadata", {}) or {})
        candidate = metadata.get("document_id")
        if candidate:
            return str(candidate)
        source = metadata.get("source") or "unknown-source"
        text = (getattr(document, "page_content", "") or "").strip()
        return hashlib.sha256(f"{source}:{text}".encode("utf-8")).hexdigest()[:16]

    def _read_chunks_from_db(self) -> List[Dict[str, Any]]:
        with self._connect_db() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, source, file_type, page, text
                FROM chunks
                ORDER BY document_id, page, chunk_id
                """
            ).fetchall()
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "file_type": row[3],
                "page": row[4],
                "text": row[5],
            }
            for row in rows
        ]

    def _safe_write_index(self, index: Optional[faiss.Index], metadata: List[Dict[str, Any]]):
        if index is None:
            self.index = None
            self.metadata = []
            return

        tmp_index_path = f"{self.index_path}.tmp"
        tmp_metadata_path = f"{self.metadata_path}.tmp"
        faiss.write_index(index, tmp_index_path)
        with open(tmp_metadata_path, "wb") as handle:
            pickle.dump(metadata, handle)
        os.replace(tmp_index_path, self.index_path)
        os.replace(tmp_metadata_path, self.metadata_path)
        self.index = index
        self.metadata = metadata

    def _rebuild_index_from_db(self):
        rows = self._read_chunks_from_db()
        if not rows:
            self.index = None
            self.metadata = []
            return

        texts = [row["text"] for row in rows]
        embeddings = self.model.encode(texts, show_progress_bar=False).astype("float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        metadata = [
            {
                "document_id": row["document_id"],
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "file_type": row["file_type"],
                "page": row["page"],
                "text": row["text"],
            }
            for row in rows
        ]
        self._safe_write_index(index, metadata)

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        self.add_documents(documents)
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_documents(self, documents: List[Any]):
        if not documents:
            return []

        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunk_records: List[Dict[str, Any]] = []

        with self._connect_db() as connection:
            for document in documents:
                if document is None:
                    continue
                page_content = (getattr(document, "page_content", "") or "").strip()
                if not page_content:
                    continue

                metadata = dict(getattr(document, "metadata", {}) or {})
                document_id = str(metadata.get("document_id") or self._document_key(document))
                source = metadata.get("source") or "unknown-source"
                file_type = metadata.get("file_type") or "unknown"
                file_hash = metadata.get("file_hash") or hashlib.sha256(page_content.encode("utf-8")).hexdigest()

                connection.execute(
                    """
                    INSERT INTO documents (document_id, source, file_type, file_hash, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(document_id) DO UPDATE SET
                        source=excluded.source,
                        file_type=excluded.file_type,
                        file_hash=excluded.file_hash,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (document_id, source, file_type, file_hash),
                )
                connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

                chunks = emb_pipe.chunk_documents([document])
                for chunk in chunks:
                    chunk_metadata = dict(chunk.metadata or {})
                    chunk_id = chunk_metadata.get("chunk_id") or hashlib.sha256(
                        f"{document_id}:{chunk.page_content}".encode("utf-8")
                    ).hexdigest()[:16]
                    chunk_text = (chunk.page_content or "").strip()
                    if not chunk_text:
                        continue
                    chunk_metadata.update(
                        {
                            "document_id": document_id,
                            "chunk_id": chunk_id,
                            "source": chunk_metadata.get("source") or source,
                            "file_type": chunk_metadata.get("file_type") or file_type,
                            "page": chunk_metadata.get("page"),
                            "text": chunk_text,
                        }
                    )
                    connection.execute(
                        """
                        INSERT INTO chunks (chunk_id, document_id, source, file_type, page, text)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(chunk_id) DO UPDATE SET
                            document_id=excluded.document_id,
                            source=excluded.source,
                            file_type=excluded.file_type,
                            page=excluded.page,
                            text=excluded.text
                        """,
                        (
                            chunk_id,
                            document_id,
                            chunk_metadata.get("source"),
                            chunk_metadata.get("file_type"),
                            chunk_metadata.get("page"),
                            chunk_text,
                        ),
                    )
                    chunk_records.append(chunk_metadata)

        self.rebuild_index()
        return [record["chunk_id"] for record in chunk_records]

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if embeddings.size == 0:
            self.index = None
            self.metadata = []
            return
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        if self.index is None:
            self.metadata = []
            return
        self._safe_write_index(self.index, self.metadata)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        missing_files = []
        if not os.path.exists(self.index_path):
            missing_files.append(self.index_path)
        if not os.path.exists(self.metadata_path):
            missing_files.append(self.metadata_path)
        if missing_files:
            raise FileNotFoundError(f"FAISS index files are missing: {missing_files}")

        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as handle:
                self.metadata = pickle.load(handle)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Corrupt or incompatible FAISS files under {self.persist_dir}: {exc}") from exc

        if self.index is None and self.metadata:
            raise ValueError(f"Metadata existed but FAISS index was not readable: {self.persist_dir}")
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def rebuild_index(self):
        try:
            self._rebuild_index_from_db()
        except Exception as exc:
            raise ValueError(f"Failed to rebuild FAISS index from the canonical database: {exc}") from exc

    def delete_document(self, document_id: str):
        with self._connect_db() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        self.rebuild_index()

    def update_document(self, document_id: str, documents: List[Any]):
        if not documents:
            self.delete_document(document_id)
            return
        with self._connect_db() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        self.add_documents(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            return []
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        D, I = self.index.search(query_embedding, min(top_k, len(self.metadata) or 1))
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        if self.index is None:
            return []
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)


if __name__ == "__main__":
    from src.data_loader import load_all_documents

    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
