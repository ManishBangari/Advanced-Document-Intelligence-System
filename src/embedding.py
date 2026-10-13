import hashlib
from typing import Any, List, Optional

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be zero or greater")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks: List[Any] = []
        for document in documents:
            if document is None:
                continue
            document_text = (getattr(document, "page_content", "") or "").strip()
            if not document_text:
                print(f"[WARN] Skipping empty document chunking for source={document.metadata.get('source', 'unknown')}")
                continue

            split_docs = splitter.split_documents([document])
            for chunk_index, chunk in enumerate(split_docs):
                text = (chunk.page_content or "").strip()
                if not text:
                    continue

                metadata = dict(chunk.metadata or {})
                document_id = metadata.get("document_id") or (document.metadata or {}).get("document_id") or "unknown-document"
                page = metadata.get("page") or (document.metadata or {}).get("page")
                source = metadata.get("source") or (document.metadata or {}).get("source") or "unknown-source"
                file_type = metadata.get("file_type") or (document.metadata or {}).get("file_type") or "unknown"

                chunk_id = hashlib.sha256(
                    f"{document_id}:{chunk_index}:{text}".encode("utf-8")
                ).hexdigest()[:16]

                metadata.update(
                    {
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "source": source,
                        "file_type": file_type,
                        "page": page,
                        "text": text,
                    }
                )
                chunk.metadata = metadata
                chunks.append(chunk)

        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        valid_chunks = []
        for chunk in chunks:
            text = (chunk.page_content or "").strip()
            if text:
                valid_chunks.append(chunk)

        if not valid_chunks:
            dim = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype="float32")

        texts = [chunk.page_content.strip() for chunk in valid_chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
