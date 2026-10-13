import os
import re
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi

from src.qa_engine import QAEngine, build_local_llm
from src.vectorstore import FaissVectorStore

load_dotenv()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


class HybridRetriever:
    def __init__(self, vectorstore: FaissVectorStore):
        self.vectorstore = vectorstore

    def _metadata_key(self, item: Dict[str, Any]) -> str:
        metadata = item.get("metadata") or {}
        if isinstance(metadata, dict):
            return (
                metadata.get("chunk_id")
                or metadata.get("document_id")
                or metadata.get("source")
                or metadata.get("text")
                or str(item.get("index"))
            )
        return str(item.get("index"))

    def _vector_results(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not hasattr(self.vectorstore, "query"):
            return []
        return self.vectorstore.query(query, top_k=top_k)

    def _bm25_results(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        metadata_items = getattr(self.vectorstore, "metadata", []) or []
        if not metadata_items:
            return []

        corpus = []
        for metadata in metadata_items:
            if not isinstance(metadata, dict):
                continue
            text = metadata.get("text") or ""
            corpus.append(text)

        if not corpus:
            return []

        tokenized_corpus = [_tokenize(text) for text in corpus]
        tokens = _tokenize(query)
        if not tokens:
            return []

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        results = []
        for idx in ranked_indices:
            metadata = metadata_items[idx]
            if isinstance(metadata, dict):
                results.append({"index": idx, "score": float(scores[idx]), "metadata": metadata})
        return results

    def reciprocal_rank_fusion(self, result_sets: Iterable[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        fused_scores: Dict[str, float] = {}
        fused_items: Dict[str, Dict[str, Any]] = {}
        for result_set in result_sets:
            for rank, item in enumerate(result_set):
                key = self._metadata_key(item)
                fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (k + rank + 1))
                if key not in fused_items:
                    fused_items[key] = item

        ranked = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
        merged: List[Dict[str, Any]] = []
        for key, score in ranked:
            item = fused_items[key]
            item = dict(item)
            item["score"] = float(score)
            merged.append(item)
        return merged

    def search(self, query: str, top_k: int = 5, retrieval_mode: str = "hybrid") -> List[Dict[str, Any]]:
        retrieval_mode = (retrieval_mode or "hybrid").lower()
        vector_results = self._vector_results(query, top_k=top_k)
        bm25_results = self._bm25_results(query, top_k=top_k)

        if retrieval_mode == "vector":
            return vector_results[:top_k]
        if retrieval_mode == "bm25":
            return bm25_results[:top_k]

        if not vector_results and not bm25_results:
            return []

        merged = self.reciprocal_rank_fusion([vector_results, bm25_results])
        return merged[:top_k]


class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "openai/gpt-oss-120b"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.retriever = HybridRetriever(self.vectorstore)
        self.llm_model = llm_model
        self.llm = build_local_llm()
        if self.llm is None:
            print("[INFO] No local generation backend configured; retrieval-only mode is available.")
        else:
            print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search(self, query: str, top_k: int = 5, retrieval_mode: str = "hybrid") -> List[Dict[str, Any]]:
        return self.retriever.search(query=query, top_k=top_k, retrieval_mode=retrieval_mode)

    def answer_question(self, query: str, top_k: int = 5, retrieval_mode: str = "hybrid") -> Dict[str, Any]:
        results = self.search(query, top_k=top_k, retrieval_mode=retrieval_mode)
        engine = QAEngine(llm=self.llm)
        return engine.answer(query, results, llm=self.llm)

    def search_and_summarize(self, query: str, top_k: int = 5, retrieval_mode: str = "hybrid") -> str:
        results = self.search(query, top_k=top_k, retrieval_mode=retrieval_mode)
        texts = [result.get("metadata", {}).get("text", "") for result in results if isinstance(result, dict) and result.get("metadata")]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        if self.llm is None:
            return "No generation backend is configured. Retrieved evidence is available for inspection."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content


if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
