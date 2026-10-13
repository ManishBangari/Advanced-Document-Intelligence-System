import os
from typing import Any, Dict, List, Optional

from langchain_groq import ChatGroq


class DummyLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, prompt: Any):
        class Response:
            def __init__(self, text: str):
                self.content = text

        return Response(self.content)


class QAEngine:
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def _build_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations = []
        for result in results:
            metadata = result.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            text = metadata.get("text") or ""
            if not text:
                continue
            citations.append(
                {
                    "chunk_id": metadata.get("chunk_id"),
                    "document_id": metadata.get("document_id"),
                    "source": metadata.get("source"),
                    "file_type": metadata.get("file_type"),
                    "page": metadata.get("page"),
                    "text": text,
                }
            )
        return citations

    def _build_prompt(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        evidence_block = []
        for index, item in enumerate(evidence, start=1):
            source = item.get("source") or "unknown-source"
            page = item.get("page")
            page_text = f" page {page}" if page is not None else " page unknown"
            evidence_block.append(
                f"[{index}] Source: {source}{page_text}\nChunk ID: {item.get('chunk_id') or 'unknown'}\nText: {item.get('text') or ''}"
            )
        context = "\n\n".join(evidence_block)
        return (
            "Answer the user question using only the evidence provided below. "
            "If the evidence is insufficient, explicitly say so and do not invent facts. "
            "Include source identifiers and page numbers when available.\n\n"
            f"Question: {question}\n\nEvidence:\n{context}\n\nAnswer:"
        )

    def answer(self, question: str, results: List[Dict[str, Any]], llm: Optional[Any] = None) -> Dict[str, Any]:
        citations = self._build_citations(results)
        if not citations:
            return {
                "answer": "Insufficient evidence: I couldn't find enough relevant evidence in the indexed documents to answer this question reliably.",
                "status": "insufficient_evidence",
                "citations": [],
                "evidence": [],
            }

        chosen_llm = llm or self.llm
        if chosen_llm is None:
            return {
                "answer": "No generation backend is configured. Retrieved evidence is available below for manual review.",
                "status": "backend_unavailable",
                "citations": citations,
                "evidence": citations,
            }

        try:
            prompt = self._build_prompt(question, citations)
            response = chosen_llm.invoke(prompt)
            answer_text = getattr(response, "content", str(response))
        except Exception:
            return {
                "answer": "The generation backend failed while producing an answer. Retrieved evidence is shown below.",
                "status": "backend_error",
                "citations": citations,
                "evidence": citations,
            }

        return {
            "answer": answer_text,
            "status": "answered",
            "citations": citations,
            "evidence": citations,
        }


def build_local_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-120b")
    except Exception:
        return None
