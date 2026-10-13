import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "with",
    "without", "from", "into", "onto", "over", "under", "of", "to", "in", "on",
    "at", "by", "as", "is", "it", "its", "be", "been", "are", "was", "were", "this",
    "that", "these", "those", "we", "you", "they", "he", "she", "itself", "them",
    "their", "our", "your", "his", "her", "not", "no", "yes", "can", "may", "must",
    "should", "could", "would", "will", "do", "does", "did", "have", "has", "had",
    "about", "after", "before", "between", "during", "through", "within", "because",
    "while", "where", "when", "which", "who", "what", "how", "why", "there", "here",
    "other", "same", "more", "most", "some", "any", "all", "each", "both"
}


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _tokenize(value: Any) -> List[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", (str(value or "")).lower()) if token not in _STOPWORDS]


def _extract_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _safe_metadata(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {}
    return dict(getattr(item, "metadata", {}) or {})


def _extract_document_text(item: Any) -> str:
    if isinstance(item, dict):
        if isinstance(item.get("metadata"), dict):
            text = item["metadata"].get("text")
            if text:
                return str(text)
        if item.get("page_content"):
            return str(item["page_content"])
        if item.get("text"):
            return str(item["text"])
        return ""
    return str(getattr(item, "page_content", "") or getattr(item, "text", "") or "")


def _extract_citation(item: Any) -> Dict[str, Any]:
    metadata = _safe_metadata(item)
    text = _extract_document_text(item)
    return {
        "document_id": metadata.get("document_id") or metadata.get("source") or "unknown-document",
        "chunk_id": metadata.get("chunk_id"),
        "source": metadata.get("source") or "unknown-source",
        "file_type": metadata.get("file_type"),
        "page": metadata.get("page"),
        "text": text,
    }


def _infer_topic(documents: Sequence[Any]) -> str:
    if not documents:
        return "document"

    token_counts: Counter[str] = Counter()
    for document in documents:
        for token in _tokenize(_extract_document_text(document)):
            token_counts[token] += 1

    top_tokens = [token for token, _ in token_counts.most_common(5) if token]
    if top_tokens:
        return top_tokens[0]
    return "comparison"


def _find_relevant_statement(document: Any, topic: str) -> Optional[str]:
    text = _extract_document_text(document)
    sentences = _extract_sentences(text)
    if not sentences:
        return None

    normalized_topic = topic.lower()
    scored: List[Tuple[float, str]] = []
    for sentence in sentences:
        lowered = sentence.lower()
        score = 0.0
        if normalized_topic in lowered:
            score += 5.0
        for token in _tokenize(topic):
            if token in lowered:
                score += 1.0
        if re.search(r"\d+", sentence):
            score += 0.5
        if re.search(r"\d{4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", sentence.lower()):
            score += 0.5
        scored.append((score, sentence))

    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if scored[0][0] <= 0:
        return None
    return scored[0][1]


def _contains_numeric_or_date_difference(left: str, right: str) -> bool:
    left_numbers = re.findall(r"\d+(?:\.\d+)?", left)
    right_numbers = re.findall(r"\d+(?:\.\d+)?", right)
    if left_numbers and right_numbers and left_numbers != right_numbers:
        return True
    left_dates = re.findall(r"\d{4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", left.lower())
    right_dates = re.findall(r"\d{4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", right.lower())
    return bool(left_dates and right_dates and left_dates != right_dates)


def _compare_statement_pair(left: str, right: str, topic: str) -> Dict[str, Any]:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)

    if not left_norm and not right_norm:
        return {
            "status": "not_found",
            "change_type": "not_found_in_source",
            "uncertainty": "No relevant evidence was found in the selected source.",
        }
    if not left_norm:
        return {
            "status": "missing_left",
            "change_type": "not_found_in_source",
            "uncertainty": f"Not found in the selected source: no relevant passage matching '{topic}' was found in the left-hand source.",
        }
    if not right_norm:
        return {
            "status": "missing_right",
            "change_type": "not_found_in_source",
            "uncertainty": f"Not found in the selected source: no relevant passage matching '{topic}' was found in the right-hand source.",
        }
    if left_norm == right_norm:
        return {
            "status": "identical",
            "change_type": "no_substantive_change",
            "uncertainty": "The selected passages are effectively identical; no substantive change is reported.",
        }

    left_tokens = set(_tokenize(left_norm))
    right_tokens = set(_tokenize(right_norm))
    shared_tokens = left_tokens & right_tokens
    if not shared_tokens and topic.lower() not in left_norm.lower() and topic.lower() not in right_norm.lower():
        return {
            "status": "uncertain",
            "change_type": "uncertain",
            "uncertainty": "The selected passages do not clearly refer to the same topic, so the difference is uncertain.",
        }

    if any(term in left_norm.lower() or term in right_norm.lower() for term in ["may", "might", "possibly", "uncertain", "unclear", "could", "depends", "not confirmed", "not yet", "unknown", "likely"]):
        return {
            "status": "uncertain",
            "change_type": "uncertain",
            "uncertainty": "The evidence is too uncertain to distinguish a true contradiction from a tentative or context-dependent statement.",
        }

    if _contains_numeric_or_date_difference(left_norm, right_norm):
        return {
            "status": "changed",
            "change_type": "value_changed",
            "uncertainty": "The value changed in the two sources and appears to be a concrete difference.",
        }

    if len(shared_tokens) >= 2:
        return {
            "status": "changed",
            "change_type": "wording_difference",
            "uncertainty": "The sources discuss the same topic but use different wording; the distinction may reflect scope or wording rather than a direct contradiction.",
        }

    return {
        "status": "uncertain",
        "change_type": "uncertain",
        "uncertainty": "The evidence is ambiguous and cannot reliably distinguish a contradiction from a wording or scope difference.",
    }


class DocumentComparator:
    def __init__(self, retriever: Optional[Any] = None):
        self.retriever = retriever

    def _prepare_document(self, document: Any) -> Dict[str, Any]:
        metadata = _safe_metadata(document)
        if isinstance(document, dict):
            page_content = document.get("page_content") or document.get("text") or ""
        else:
            page_content = getattr(document, "page_content", "") or getattr(document, "text", "") or ""
        return {
            "metadata": metadata,
            "page_content": page_content,
            "source": metadata.get("source") or "unknown-source",
            "document_id": metadata.get("document_id") or metadata.get("source") or "unknown-document",
        }

    def compare_documents(self, documents: Sequence[Any], topic: Optional[str] = None) -> List[Dict[str, Any]]:
        normalized = [self._prepare_document(document) for document in documents if document is not None]
        if len(normalized) < 2:
            return []

        comparison_topic = (topic or _infer_topic(normalized)).strip() or "comparison"
        prepared_results: List[Dict[str, Any]] = []

        for index in range(len(normalized) - 1):
            left = normalized[index]
            right = normalized[index + 1]
            left_text = _find_relevant_statement(left, comparison_topic)
            right_text = _find_relevant_statement(right, comparison_topic)

            result = _compare_statement_pair(left_text, right_text, comparison_topic)
            result.update(
                {
                    "topic": comparison_topic,
                    "left": {
                        "document_id": left["document_id"],
                        "source": left["source"],
                        "citation": _extract_citation(left),
                        "value": left_text,
                    },
                    "right": {
                        "document_id": right["document_id"],
                        "source": right["source"],
                        "citation": _extract_citation(right),
                        "value": right_text,
                    },
                }
            )
            prepared_results.append(result)

        return prepared_results

    def compare(self, documents: Sequence[Any], topic: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.compare_documents(documents=documents, topic=topic)


def compare_documents(documents: Sequence[Any], topic: Optional[str] = None) -> List[Dict[str, Any]]:
    return DocumentComparator().compare_documents(documents=documents, topic=topic)


def compare_document_pair(left: Any, right: Any, topic: Optional[str] = None) -> Dict[str, Any]:
    comparison_topic = (topic or _infer_topic([left, right])).strip() or "comparison"
    left_text = _find_relevant_statement(left, comparison_topic)
    right_text = _find_relevant_statement(right, comparison_topic)
    result = _compare_statement_pair(left_text, right_text, comparison_topic)
    result.update(
        {
            "topic": comparison_topic,
            "left": {
                "document_id": _safe_metadata(left).get("document_id") or _safe_metadata(left).get("source") or "unknown-document",
                "source": _safe_metadata(left).get("source") or "unknown-source",
                "citation": _extract_citation(left),
                "value": left_text,
            },
            "right": {
                "document_id": _safe_metadata(right).get("document_id") or _safe_metadata(right).get("source") or "unknown-document",
                "source": _safe_metadata(right).get("source") or "unknown-source",
                "citation": _extract_citation(right),
                "value": right_text,
            },
        }
    )
    return result
