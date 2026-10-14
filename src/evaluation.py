from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union


def _normalize_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_id_list(values: Optional[Iterable[Any]]) -> List[str]:
    if values is None:
        return []
    return [str(value) for value in values if value is not None]


def _extract_result_id(item: Any) -> str:
    if isinstance(item, Mapping):
        for key in ("chunk_id", "document_id", "id", "source"):
            if key in item and item.get(key) is not None:
                return _normalize_id(item.get(key))
        if "metadata" in item and isinstance(item["metadata"], Mapping):
            for key in ("chunk_id", "document_id", "id", "source"):
                if key in item["metadata"] and item["metadata"].get(key) is not None:
                    return _normalize_id(item["metadata"].get(key))
        return ""

    if hasattr(item, "chunk_id"):
        return _normalize_id(getattr(item, "chunk_id"))
    if hasattr(item, "document_id"):
        return _normalize_id(getattr(item, "document_id"))
    if hasattr(item, "id"):
        return _normalize_id(getattr(item, "id"))
    if hasattr(item, "source"):
        return _normalize_id(getattr(item, "source"))
    return _normalize_id(item)


def compute_recall_at_k(
    retrieved: Sequence[Any],
    relevant: Sequence[Any],
    k: Optional[int] = None,
) -> float:
    ranked = list(retrieved[:k] if k is not None else retrieved)
    relevant_ids = set(_as_id_list(relevant))
    if not relevant_ids:
        return 0.0
    if not ranked:
        return 0.0
    hits = sum(1 for item in ranked if _extract_result_id(item) in relevant_ids)
    return hits / len(relevant_ids)


def compute_precision_at_k(
    retrieved: Sequence[Any],
    relevant: Sequence[Any],
    k: Optional[int] = None,
) -> float:
    ranked = list(retrieved[:k] if k is not None else retrieved)
    relevant_ids = set(_as_id_list(relevant))
    if not ranked:
        return 0.0
    hits = sum(1 for item in ranked if _extract_result_id(item) in relevant_ids)
    return hits / len(ranked)


def compute_mrr(
    retrieved: Sequence[Any],
    relevant: Sequence[Any],
    k: Optional[int] = None,
) -> float:
    ranked = list(retrieved[:k] if k is not None else retrieved)
    relevant_ids = set(_as_id_list(relevant))
    if not relevant_ids:
        return 0.0
    for rank, item in enumerate(ranked, start=1):
        if _extract_result_id(item) in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved: Sequence[Any],
    relevant: Sequence[Any],
    k: Optional[int] = None,
    relevance_grades: Optional[Mapping[Any, Union[int, float]]] = None,
) -> float:
    ranked = list(retrieved[:k] if k is not None else retrieved)
    if not ranked:
        return 0.0

    if relevance_grades is None:
        relevant_ids = set(_as_id_list(relevant))
        relevance = {str(item): (1.0 if _extract_result_id(item) in relevant_ids else 0.0) for item in ranked}
    else:
        relevance = {str(key): float(value) for key, value in relevance_grades.items()}

    dcg = 0.0
    for rank, item in enumerate(ranked, start=1):
        item_id = _extract_result_id(item)
        relevance_score = relevance.get(item_id, relevance.get(str(item), 0.0))
        if relevance_score <= 0:
            continue
        dcg += ((2**relevance_score) - 1.0) / math.log2(rank + 1)

    if relevance_grades is None:
        ideal = [1.0 if _extract_result_id(item) in set(_as_id_list(relevant)) else 0.0 for item in ranked]
    else:
        ideal = [float(value) for value in relevance_grades.values()]

    if not any(score > 0 for score in ideal):
        return 0.0

    ideal_sorted = sorted(ideal, reverse=True)
    idcg = 0.0
    for rank, value in enumerate(ideal_sorted[: len(ranked)], start=1):
        if value <= 0:
            continue
        idcg += ((2**value) - 1.0) / math.log2(rank + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


@dataclass
class EvaluationConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_mode: str = "hybrid"
    top_k: int = 5
    reranker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalExample:
    query: str
    relevant_document_ids: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)
    reference_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalEvaluationReport:
    config: EvaluationConfig
    metrics: Dict[str, float]
    per_query: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "per_query": self.per_query,
        }


def build_synthetic_evaluation_dataset() -> List[RetrievalExample]:
    return [
        RetrievalExample(
            query="What is the attention mechanism?",
            relevant_document_ids=["doc_attention"],
            relevant_chunk_ids=["chunk_attention_01", "chunk_attention_02"],
            reference_answer="The attention mechanism weights different tokens based on relevance to each other.",
            metadata={"domain": "transformers"},
        ),
        RetrievalExample(
            query="How does retrieval augmented generation work?",
            relevant_document_ids=["doc_rag"],
            relevant_chunk_ids=["chunk_rag_01", "chunk_rag_02"],
            reference_answer="RAG combines a retriever with a generator to ground responses in documents.",
            metadata={"domain": "rag"},
        ),
        RetrievalExample(
            query="What are the trade-offs between supervised and unsupervised learning?",
            relevant_document_ids=["doc_ml_basics"],
            relevant_chunk_ids=["chunk_ml_01"],
            reference_answer="Supervised learning uses labeled examples; unsupervised learning finds structure without labels.",
            metadata={"domain": "machine_learning"},
        ),
    ]


def load_synthetic_evaluation_dataset(path: Optional[Union[str, Path]] = None) -> List[RetrievalExample]:
    if path is None:
        default_path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "synthetic_corpus.json"
        if default_path.exists():
            path = default_path
        else:
            return build_synthetic_evaluation_dataset()

    file_path = Path(path)
    if not file_path.exists():
        return build_synthetic_evaluation_dataset()

    raw = json.loads(file_path.read_text(encoding="utf-8"))
    examples = raw.get("queries") or raw.get("examples") or []
    return [
        RetrievalExample(
            query=item.get("query", ""),
            relevant_document_ids=item.get("relevant_document_ids", []),
            relevant_chunk_ids=item.get("relevant_chunk_ids", []),
            reference_answer=item.get("reference_answer", ""),
            metadata=item.get("metadata", {}),
        )
        for item in examples
    ]


def evaluate_retrieval(
    retrieved: Sequence[Any],
    relevant: Sequence[Any],
    top_k: Optional[int] = None,
    relevance_grades: Optional[Mapping[Any, Union[int, float]]] = None,
) -> Dict[str, float]:
    if top_k is None:
        ranked = list(retrieved)
    else:
        ranked = list(retrieved[:top_k])

    return {
        "recall@k": compute_recall_at_k(ranked, relevant, k=top_k),
        "precision@k": compute_precision_at_k(ranked, relevant, k=top_k),
        "mrr": compute_mrr(ranked, relevant, k=top_k),
        "ndcg@k": compute_ndcg_at_k(ranked, relevant, k=top_k, relevance_grades=relevance_grades),
    }


def evaluate_dataset(
    dataset: Sequence[RetrievalExample],
    ranked_results: Sequence[Sequence[Any]],
    config: Optional[EvaluationConfig] = None,
) -> RetrievalEvaluationReport:
    if config is None:
        config = EvaluationConfig()

    per_query: List[Dict[str, Any]] = []
    aggregate: Dict[str, List[float]] = {"recall@k": [], "precision@k": [], "mrr": [], "ndcg@k": []}

    for example, result_list in zip(dataset, ranked_results):
        metrics = evaluate_retrieval(result_list, example.relevant_document_ids, top_k=config.top_k)
        per_query.append(
            {
                "query": example.query,
                "relevant_document_ids": example.relevant_document_ids,
                "retrieved": [_extract_result_id(item) for item in result_list[: config.top_k]],
                "metrics": metrics,
            }
        )
        for metric_name, metric_value in metrics.items():
            aggregate[metric_name].append(metric_value)

    averaged = {
        name: (sum(values) / len(values) if values else 0.0) for name, values in aggregate.items()
    }
    return RetrievalEvaluationReport(config=config, metrics=averaged, per_query=per_query)


def measure_latency(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {"result": result, "latency_ms": elapsed_ms}


time_operation = measure_latency
calculate_retrieval_metrics = evaluate_retrieval
load_evaluation_dataset = load_synthetic_evaluation_dataset
build_default_eval_dataset = build_synthetic_evaluation_dataset


__all__ = [
    "EvaluationConfig",
    "RetrievalExample",
    "RetrievalEvaluationReport",
    "build_synthetic_evaluation_dataset",
    "build_default_eval_dataset",
    "evaluate_dataset",
    "evaluate_retrieval",
    "calculate_retrieval_metrics",
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_mrr",
    "compute_ndcg_at_k",
    "load_synthetic_evaluation_dataset",
    "load_evaluation_dataset",
    "measure_latency",
    "time_operation",
]
