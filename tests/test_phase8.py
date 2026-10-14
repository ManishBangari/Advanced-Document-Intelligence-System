from src.evaluation import (
    EvaluationConfig,
    build_synthetic_evaluation_dataset,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_dataset,
    evaluate_retrieval,
    load_synthetic_evaluation_dataset,
    measure_latency,
)


def test_retrieval_metrics_match_expected_values_for_fixed_ranked_list():
    retrieved = ["doc-3", "doc-1", "doc-9", "doc-2"]
    relevant = ["doc-1", "doc-2"]

    assert compute_recall_at_k(retrieved, relevant, k=3) == 0.5
    assert compute_precision_at_k(retrieved, relevant, k=3) == 1 / 3
    assert compute_mrr(retrieved, relevant, k=3) == 0.5
    assert compute_ndcg_at_k(retrieved, relevant, k=3) > 0.35


def test_generate_synthetic_dataset_and_load_dataset_from_disk():
    dataset = build_synthetic_evaluation_dataset()
    assert len(dataset) >= 3
    assert all(example.query for example in dataset)

    loaded = load_synthetic_evaluation_dataset("data/evaluation/synthetic_corpus.json")
    assert len(loaded) == len(dataset)
    assert loaded[0].query == dataset[0].query


def test_evaluate_dataset_reports_averaged_metrics():
    dataset = build_synthetic_evaluation_dataset()
    ranked_results = [
        ["doc_attention", "noise-1", "noise-2"],
        ["noise-1", "doc_rag", "noise-2"],
        ["noise-1", "noise-2", "doc_ml_basics"],
    ]

    report = evaluate_dataset(dataset, ranked_results, config=EvaluationConfig(top_k=3))

    assert set(report.metrics) == {"recall@k", "precision@k", "mrr", "ndcg@k"}
    assert report.metrics["recall@k"] >= 0.0
    assert report.metrics["mrr"] >= 0.0


def test_measure_latency_reports_elapsed_time_for_callable():
    def slow_add(left: int, right: int) -> int:
        return left + right

    result = measure_latency(slow_add, 2, 3)
    assert result["result"] == 5
    assert result["latency_ms"] >= 0.0


def test_aliases_still_expose_common_function_names():
    metrics = evaluate_retrieval(["doc-3", "doc-1"], ["doc-1"], top_k=2)
    assert metrics["recall@k"] == 1.0
