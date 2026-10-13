from src.search import HybridRetriever


class DummyVectorStore:
    def __init__(self, metadata, vector_results):
        self.metadata = metadata
        self._vector_results = vector_results

    def query(self, query_text, top_k=5):
        query = query_text.lower()
        if "alpha" in query:
            return [
                {"index": 0, "distance": 0.1, "metadata": self.metadata[0]},
                {"index": 2, "distance": 0.4, "metadata": self.metadata[2]},
            ]
        if "beta" in query:
            return [
                {"index": 0, "distance": 0.2, "metadata": self.metadata[0]},
                {"index": 1, "distance": 0.5, "metadata": self.metadata[1]},
            ]
        return []


def test_bm25_only_mode_prioritizes_exact_terms():
    metadata = [
        {"chunk_id": "c1", "text": "Alpha beta gamma"},
        {"chunk_id": "c2", "text": "Delta epsilon zeta"},
        {"chunk_id": "c3", "text": "Beta only"},
    ]
    retriever = HybridRetriever(DummyVectorStore(metadata, []))

    results = retriever.search("beta", top_k=2, retrieval_mode="bm25")

    assert results[0]["metadata"]["chunk_id"] == "c3"
    assert all(item["metadata"]["chunk_id"] in {"c3", "c1"} for item in results)


def test_vector_only_mode_returns_vector_candidates():
    metadata = [
        {"chunk_id": "c1", "text": "Alpha beta gamma"},
        {"chunk_id": "c2", "text": "Delta epsilon zeta"},
        {"chunk_id": "c3", "text": "Beta only"},
    ]
    retriever = HybridRetriever(DummyVectorStore(metadata, []))

    results = retriever.search("alpha", top_k=2, retrieval_mode="vector")

    assert [item["metadata"]["chunk_id"] for item in results] == ["c1", "c3"]


def test_hybrid_mode_fuses_results_without_duplicates():
    metadata = [
        {"chunk_id": "c1", "text": "Alpha beta gamma"},
        {"chunk_id": "c2", "text": "Delta epsilon zeta"},
        {"chunk_id": "c3", "text": "Beta only"},
    ]
    retriever = HybridRetriever(DummyVectorStore(metadata, []))

    results = retriever.search("beta", top_k=3, retrieval_mode="hybrid")

    chunk_ids = [item["metadata"]["chunk_id"] for item in results]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert chunk_ids[0] in {"c3", "c1"}


def test_empty_result_sets_are_handled_gracefully():
    retriever = HybridRetriever(DummyVectorStore([], []))

    assert retriever.search("nothing", top_k=3, retrieval_mode="hybrid") == []
    assert retriever.search("nothing", top_k=3, retrieval_mode="vector") == []
    assert retriever.search("nothing", top_k=3, retrieval_mode="bm25") == []
