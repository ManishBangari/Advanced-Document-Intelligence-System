from src.qa_engine import QAEngine


class DummyLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return type("Result", (), {"content": "The answer is supported by the provided evidence."})()


def test_answer_question_returns_citations_for_real_evidence():
    results = [
        {
            "index": 0,
            "metadata": {
                "chunk_id": "chunk-001",
                "document_id": "doc-001",
                "source": "alpha.txt",
                "file_type": "txt",
                "page": 3,
                "text": "Alpha is a required field and must be present.",
            },
        }
    ]
    engine = QAEngine(llm=DummyLLM())

    response = engine.answer("What is required?", results)

    assert response["status"] == "answered"
    assert response["citations"][0]["chunk_id"] == "chunk-001"
    assert response["citations"][0]["source"] == "alpha.txt"
    assert response["citations"][0]["page"] == 3


def test_missing_page_numbers_are_reported_honestly():
    results = [
        {
            "index": 0,
            "metadata": {
                "chunk_id": "chunk-002",
                "document_id": "doc-002",
                "source": "beta.txt",
                "file_type": "txt",
                "text": "Beta is optional when there is no matching record.",
            },
        }
    ]
    engine = QAEngine(llm=DummyLLM())

    response = engine.answer("Is beta optional?", results)

    assert response["status"] == "answered"
    assert response["citations"][0]["page"] is None
    assert "beta.txt" in response["citations"][0]["source"]


def test_insufficient_evidence_returns_abstention():
    engine = QAEngine(llm=DummyLLM())

    response = engine.answer("What about a non-existent concept?", [])

    assert response["status"] == "insufficient_evidence"
    assert "insufficient" in response["answer"].lower()
    assert response["citations"] == []


def test_backend_unavailable_still_returns_evidence_only():
    results = [
        {
            "index": 1,
            "metadata": {
                "chunk_id": "chunk-003",
                "document_id": "doc-003",
                "source": "gamma.txt",
                "file_type": "txt",
                "text": "Gamma should be reviewed before shipping.",
            },
        }
    ]
    engine = QAEngine(llm=None)

    response = engine.answer("What should I review before shipping?", results)

    assert response["status"] == "backend_unavailable"
    assert response["citations"][0]["chunk_id"] == "chunk-003"
    assert "Retrieved evidence is available below" in response["answer"]
