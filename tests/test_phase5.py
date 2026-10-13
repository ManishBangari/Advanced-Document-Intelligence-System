from src.comparison import compare_document_pair, compare_documents


def _doc(source, text, document_id=None, chunk_id=None, page=1):
    return {
        "metadata": {
            "document_id": document_id or source,
            "chunk_id": chunk_id or f"{source}-chunk",
            "source": source,
            "file_type": "txt",
            "page": page,
            "text": text,
        },
        "page_content": text,
    }


def test_identical_passages_are_not_reported_as_changes():
    left = _doc("doc_a", "The fee is 10 dollars.")
    right = _doc("doc_b", "The fee is 10 dollars.")

    result = compare_document_pair(left, right, topic="fee")

    assert result["status"] == "identical"
    assert result["change_type"] == "no_substantive_change"
    assert result["left"]["citation"]["source"] == "doc_a"
    assert result["right"]["citation"]["source"] == "doc_b"


def test_changed_values_are_detected_with_real_citations():
    left = _doc("doc_a", "The fee is 10 dollars.")
    right = _doc("doc_b", "The fee is 20 dollars.")

    result = compare_document_pair(left, right, topic="fee")

    assert result["status"] == "changed"
    assert result["change_type"] == "value_changed"
    assert result["left"]["citation"]["text"] == "The fee is 10 dollars."
    assert result["right"]["citation"]["text"] == "The fee is 20 dollars."


def test_missing_sections_are_reported_as_not_found_not_absence_proof():
    left = _doc("doc_a", "General project notes only.")
    right = _doc("doc_b", "The fee is 10 dollars.")

    result = compare_document_pair(left, right, topic="fee")

    assert result["status"] == "missing_left"
    assert result["change_type"] == "not_found_in_source"
    assert "not found" in result["uncertainty"].lower()


def test_ambiguous_comparisons_are_marked_uncertain():
    left = _doc("doc_a", "The policy may be reviewed later.")
    right = _doc("doc_b", "The policy could change next quarter.")

    result = compare_document_pair(left, right, topic="policy")

    assert result["status"] == "uncertain"
    assert result["change_type"] == "uncertain"


def test_pairwise_compare_documents_returns_multiple_comparison_rows():
    docs = [
        _doc("doc_a", "The fee is 10 dollars."),
        _doc("doc_b", "The fee is 20 dollars."),
        _doc("doc_c", "The fee is 20 dollars."),
    ]

    results = compare_documents(docs, topic="fee")

    assert len(results) == 2
    assert results[0]["change_type"] == "value_changed"
    assert results[1]["status"] == "identical"
