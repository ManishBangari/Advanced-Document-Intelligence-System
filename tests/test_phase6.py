from src.information_extractor import extract_document_fields, extract_fields


def build_document(source, content, page=1, chunk_id="chunk-1"):
    return {
        "metadata": {
            "source": source,
            "file_type": "txt",
            "page": page,
            "chunk_id": chunk_id,
        },
        "page_content": content,
    }


def test_extractor_populates_known_fields_and_citations():
    doc = build_document(
        "agreement.txt",
        "Contract ID: C-2042. Amount: $1,250.50 USD. Effective date: 2024-09-01. Status: active.",
    )

    result = extract_document_fields(doc)

    assert result.values["contract_id"] == "C-2042"
    assert result.values["amount"] == 1250.5
    assert result.values["effective_date"] == "2024-09-01"
    assert result.values["status"] == "active"
    assert result.evidence[0].source == "agreement.txt"
    assert result.evidence[0].page == 1


def test_missing_required_fields_are_flagged_without_guessing():
    doc = build_document("agreement.txt", "This document describes a policy but no amount is listed.")

    result = extract_document_fields(doc, schema={
        "contract_id": {"type": "string", "required": True},
        "amount": {"type": "number", "required": True},
    })

    assert "contract_id" in result.missing_fields
    assert "amount" in result.missing_fields
    assert result.values == {}


def test_custom_schema_supports_user_defined_fields_and_json_output():
    doc = build_document("invoice.txt", "Invoice number INV-77. Currency: GBP. Total: £900.00")

    schema = {
        "invoice_number": {"type": "string", "required": True},
        "currency": {"type": "string", "required": True},
        "total": {"type": "number", "required": True},
    }

    result = extract_document_fields(doc, schema=schema)

    assert result.values["invoice_number"] == "INV-77"
    assert result.values["currency"] == "GBP"
    assert result.values["total"] == 900.0
    assert "invoice_number" in result.to_json()


def test_extraction_handles_multiple_documents_and_preserves_source_reference():
    docs = [
        build_document("a.txt", "Status: pending."),
        build_document("b.txt", "Status: expired."),
    ]

    result = extract_fields(docs, schema={
        "status": {"type": "enum", "values": ["active", "pending", "expired"], "required": True},
    })

    assert result.values["status"] == "pending"
    assert result.evidence[0].source == "a.txt"
    assert len(result.evidence) >= 1
