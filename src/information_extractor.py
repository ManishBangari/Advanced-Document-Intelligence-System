import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Type, Union, get_args

from pydantic import BaseModel, Field, ValidationError, create_model


class ExtractionEvidence(BaseModel):
    source: str
    file_type: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    text: str


class ExtractionResult(BaseModel):
    values: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[ExtractionEvidence] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2)


_DEFAULT_SCHEMA: Dict[str, Dict[str, Any]] = {
    "contract_id": {"type": "string", "required": True},
    "amount": {"type": "number", "required": True},
    "currency": {"type": "string", "required": False},
    "effective_date": {"type": "date", "required": False},
    "status": {"type": "enum", "values": ["active", "pending", "expired", "unknown"], "required": False},
}


class StructuredExtractor:
    def __init__(self, schema: Optional[Mapping[str, Any]] = None):
        self.schema = dict(schema or _DEFAULT_SCHEMA)

    def _field_hints(self, field_name: str) -> List[str]:
        aliases = {
            "contract_id": ["contract id", "contract number", "agreement no", "agreement number", "policy id"],
            "amount": ["amount", "total", "fee", "price", "value"],
            "currency": ["currency", "usd", "eur", "gbp", "inr", "cad", "aud", "jpy"],
            "effective_date": ["effective date", "start date", "issued on", "signed on", "valid from"],
            "status": ["status", "state", "condition", "validity"],
        }
        return aliases.get(field_name, [field_name.replace("_", " ")])

    def _coerce_field_value(self, field_name: str, raw_value: Any, field_type: str) -> Any:
        if raw_value is None:
            return None
        if not isinstance(raw_value, str):
            raw_value = str(raw_value)
        value = raw_value.strip()
        if not value:
            return None

        if field_type == "number":
            match = re.search(r"[-+]?(?:[$€£¥]|USD|EUR|GBP|INR|CAD|AUD|JPY)?\s*(\d[\d,]*(?:\.\d+)?)", value, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    return None
            cleaned = re.sub(r"[^0-9.\-]", "", value)
            try:
                return float(cleaned)
            except ValueError:
                return None

        if field_type == "integer":
            cleaned = re.sub(r"[^0-9\-]", "", value)
            try:
                return int(cleaned)
            except ValueError:
                return None

        if field_type == "date":
            for pattern in [r"\d{4}-\d{2}-\d{2}", r"\d{1,2}/\d{1,2}/\d{4}", r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}\b"]:
                match = re.search(pattern, value, flags=re.IGNORECASE)
                if match:
                    found = match.group(0)
                    try:
                        return datetime.strptime(found, "%Y-%m-%d").strftime("%Y-%m-%d")
                    except ValueError:
                        try:
                            return datetime.strptime(found, "%m/%d/%Y").strftime("%Y-%m-%d")
                        except ValueError:
                            try:
                                return datetime.strptime(found, "%b %d, %Y").strftime("%Y-%m-%d")
                            except ValueError:
                                return found
            return None

        if field_type == "boolean":
            lowered = value.lower()
            if lowered in {"true", "yes", "y", "required"}:
                return True
            if lowered in {"false", "no", "n", "optional"}:
                return False
            return None

        if field_type == "enum":
            lowered = value.lower()
            allowed = [item.lower() for item in (self.schema.get(field_name, {}).get("values") or [])]
            if lowered in allowed:
                return next(item for item in (self.schema.get(field_name, {}).get("values") or []) if item.lower() == lowered)
            return None

        return value

    def _extract_field_value(self, field_name: str, field_text: str, field_definition: Dict[str, Any]) -> Any:
        field_type = (field_definition.get("type") or "string").lower()
        hints = self._field_hints(field_name)
        patterns: List[str] = []

        for hint in hints:
            escaped = re.escape(hint)
            patterns.append(rf"(?i){escaped}\s*(?:[:=]|\s+)\s*(?:\"|')?([A-Za-z0-9][A-Za-z0-9\-/ ]{{0,50}})(?=\s*(?:[.;]|$))")
            patterns.append(rf"(?i){escaped}\s*[:=]\s*([^\n]+)")

        for pattern in patterns:
            match = re.search(pattern, field_text)
            if match:
                value = match.group(1).strip(" \t\r\n\"'")
                coerced = self._coerce_field_value(field_name, value, field_type)
                if coerced is not None:
                    return coerced

        if field_type == "enum":
            allowed = self.schema.get(field_name, {}).get("values") or []
            normalized = field_text.lower()
            for item in allowed:
                if item.lower() in normalized:
                    return item

        if field_type == "number":
            matches = re.findall(r"(?:\$|USD|EUR|GBP|INR|CAD|AUD|JPY)?\s*(\d[\d,]*\.?\d*)", field_text)
            if matches:
                value = matches[0].replace(",", "")
                return float(value)

        if field_type == "date":
            for pattern in [r"\d{4}-\d{2}-\d{2}", r"\d{1,2}/\d{1,2}/\d{4}", r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}\b"]:
                match = re.search(pattern, field_text, flags=re.IGNORECASE)
                if match:
                    return self._coerce_field_value(field_name, match.group(0), field_type)

        if field_type == "string":
            token = re.search(r"(?i)(?:contract\s*id|agreement\s*no|contract\s*number|policy\s*id)\s*[:=]\s*([A-Za-z0-9\-]+)", field_text)
            if token:
                return token.group(1)

        return None

    def _build_dynamic_model(self, schema: Mapping[str, Any]) -> Type[BaseModel]:
        field_definitions: Dict[str, Any] = {}
        for name, config in schema.items():
            field_type = (config.get("type") or "string").lower()
            python_type = str
            if field_type in {"number", "float"}:
                python_type = float
            elif field_type in {"integer", "int"}:
                python_type = int
            elif field_type == "boolean":
                python_type = bool
            elif field_type == "date":
                python_type = str
            elif field_type == "enum":
                python_type = str

            if config.get("required"):
                field_definitions[name] = (python_type, Field(...))
            else:
                field_definitions[name] = (Optional[python_type], Field(default=None))

        return create_model("ExtractedDocumentModel", **field_definitions)

    def extract(self, documents: Sequence[Any], schema: Optional[Mapping[str, Any]] = None) -> ExtractionResult:
        selected_schema = dict(schema or self.schema)
        dynamic_model = self._build_dynamic_model(selected_schema)
        extracted_values: Dict[str, Any] = {}
        evidence: List[ExtractionEvidence] = []
        missing_fields: List[str] = []

        for document in documents:
            metadata = dict(getattr(document, "metadata", {}) or {}) if not isinstance(document, dict) else dict(document.get("metadata") or {})
            if isinstance(document, dict):
                text = document.get("page_content") or document.get("text") or ""
            else:
                text = getattr(document, "page_content", "") or getattr(document, "text", "") or ""

            if not text:
                continue

            source = metadata.get("source") or "unknown-source"
            file_type = metadata.get("file_type")
            page = metadata.get("page")
            chunk_id = metadata.get("chunk_id")

            for field_name, field_definition in selected_schema.items():
                if field_name in extracted_values and extracted_values[field_name] is not None:
                    continue
                value = self._extract_field_value(field_name, text, field_definition)
                if value is not None:
                    extracted_values[field_name] = value
                    evidence.append(
                        ExtractionEvidence(
                            source=source,
                            file_type=file_type,
                            page=page,
                            chunk_id=chunk_id,
                            text=text,
                        )
                    )

        for field_name, field_definition in selected_schema.items():
            if field_definition.get("required") and field_name not in extracted_values:
                missing_fields.append(field_name)

        try:
            validated = dynamic_model.model_validate(extracted_values)
            normalized_values = validated.model_dump(exclude_none=True)
        except ValidationError:
            normalized_values = extracted_values

        return ExtractionResult(values=normalized_values, evidence=evidence, missing_fields=missing_fields)

    def extract_from_document(self, document: Any, schema: Optional[Mapping[str, Any]] = None) -> ExtractionResult:
        return self.extract([document], schema=schema)


def extract_fields(documents: Sequence[Any], schema: Optional[Mapping[str, Any]] = None) -> ExtractionResult:
    return StructuredExtractor(schema=schema).extract(documents)


def extract_document_fields(document: Any, schema: Optional[Mapping[str, Any]] = None) -> ExtractionResult:
    return StructuredExtractor(schema=schema).extract_from_document(document)
