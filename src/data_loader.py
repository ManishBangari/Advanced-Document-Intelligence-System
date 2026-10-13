import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, JSONLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".docx": "docx",
    ".json": "json",
}


def _safe_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_source_name(value: Any, fallback: str) -> str:
    normalized = _safe_str(value)
    if not normalized:
        return fallback
    return Path(normalized).name or normalized


def compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_page_number(page_value: Any) -> Optional[int]:
    if page_value is None:
        return None
    try:
        page = int(page_value)
    except (TypeError, ValueError):
        return None
    if page < 0:
        return None
    return page + 1


def build_document_id(file_path: Path, file_hash: str) -> str:
    source_name = file_path.name
    return hashlib.sha256(f"{source_name}:{file_hash}".encode("utf-8")).hexdigest()[:16]


def _normalize_document_metadata(file_path: Path, file_hash: str, original_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata = dict(original_metadata or {})
    source = _safe_source_name(metadata.get("source"), file_path.name)
    file_type = metadata.get("file_type") or SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "unknown")
    document_id = metadata.get("document_id") or build_document_id(file_path, file_hash)
    metadata.update(
        {
            "document_id": document_id,
            "source": source,
            "file_type": file_type,
            "file_hash": file_hash,
            "page": normalize_page_number(metadata.get("page")),
        }
    )
    if metadata.get("page") is None:
        metadata.pop("page", None)
    return metadata


def _load_single_file(file_path: Path):
    suffix = file_path.suffix.lower()
    loader_map = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
        ".docx": Docx2txtLoader,
        ".json": JSONLoader,
    }

    if suffix not in loader_map:
        return []

    loader_cls = loader_map[suffix]
    doc_hash = compute_file_sha256(file_path)
    document_id = build_document_id(file_path, doc_hash)

    try:
        loader = loader_cls(str(file_path))
        loaded = loader.load()
    except Exception as exc:  # pragma: no cover - logging only
        print(f"[ERROR] Failed to load {file_path}: {exc}")
        return []

    documents = []
    for index, document in enumerate(loaded):
        content = (document.page_content or "").strip()
        if not content:
            print(f"[WARN] Empty document content skipped for {file_path}")
            continue
        document.metadata = _normalize_document_metadata(file_path, doc_hash, document.metadata)
        document.metadata["document_id"] = document.metadata.get("document_id") or document_id
        document.metadata["source"] = _safe_source_name(document.metadata.get("source"), file_path.name)
        document.metadata["file_type"] = document.metadata.get("file_type") or SUPPORTED_EXTENSIONS.get(suffix, "unknown")
        document.metadata["page"] = normalize_page_number(document.metadata.get("page"))
        if document.metadata.get("page") is None:
            document.metadata.pop("page", None)
        document.metadata["chunk_index"] = index
        documents.append(document)

    return documents


def detect_duplicate_hashes(documents: Iterable[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for document in documents:
        file_hash = (document.metadata or {}).get("file_hash")
        if file_hash:
            grouped[file_hash].append(document)
    return {file_hash: items for file_hash, items in grouped.items() if len(items) > 1}


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON.
    """
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    documents: List[Any] = []

    print(f"[DEBUG] Data path: {data_path}")
    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            print(f"[WARN] Unsupported file type skipped: {file_path}")
            continue

        file_hash = compute_file_sha256(file_path)
        try:
            loaded_documents = _load_single_file(file_path)
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[ERROR] Failed to process {file_path}: {exc}")
            continue

        if not loaded_documents:
            print(f"[WARN] No usable content loaded from {file_path}")
            continue

        for document in loaded_documents:
            document.metadata["duplicate_hash"] = False
            document.metadata["file_hash"] = file_hash
            document.metadata["document_id"] = document.metadata.get("document_id") or build_document_id(file_path, file_hash)
            documents.append(document)

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents


if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)