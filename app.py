import os
from pathlib import Path
from typing import Any, List

import streamlit as st

from src.comparison import compare_document_pair, compare_documents
from src.data_loader import SUPPORTED_EXTENSIONS, load_all_documents
from src.information_extractor import StructuredExtractor
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "faiss_store"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)


@st.cache_resource
def get_search_service() -> RAGSearch:
    ensure_dirs()
    return RAGSearch(str(INDEX_DIR))


def list_supported_files() -> List[str]:
    ensure_dirs()
    if not DATA_DIR.exists():
        return []
    return sorted(
        [path.name for path in DATA_DIR.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    )


def rebuild_index_from_data() -> None:
    ensure_dirs()
    for filename in ["faiss.index", "metadata.pkl", "documents.db"]:
        target = INDEX_DIR / filename
        if target.exists():
            target.unlink()

    docs = load_all_documents(str(DATA_DIR))
    if docs:
        store = FaissVectorStore(str(INDEX_DIR))
        store.build_from_documents(docs)
    get_search_service.clear()


def load_documents_for_sources(sources: List[str]) -> List[Any]:
    if not sources:
        return []
    all_documents = load_all_documents(str(DATA_DIR))
    selected = []
    seen = set()
    for document in all_documents:
        metadata = getattr(document, "metadata", {}) or {}
        source_name = metadata.get("source")
        if source_name in sources and source_name not in seen:
            selected.append(document)
            seen.add(source_name)
    return selected


def render_dashboard() -> None:
    st.title("Advanced Document Intelligence System")
    st.caption("Local-first document intelligence with retrieval, comparison, and extraction workflows.")

    files = list_supported_files()
    backend_status = "configured" if os.getenv("GROQ_API_KEY") else "disabled"

    col1, col2, col3 = st.columns(3)
    col1.metric("Indexed documents", len(files))
    col2.metric("Search backend", "Groq" if backend_status == "configured" else "Local-only")
    col3.metric("Retrieval modes", "vector / bm25 / hybrid")

    st.subheader("Overview")
    if not files:
        st.info("No indexed files yet. Use the Document Management tab to upload supported documents and rebuild the local index.")
        return

    st.write("Indexed files:")
    for file_name in files:
        st.markdown(f"- {file_name}")


def render_document_management() -> None:
    st.subheader("Document Management")
    uploaded_files = st.file_uploader("Upload supported files", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            target = DATA_DIR / uploaded_file.name
            target.write_bytes(uploaded_file.getvalue())
        rebuild_index_from_data()
        st.success(f"Uploaded {len(uploaded_files)} file(s) and rebuilt the local index.")

    files = list_supported_files()
    if not files:
        st.info("The index is empty. Upload at least one supported file to begin.")
        return

    selected_file = st.selectbox("Select a file to remove", files)
    if st.button("Delete selected file"):
        target = DATA_DIR / selected_file
        if target.exists():
            target.unlink()
        rebuild_index_from_data()
        st.warning(f"Removed {selected_file} from the local dataset.")


def render_ask_documents() -> None:
    st.subheader("Ask Documents")
    if not list_supported_files():
        st.info("Upload some documents before asking questions.")
        return

    with st.form("ask_documents_form"):
        query = st.text_area("Question", placeholder="Ask about the indexed documents")
        retrieval_mode = st.selectbox("Retrieval mode", ["hybrid", "vector", "bm25"])
        top_k = st.slider("Top-k", min_value=1, max_value=10, value=5)
        submitted = st.form_submit_button("Search")

    if submitted and query.strip():
        search_service = get_search_service()
        results = search_service.search(query=query, top_k=top_k, retrieval_mode=retrieval_mode)
        response = search_service.answer_question(query=query, top_k=top_k, retrieval_mode=retrieval_mode)

        st.write(f"Retrieved {len(results)} relevant result(s).")
        st.markdown("### Answer")
        st.write(response.get("answer", "No answer available."))

        citations = response.get("citations") or []
        if citations:
            st.markdown("### Evidence")
            for idx, citation in enumerate(citations, start=1):
                source = citation.get("source") or "unknown-source"
                page = citation.get("page")
                location = f" (page {page})" if page is not None else ""
                st.markdown(f"**{idx}. {source}{location}**")
                st.write(citation.get("text", ""))


def render_compare_documents() -> None:
    st.subheader("Compare Documents")
    files = list_supported_files()
    if len(files) < 2:
        st.info("Upload at least two documents to compare them.")
        return

    left_file = st.selectbox("Left document", files, index=0)
    right_file = st.selectbox("Right document", files, index=1 if len(files) > 1 else 0)
    topic = st.text_input("Optional topic or field", placeholder="fee, policy, date, status")
    if st.button("Compare"):
        all_docs = load_all_documents(str(DATA_DIR))
        selected = [
            document for document in all_docs
            if (getattr(document, "metadata", {}) or {}).get("source") in {left_file, right_file}
        ]
        if len(selected) < 2:
            st.warning("Unable to compare the selected files because the source documents were not found in the local dataset.")
            return

        comparison_result = compare_document_pair(selected[0], selected[1], topic=topic or None)
        st.json(comparison_result)


def render_extract_information() -> None:
    st.subheader("Extract Information")
    files = list_supported_files()
    if not files:
        st.info("Upload a document before trying to extract structured fields.")
        return

    selected_file = st.selectbox("Document to inspect", files)
    schema = {
        "contract_id": {"type": "string", "required": True},
        "amount": {"type": "number", "required": True},
        "currency": {"type": "string", "required": False},
        "effective_date": {"type": "date", "required": False},
        "status": {"type": "enum", "values": ["active", "pending", "expired", "unknown"], "required": False},
    }

    if st.button("Extract fields"):
        documents = load_all_documents(str(DATA_DIR))
        matching = [document for document in documents if (getattr(document, "metadata", {}) or {}).get("source") == selected_file]
        if not matching:
            st.warning("The selected document could not be loaded for extraction.")
            return

        extractor = StructuredExtractor(schema=schema)
        result = extractor.extract(matching)
        st.json({
            "values": result.values,
            "missing_fields": result.missing_fields,
            "evidence": [
                {
                    "source": item.source,
                    "page": item.page,
                    "chunk_id": item.chunk_id,
                    "text": item.text,
                }
                for item in result.evidence
            ],
        })


def main() -> None:
    ensure_dirs()
    st.set_page_config(page_title="Document Intelligence", page_icon="📄", layout="wide")

    tabs = st.tabs(["Dashboard", "Document Management", "Ask Documents", "Compare Documents", "Extract Information"])
    with tabs[0]:
        render_dashboard()
    with tabs[1]:
        render_document_management()
    with tabs[2]:
        render_ask_documents()
    with tabs[3]:
        render_compare_documents()
    with tabs[4]:
        render_extract_information()


if __name__ == "__main__":
    main()