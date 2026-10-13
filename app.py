from __future__ import annotations

import uuid

import streamlit as st

from policypilot.config import Settings
from policypilot.errors import friendly_service_error
from policypilot.history import ChatHistory
from policypilot.pdf_reader import PDFExtractionError, extract_pdf_pages
from policypilot.pipeline import RAGPipeline
from policypilot.services import OpenAIService, PineconeStore


st.set_page_config(page_title="PolicyPilot", page_icon="📚", layout="wide")
st.title("📚 PolicyPilot")
st.caption("Ask grounded questions about your company policy PDFs.")
settings = Settings.from_env()
history = ChatHistory(settings.sqlite_path)
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def pipeline() -> RAGPipeline:
    settings.require_ingestion_credentials()
    ai = OpenAIService(settings)
    return RAGPipeline(settings, ai, ai, PineconeStore(settings), history)


with st.sidebar:
    st.header("Document library")
    uploads = st.file_uploader("Upload policy PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Process uploaded PDFs", disabled=not uploads):
        try:
            rag = pipeline()
            total = 0
            for upload in uploads or []:
                total += rag.index_pages(extract_pdf_pages(upload.getvalue(), upload.name))
            st.success(f"Indexed {total} chunks. You can ask questions now.")
        except (ValueError, PDFExtractionError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(friendly_service_error(exc))
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")
    if st.button("Start a new chat"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.caption("Credentials are read only from your local .env file.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            st.caption("Sources: " + "; ".join(message["citations"]))

if question := st.chat_input("Ask about refunds, returns, payments, or orders..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = pipeline().ask(st.session_state.conversation_id, question)
            st.markdown(result.text)
            citation_labels = sorted({f"{item.chunk.document_name}, p. {item.chunk.page_number}" for item in result.citations})
            if citation_labels:
                st.caption("Sources: " + "; ".join(citation_labels))
            st.session_state.messages.append({"role": "assistant", "content": result.text,
                                              "citations": citation_labels})
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(friendly_service_error(exc))
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")
