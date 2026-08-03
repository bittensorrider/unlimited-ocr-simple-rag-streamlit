"""
Streamlit chat UI for Unlimited OCR + RAG

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from unlimited_ocr_rag import Settings, SimpleRAG
from unlimited_ocr_rag.ocr.factory import build_ocr_backend

st.set_page_config(page_title="Unlimited OCR + RAG", page_icon="📄", layout="wide")


def _save_uploads(uploaded_files) -> list[str]:
    tmp_dir = tempfile.mkdtemp(prefix="uocr_uploads_")
    paths = []
    for uf in uploaded_files:
        path = os.path.join(tmp_dir, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        paths.append(path)
    return paths


def main() -> None:
    st.title("📄 Unlimited OCR + RAG")
    st.caption(
        "Upload documents, extract text with Baidu Unlimited-OCR (or the CPU fallback), "
        "and ask questions over them."
    )

    settings = Settings.from_env()

    with st.sidebar:
        st.header("Setup")
        backend = build_ocr_backend(settings)
        st.write(f"**OCR backend:** `{backend.name}`")
        st.write(f"**LLM:** `{settings.llm_provider}` / `{settings.llm_model}`")
        st.write(f"**Embeddings:** `{settings.embedding_model}`")
        if settings.llm_provider == "openai" and not settings.openai_api_key:
            st.warning("OPENAI_API_KEY is not set — answering will fail. Add it to your .env.")

        uploaded = st.file_uploader(
            "Upload images or PDFs",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "pdf"],
            accept_multiple_files=True,
        )
        if st.button("Index documents", type="primary", disabled=not uploaded):
            paths = _save_uploads(uploaded)
            with st.spinner(f"Running OCR ({backend.name}) and building the index..."):
                rag = SimpleRAG(settings=settings, ocr_backend=backend)
                rag.build_index(paths)
            st.session_state.rag = rag
            st.session_state.messages = []
            st.success(f"Indexed {len(paths)} document(s).")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    ready = "rag" in st.session_state
    prompt = st.chat_input("Ask a question about your documents..." if ready
                           else "Upload and index documents first.")
    if prompt and ready:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = st.session_state.rag.query(prompt)
            answer = res["answer"]
            if res.get("sources"):
                answer += "\n\n---\n*Sources: " + ", ".join(res["sources"]) + "*"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
