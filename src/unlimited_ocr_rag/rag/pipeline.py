from __future__ import annotations

import logging

try:  # langchain < 1.0 (as in the article)
    from langchain.chains import RetrievalQA
except ImportError:  # langchain >= 1.0 moved legacy chains into langchain-classic
    from langchain_classic.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import Settings
from ..ocr.base import OCRBackend
from ..ocr.factory import build_ocr_backend
from .llm import build_embeddings, build_llm

logger = logging.getLogger(__name__)


class SimpleRAG:
    """
    Combine Unlimited-OCR text extraction with an LLM for querying documents
    """

    def __init__(
        self,
        settings: Settings | None = None,
        ocr_backend: OCRBackend | None = None,
        embeddings=None,
        llm=None,
    ):
        self.settings = settings or Settings.from_env()
        # Lazily-constructed heavy pieces; injectable for tests.
        self._ocr = ocr_backend
        self._embeddings = embeddings
        self._llm = llm
        self.vectorstore: FAISS | None = None
        self.qa_chain: RetrievalQA | None = None

    # --- lazy accessors -------------------------------------------------
    @property
    def ocr(self) -> OCRBackend:
        if self._ocr is None:
            self._ocr = build_ocr_backend(self.settings)
        return self._ocr

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = build_embeddings(self.settings)
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    # --- OCR -> Documents ----------------------------------------------
    def extract_documents(self, file_paths: list[str]) -> list[Document]:
        docs: list[Document] = []
        for path in file_paths:
            logger.info("OCR (%s): %s", self.ocr.name, path)
            text = self.ocr.extract(path)
            if not text.strip():
                text = "No text found"
            docs.append(Document(page_content=text, metadata={"source": path}))
        return docs

    # --- indexing -------------------------------------------------------
    def build_index(self, file_paths: list[str]) -> "SimpleRAG":
        """
        Extract, chunk, embed into FAISS, and build the RetrievalQA chain
        """
        docs = self.extract_documents(file_paths)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        splits = splitter.split_documents(docs)
        if not splits:
            raise ValueError("No text was extracted from the provided files.")

        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        self._build_chain()
        logger.info("Indexed %d chunks from %d file(s).", len(splits), len(file_paths))
        return self

    def _build_chain(self) -> None:
        if self.vectorstore is None:
            raise RuntimeError("Vector store is not built yet.")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.settings.retrieval_k}
            ),
            return_source_documents=True,
        )

    # --- persistence ----------------------------------------------------
    def save_index(self, path: str) -> None:
        if self.vectorstore is None:
            raise RuntimeError("Nothing to save; build the index first.")
        self.vectorstore.save_local(path)

    def load_index(self, path: str) -> "SimpleRAG":
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        self._build_chain()
        return self

    # --- querying -------------------------------------------------------
    def query(self, question: str) -> dict:
        if self.qa_chain is None:
            raise RuntimeError("Index not built. Call build_index(...) or load_index(...) first.")
        result = self.qa_chain.invoke({"query": question})
        sources = []
        for d in result.get("source_documents", []) or []:
            src = d.metadata.get("source")
            if src and src not in sources:
                sources.append(src)
        return {"answer": result.get("result", ""), "sources": sources}
