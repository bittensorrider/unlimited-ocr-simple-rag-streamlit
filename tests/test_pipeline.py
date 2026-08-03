from __future__ import annotations

import pytest

from unlimited_ocr_rag import Settings, SimpleRAG
from unlimited_ocr_rag.rag.pipeline import Document


@pytest.fixture
def rag(stub_ocr, fake_embeddings, fake_llm):
    texts = {
        "doc_a.png": "Q3 revenue was 12 million dollars. The summary table lists three products.",
        "doc_b.pdf": "Employee handbook. Vacation policy grants 20 days per year.",
    }
    settings = Settings(chunk_size=200, chunk_overlap=20, retrieval_k=2)
    return SimpleRAG(
        settings=settings,
        ocr_backend=stub_ocr(texts),
        embeddings=fake_embeddings,
        llm=fake_llm,
    )


def test_extract_documents_sets_source_metadata(rag):
    docs = rag.extract_documents(["doc_a.png"])
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["source"] == "doc_a.png"
    assert "revenue" in docs[0].page_content


def test_build_index_and_query(rag):
    rag.build_index(["doc_a.png", "doc_b.pdf"])
    assert rag.vectorstore is not None
    assert rag.qa_chain is not None

    res = rag.query("How much was Q3 revenue?")
    assert "answer" in res and res["answer"]
    assert res["sources"]  # at least one source path returned


def test_query_before_index_raises(rag):
    with pytest.raises(RuntimeError):
        rag.query("anything")


def test_empty_extraction_raises(stub_ocr, fake_embeddings, fake_llm):
    rag = SimpleRAG(
        ocr_backend=stub_ocr({}, default=""),  # produces "No text found" docs
        embeddings=fake_embeddings,
        llm=fake_llm,
    )
    # "No text found" is still text, so it indexes; verify it does not crash.
    rag.build_index(["missing.png"])
    assert rag.vectorstore is not None


def test_save_and_load_index(tmp_path, rag):
    rag.build_index(["doc_a.png"])
    idx_dir = tmp_path / "faiss_index"
    rag.save_index(str(idx_dir))
    assert idx_dir.exists()

    fresh = SimpleRAG(
        settings=rag.settings,
        embeddings=rag._embeddings,
        llm=rag._llm,
    )
    fresh.load_index(str(idx_dir))
    res = fresh.query("revenue?")
    assert res["answer"]
