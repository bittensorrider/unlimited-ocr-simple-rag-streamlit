from __future__ import annotations

import hashlib

import pytest
from langchain_core.embeddings import Embeddings

from unlimited_ocr_rag.ocr.base import OCRBackend


class StubOCRBackend(OCRBackend):
    name = "stub"

    def __init__(self, texts: dict[str, str] | None = None, default: str = "No text found"):
        self.texts = texts or {}
        self.default = default

    def extract(self, file_path: str) -> str:
        return self.texts.get(file_path, self.default)


class FakeEmbeddings(Embeddings):
    dim = 64

    def _vec(self, text: str) -> list[float]:
        out = [0.0] * self.dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[h % self.dim] += 1.0
        norm = sum(v * v for v in out) ** 0.5 or 1.0
        return [v / norm for v in out]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def stub_ocr():
    return StubOCRBackend


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def fake_llm():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return FakeListChatModel(responses=["The document mentions revenue and a summary table."])
