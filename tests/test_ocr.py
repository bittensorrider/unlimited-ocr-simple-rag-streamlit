from __future__ import annotations

import importlib.util

import pytest

from unlimited_ocr_rag.config import Settings
from unlimited_ocr_rag.ocr.base import OCRBackend
from unlimited_ocr_rag.ocr.factory import build_ocr_backend
from unlimited_ocr_rag.ocr.fallback import FallbackOCRBackend
from unlimited_ocr_rag.ocr.unlimited_ocr import UnlimitedOCRBackend

HAS_FITZ = importlib.util.find_spec("fitz") is not None
HAS_RAPIDOCR = importlib.util.find_spec("rapidocr_onnxruntime") is not None
HAS_PIL = importlib.util.find_spec("PIL") is not None


def test_is_pdf_and_image_detection():
    assert OCRBackend.is_pdf("a.PDF")
    assert OCRBackend.is_image("b.PNG")
    assert not OCRBackend.is_image("b.pdf")


def test_factory_forces_fallback():
    b = build_ocr_backend(Settings(ocr_backend="fallback"))
    assert isinstance(b, FallbackOCRBackend)


def test_factory_forces_unlimited():
    b = build_ocr_backend(Settings(ocr_backend="unlimited"))
    assert isinstance(b, UnlimitedOCRBackend)


def test_factory_auto_without_gpu_is_fallback(monkeypatch):
    monkeypatch.setattr("unlimited_ocr_rag.ocr.factory.cuda_available", lambda: False)
    b = build_ocr_backend(Settings(ocr_backend="auto"))
    assert isinstance(b, FallbackOCRBackend)


@pytest.mark.skipif(not (HAS_FITZ and HAS_PIL), reason="needs pymupdf + pillow")
def test_pdf_to_images(tmp_path):
    import fitz

    from unlimited_ocr_rag.utils.pdf import pdf_to_images

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello Unlimited OCR")
    doc.save(str(pdf_path))
    doc.close()

    imgs = pdf_to_images(str(pdf_path), dpi=100, out_dir=str(tmp_path / "imgs"))
    assert len(imgs) == 1
    assert imgs[0].endswith(".png")


@pytest.mark.skipif(
    not (HAS_RAPIDOCR and HAS_FITZ and HAS_PIL),
    reason="needs rapidocr-onnxruntime + pymupdf + pillow",
)
def test_fallback_ocr_reads_text(tmp_path):
    from PIL import Image, ImageDraw

    img_path = tmp_path / "text.png"
    img = Image.new("RGB", (600, 160), "white")
    ImageDraw.Draw(img).text((20, 60), "INVOICE 2026", fill="black")
    img.save(img_path)

    text = FallbackOCRBackend().extract(str(img_path))
    assert isinstance(text, str) and text.strip()
