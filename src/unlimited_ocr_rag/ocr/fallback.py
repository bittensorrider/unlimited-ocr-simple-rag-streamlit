from __future__ import annotations

from .base import OCRBackend


class FallbackOCRBackend(OCRBackend):
    name = "fallback-rapidocr"

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self._engine = None

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:  # pragma: no cover - dependency guidance
            raise ImportError(
                "Fallback OCR needs rapidocr-onnxruntime. "
                "Install it with: pip install rapidocr-onnxruntime"
            ) from e
        self._engine = RapidOCR()

    def _ocr_image(self, image_path: str) -> str:
        self._ensure_loaded()
        result, _ = self._engine(image_path)
        if not result:
            return ""
        # result: list of [box, text, score]; already in reading order.
        return "\n".join(line[1] for line in result)

    def extract(self, file_path: str) -> str:
        if self.is_pdf(file_path):
            from ..utils.pdf import pdf_to_images

            pages = pdf_to_images(file_path, dpi=self.dpi)
            chunks = [self._ocr_image(p) for p in pages]
            text = "\n\n".join(c for c in chunks if c.strip())
        else:
            text = self._ocr_image(file_path)
        return text if text.strip() else "No text found"
