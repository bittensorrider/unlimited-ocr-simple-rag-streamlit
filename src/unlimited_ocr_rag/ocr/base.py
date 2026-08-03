from __future__ import annotations

from abc import ABC, abstractmethod

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
PDF_EXTS = (".pdf",)


class OCRBackend(ABC):
    """Extract text from an image or PDF file."""

    name: str = "base"

    @abstractmethod
    def extract(self, file_path: str) -> str:
        """Return extracted text (Markdown) for one image or PDF file."""

    @staticmethod
    def is_pdf(file_path: str) -> bool:
        return file_path.lower().endswith(PDF_EXTS)

    @staticmethod
    def is_image(file_path: str) -> bool:
        return file_path.lower().endswith(IMAGE_EXTS)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<OCRBackend {self.name}>"
