from __future__ import annotations

import os
import tempfile


def pdf_to_images(pdf_path: str, dpi: int = 300, out_dir: str | None = None) -> list[str]:
    import fitz  # PyMuPDF; imported lazily so the package imports without it

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths: list[str] = []
    try:
        for i, page in enumerate(doc):
            out = os.path.join(out_dir, f"page_{i + 1:04d}.png")
            page.get_pixmap(matrix=mat).save(out)
            paths.append(out)
    finally:
        doc.close()
    return paths
