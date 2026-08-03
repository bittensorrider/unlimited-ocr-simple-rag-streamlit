from __future__ import annotations

import logging

from ..config import Settings, settings as default_settings
from .base import OCRBackend
from .fallback import FallbackOCRBackend
from .unlimited_ocr import UnlimitedOCRBackend, cuda_available

logger = logging.getLogger(__name__)


def build_ocr_backend(settings: Settings | None = None) -> OCRBackend:
    cfg = settings or default_settings
    choice = cfg.ocr_backend

    if choice == "unlimited":
        return UnlimitedOCRBackend(model_name=cfg.ocr_model, dpi=cfg.ocr_dpi)
    if choice == "fallback":
        return FallbackOCRBackend(dpi=cfg.ocr_dpi)

    # auto
    if cuda_available():
        logger.info("CUDA detected -> using Unlimited-OCR (%s)", cfg.ocr_model)
        return UnlimitedOCRBackend(model_name=cfg.ocr_model, dpi=cfg.ocr_dpi)
    logger.warning(
        "No CUDA GPU detected -> using CPU fallback OCR. "
        "For full Unlimited-OCR quality, run on a CUDA machine with OCR_BACKEND=unlimited."
    )
    return FallbackOCRBackend(dpi=cfg.ocr_dpi)
