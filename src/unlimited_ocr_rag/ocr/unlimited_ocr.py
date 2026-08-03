from __future__ import annotations

import os
import tempfile

from .base import OCRBackend

# Prompts / params straight from the model card.
PROMPT_MULTI = "<image>Multi page parsing."
PROMPT_SINGLE = "<image>document parsing."
MAX_LENGTH = 32768
NO_REPEAT_NGRAM_SIZE = 35


def cuda_available() -> bool:
    """True if torch is importable and reports a CUDA device"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class UnlimitedOCRBackend(OCRBackend):
    name = "unlimited-ocr"

    def __init__(self, model_name: str = "baidu/Unlimited-OCR", dpi: int = 300):
        self.model_name = model_name
        self.dpi = dpi
        self._model = None
        self._tokenizer = None

    # --- lazy model loading (heavy; only happens on first extract) ---
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        self._model = model.eval().cuda()

    def _read_output_text(self, output_path: str) -> str:
        """Read all .md / .txt files written by save_results=True."""
        texts: list[str] = []
        for fname in sorted(os.listdir(output_path)):
            if fname.endswith((".md", ".txt")):
                with open(os.path.join(output_path, fname), "r", encoding="utf-8") as f:
                    texts.append(f.read())
        return "\n\n".join(texts)

    def extract(self, file_path: str) -> str:
        from ..utils.pdf import pdf_to_images

        self._ensure_loaded()
        tmp_out = tempfile.mkdtemp(prefix="ocr_out_")

        if self.is_pdf(file_path):
            pages = pdf_to_images(file_path, dpi=self.dpi)
            self._model.infer_multi(
                self._tokenizer,
                prompt=PROMPT_MULTI,
                image_files=pages,
                output_path=tmp_out,
                image_size=1024,
                max_length=MAX_LENGTH,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                ngram_window=1024,
                save_results=True,
            )
        else:
            self._model.infer(
                self._tokenizer,
                prompt=PROMPT_SINGLE,
                image_file=file_path,
                output_path=tmp_out,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                max_length=MAX_LENGTH,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                ngram_window=128,
                save_results=True,
            )

        text = self._read_output_text(tmp_out)
        return text if text.strip() else "No text found"
