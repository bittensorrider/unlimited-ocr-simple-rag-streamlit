from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional: load a local .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _get(name: str, default: str) -> str:
    val = os.getenv(name)
    return default if val is None or val == "" else val


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: _get("LLM_MODEL", "gpt-4o"))
    llm_temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.0))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    # Embeddings
    # Provider: "fastembed" (ONNX BGE, torch-free, default) | "openai" | "huggingface"
    embedding_provider: str = field(
        default_factory=lambda: _get("EMBEDDING_PROVIDER", "fastembed").lower()
    )
    embedding_model: str = field(
        default_factory=lambda: _get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # OCR
    ocr_backend: str = field(default_factory=lambda: _get("OCR_BACKEND", "auto").lower())
    ocr_model: str = field(default_factory=lambda: _get("OCR_MODEL", "baidu/Unlimited-OCR"))
    ocr_dpi: int = field(default_factory=lambda: _get_int("OCR_DPI", 300))

    # Retrieval / chunking
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 200))
    retrieval_k: int = field(default_factory=lambda: _get_int("RETRIEVAL_K", 3))

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()


# Convenient module-level default; call Settings.from_env() for a fresh read.
settings = Settings.from_env()
