from __future__ import annotations

from ..config import Settings


def build_llm(settings: Settings):
    """
    Return a LangChain chat model. Faithful default: OpenAI GPT-4o, temperature 0
    """
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env, "
                "or set LLM_PROVIDER to a local provider."
            )
        return ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature)

    if provider in ("ollama", "local"):
        # Optional convenience path for a fully local setup.
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model=settings.llm_model, temperature=settings.llm_temperature)

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def build_embeddings(settings: Settings):
    """
    Return an embeddings object based on EMBEDDING_PROVIDER.

    * fastembed (default): BGE via ONNX Runtime -- torch-free, local, free. Keeps
      the article's BAAI/bge-small-en-v1.5 model without pulling PyTorch, so it
      installs cleanly where torch is unavailable/old (e.g. Intel macOS).
    * openai: OpenAIEmbeddings (needs OPENAI_API_KEY) -- no local model.
    * huggingface: the article's exact HuggingFaceEmbeddings path via
      sentence-transformers (requires the optional [hf] extra + a working torch).
    """
    provider = settings.embedding_provider.lower()

    if provider in ("fastembed", "onnx", "default"):
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=settings.embedding_model)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set.")
        model = settings.embedding_model
        if not model.startswith("text-embedding"):
            model = "text-embedding-3-small"  # BGE ids aren't OpenAI models
        return OpenAIEmbeddings(model=model)

    if provider in ("huggingface", "hf", "sentence-transformers"):
        try:  # preferred standalone package
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:  # fallback to the community implementation
            from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
