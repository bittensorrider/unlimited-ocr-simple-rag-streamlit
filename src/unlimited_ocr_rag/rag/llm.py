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
    Return HuggingFace BGE embeddings (article default: BAAI/bge-small-en-v1.5)
    """
    try:  # preferred standalone package
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:  # fallback to the community implementation
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
