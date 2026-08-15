from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from hybrid_rag.config.settings import Settings, get_settings


@lru_cache
def get_dense_embeddings() -> OllamaEmbeddings:
    cfg = get_settings()
    return OllamaEmbeddings(
        model=cfg.embedding_model,
        base_url=cfg.ollama_base_url,
    )


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    return get_dense_embeddings().embed_query(text)


def embed_documents(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    return get_dense_embeddings().embed_documents(texts)
