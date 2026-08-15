from langchain_core.documents import Document

from hybrid_rag.config.settings import Settings, get_settings
from hybrid_rag.retrieval.dense_retriever import dense_search


def naive_search(
    query: str,
    k: int | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    cfg = settings or get_settings()
    return dense_search(query, k=k or cfg.naive_k, company=None, settings=cfg)
