from langchain_core.documents import Document

from hybrid_rag.config.settings import Settings, get_settings
from hybrid_rag.retrieval.qdrant_store import QdrantStore


def dense_search(
    query: str,
    k: int | None = None,
    company: str | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    return QdrantStore(settings).dense_search(query, k=k, company=company)
