from langchain_core.documents import Document

from hybrid_rag.config.settings import Settings, get_settings
from hybrid_rag.retrieval.prefilter import detect_company_filter
from hybrid_rag.retrieval.qdrant_store import QdrantStore


def hybrid_search(
    query: str,
    k: int | None = None,
    use_prefilter: bool = True,
    settings: Settings | None = None,
) -> list[Document]:
    cfg = settings or get_settings()
    company = detect_company_filter(query, cfg) if use_prefilter else None
    return QdrantStore(cfg).hybrid_search(query, k=k, company=company)
