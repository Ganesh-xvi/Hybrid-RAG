from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from hybrid_rag.config.settings import Settings, get_settings


@lru_cache
def _get_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(
    query: str,
    docs: list[Document],
    top_n: int | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    if not docs:
        return []
    cfg = settings or get_settings()
    n = top_n or cfg.rerank_top_n
    model = _get_cross_encoder(cfg.reranker_model)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(docs, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:n]]


def reorder_for_attention(docs: list[Document]) -> list[Document]:
    """Place most relevant chunk first and last (U-shaped attention mitigation)."""
    if len(docs) <= 2:
        return docs
    first = docs[0]
    middle = docs[1:-1] if len(docs) > 2 else []
    last = docs[0]
    return [first, *middle, last]
