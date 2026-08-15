from functools import lru_cache

from fastembed import SparseTextEmbedding

from hybrid_rag.config.settings import get_settings
from hybrid_rag.utils.qdrant import encode_sparse_documents, encode_sparse_query, get_sparse_model

__all__ = [
    "get_sparse_model",
    "encode_sparse_documents",
    "encode_sparse_query",
]


@lru_cache
def sparse_model_cached() -> SparseTextEmbedding:
    return get_sparse_model(get_settings().sparse_model)
