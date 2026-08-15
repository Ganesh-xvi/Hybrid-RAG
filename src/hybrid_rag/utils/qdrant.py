from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from hybrid_rag.config.settings import Settings, get_settings


@lru_cache
def get_qdrant_client(url: str | None = None) -> QdrantClient:
    return QdrantClient(url=url or get_settings().qdrant_url)


@lru_cache
def get_sparse_model(model_name: str | None = None) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=model_name or get_settings().sparse_model)


def encode_sparse_documents(texts: list[str], settings: Settings | None = None) -> list[qmodels.SparseVector]:
    model = get_sparse_model(settings.sparse_model if settings else None)
    vectors = []
    for embedding in model.embed(texts):
        vectors.append(
            qmodels.SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            )
        )
    return vectors


def encode_sparse_query(query: str, settings: Settings | None = None) -> qmodels.SparseVector:
    model = get_sparse_model(settings.sparse_model if settings else None)
    embedding = next(model.query_embed(query))
    return qmodels.SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )


def ping_qdrant(client: QdrantClient | None = None) -> bool:
    try:
        (client or get_qdrant_client()).get_collections()
        return True
    except Exception:
        return False
