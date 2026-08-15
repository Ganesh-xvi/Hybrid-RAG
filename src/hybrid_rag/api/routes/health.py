import httpx
from fastapi import APIRouter

from hybrid_rag.api.schemas import HealthResponse
from hybrid_rag.cache.semantic_cache import SemanticCache
from hybrid_rag.config.settings import get_settings
from hybrid_rag.retrieval.qdrant_store import QdrantStore
from hybrid_rag.utils.qdrant import ping_qdrant

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    redis_ok = SemanticCache(settings).ping()
    qdrant_ok = ping_qdrant()
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = response.status_code == 200
    except Exception:
        ollama_ok = False

    points = QdrantStore(settings).point_count()
    all_ok = redis_ok and qdrant_ok and ollama_ok
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        redis=redis_ok,
        qdrant=qdrant_ok,
        ollama=ollama_ok,
        collection_points=points,
    )
