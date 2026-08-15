from fastapi import APIRouter, Depends, Request

from hybrid_rag.api.dependencies import get_rag_chain
from hybrid_rag.api.middleware.rate_limiter import limiter
from hybrid_rag.api.schemas import QueryRequest, QueryResponse, SourceItem
from hybrid_rag.api.security import verify_query_api_key
from hybrid_rag.config.settings import get_settings
from hybrid_rag.core.rag_chain import RAGChain

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
@limiter.limit(get_settings().rate_limit)
async def query(
    request: Request,
    body: QueryRequest,
    chain: RAGChain = Depends(get_rag_chain),
    _: None = Depends(verify_query_api_key),
) -> QueryResponse:
    result = await chain.ainvoke(body.query)
    return QueryResponse(
        answer=result.answer,
        sources=[SourceItem(**s) for s in result.sources],
        cache_hit=result.cache_hit,
        latency_ms=result.latency_ms,
    )
