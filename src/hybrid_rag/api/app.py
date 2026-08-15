from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from hybrid_rag.api.middleware.rate_limiter import limiter
from hybrid_rag.api.routes import auth, health, ingest, query
from hybrid_rag.config.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Hybrid RAG API",
        description="Employment contracts Hybrid RAG (Qdrant dense + sparse)",
        version="0.1.0",
    )
    application.state.limiter = limiter
    application.state.settings = settings
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    application.include_router(auth.router)
    application.include_router(query.router)
    application.include_router(health.router)
    application.include_router(ingest.router)
    return application


app = create_app()
