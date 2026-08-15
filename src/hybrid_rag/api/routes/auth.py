from fastapi import APIRouter

from hybrid_rag.config.settings import get_settings

router = APIRouter(tags=["auth"])


@router.get("/auth/status")
async def auth_status() -> dict:
    """Shows whether API keys are required (not the key values)."""
    settings = get_settings()
    return {
        "ingest_key_required": bool(settings.ingest_api_key),
        "query_key_required": bool(settings.query_api_key),
        "header_name": "X-API-Key",
    }
