from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from hybrid_rag.config.settings import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_ingest_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    _verify_key(api_key, get_settings().ingest_api_key, "INGEST_API_KEY")


def verify_query_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    _verify_key(api_key, get_settings().query_api_key, "QUERY_API_KEY")


def _verify_key(provided: str | None, expected: str, env_name: str) -> None:
    expected = (expected or "").strip()
    if not expected:
        return

    provided = _normalize_provided_key(provided)
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"Invalid or missing API key. Send header X-API-Key matching {env_name} in .env"
            ),
        )


def _normalize_provided_key(provided: str | None) -> str:
    if not provided:
        return ""
    value = provided.strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value
