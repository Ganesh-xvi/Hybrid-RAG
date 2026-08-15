import uvicorn

from hybrid_rag.config.settings import get_settings
from hybrid_rag.utils.logging import setup_logging


def main() -> None:
    settings = get_settings()
    setup_logging(settings)
    uvicorn.run(
        "hybrid_rag.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    main()
