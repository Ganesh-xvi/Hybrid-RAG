from functools import lru_cache

from hybrid_rag.config.settings import get_settings
from hybrid_rag.core.rag_chain import RAGChain
from hybrid_rag.ingestion.job_manager import IngestJobManager


@lru_cache
def get_rag_chain() -> RAGChain:
    return RAGChain(get_settings())


@lru_cache
def get_ingest_job_manager() -> IngestJobManager:
    return IngestJobManager(get_settings())
