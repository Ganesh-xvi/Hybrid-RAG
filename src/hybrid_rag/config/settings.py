from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- API keys & models ---
    groq_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GROQ_API_KEY", "groq_api_key"),
    )
    llm_model: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias=AliasChoices("LLM_MODEL", "LLM_model", "llm_model"),
    )
    embedding_model: str = Field(
        default="snowflake-arctic-embed:137m",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model"),
    )
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    # --- API server ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_reload: bool = Field(default=True, alias="API_RELOAD")

    # --- Logging ---
    log_name: str = Field(default="hybrid_rag", alias="LOG_NAME")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/hybrid_rag.log", alias="LOG_FILE")
    log_to_console: bool = Field(default=True, alias="LOG_TO_CONSOLE")

    # --- Qdrant ---
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="employment_contracts", alias="QDRANT_COLLECTION")
    qdrant_dense_vector_name: str = Field(default="dense", alias="QDRANT_DENSE_VECTOR_NAME")
    qdrant_sparse_vector_name: str = Field(default="sparse", alias="QDRANT_SPARSE_VECTOR_NAME")
    qdrant_upsert_batch_size: int = Field(default=32, alias="QDRANT_UPSERT_BATCH_SIZE")

    # --- Redis ---
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # --- Paths ---
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    storage_dir: Path = Field(default=Path("./storage"), alias="STORAGE_DIR")

    # --- Retrieval ---
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")
    dense_k: int = Field(default=10, alias="DENSE_K")
    sparse_k: int = Field(default=10, alias="SPARSE_K")
    hybrid_k: int = Field(default=10, alias="HYBRID_K")
    naive_k: int = Field(default=5, alias="NAIVE_K")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL",
    )
    sparse_model: str = Field(default="Qdrant/bm25", alias="SPARSE_MODEL")
    company_keywords: list[str] = Field(
        default=["honeywell", "cloudflare"],
        alias="COMPANY_KEYWORDS",
    )

    # --- Semantic cache ---
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_similarity_threshold: float = Field(default=0.92, alias="CACHE_SIMILARITY_THRESHOLD")
    cache_ttl_seconds: int = Field(default=86400, alias="CACHE_TTL_SECONDS")

    # --- Rate limiting & security ---
    rate_limit: str = Field(default="10/minute", alias="RATE_LIMIT")
    ingest_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("INGEST_API_KEY", "ingest_api_key"),
    )
    query_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("QUERY_API_KEY", "query_api_key"),
    )

    # --- Reliability ---
    dlq_max_retries: int = Field(default=3, alias="DLQ_MAX_RETRIES")
    ingest_job_ttl_seconds: int = Field(default=86400, alias="INGEST_JOB_TTL_SECONDS")

    # --- Evaluation ---
    eval_concurrency: int = Field(default=5, alias="EVAL_CONCURRENCY")
    eval_output_path: Path = Field(default=Path("evaluation_results.md"), alias="EVAL_OUTPUT_PATH")

    @field_validator("ingest_api_key", "query_api_key", "groq_api_key", mode="before")
    @classmethod
    def strip_secrets(cls, value: str | None) -> str:
        if value is None:
            return ""
        return str(value).strip().strip("\ufeff").strip("\r")

    @field_validator("company_keywords", mode="before")
    @classmethod
    def parse_company_keywords(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return value

    @property
    def dlq_db_path(self) -> Path:
        return self.storage_dir / "failed_jobs.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
