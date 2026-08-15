from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class SourceItem(BaseModel):
    company: str | None = None
    page: int | None = None
    source: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    cache_hit: bool
    latency_ms: int


class IngestStartResponse(BaseModel):
    job_id: str
    status: str
    message: str


class IngestStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    stage: str | None = None
    pdfs_total: int | None = None
    pdfs_processed: int | None = None
    chunks_upserted: int | None = None
    pdfs: int | None = None
    chunks: int | None = None
    dlq_count: int | None = None
    error: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    redis: bool
    qdrant: bool
    ollama: bool
    collection_points: int
