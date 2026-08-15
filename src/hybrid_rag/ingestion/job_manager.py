import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import redis

from hybrid_rag.config.settings import Settings, get_settings


class IngestStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestJobManager:
    KEY_PREFIX = "ingest:job:"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._redis = redis.from_url(self.settings.redis_url, decode_responses=True)

    def _key(self, job_id: str) -> str:
        return f"{self.KEY_PREFIX}{job_id}"

    def create_job(self, pdfs_total: int = 0) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        payload = {
            "job_id": job_id,
            "status": IngestStatus.PENDING,
            "progress": 0,
            "stage": "queued",
            "pdfs_total": pdfs_total,
            "pdfs_processed": 0,
            "chunks_upserted": 0,
            "dlq_count": 0,
            "started_at": now,
            "updated_at": now,
        }
        self._save(job_id, payload)
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        raw = self._redis.get(self._key(job_id))
        if not raw:
            return None
        return json.loads(raw)

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(f"Ingest job not found: {job_id}")
        job.update(fields)
        job["updated_at"] = datetime.now(UTC).isoformat()
        self._save(job_id, job)
        return job

    def mark_running(self, job_id: str, stage: str, progress: int = 0) -> None:
        self.update_job(
            job_id,
            status=IngestStatus.RUNNING,
            stage=stage,
            progress=progress,
        )

    def mark_completed(
        self,
        job_id: str,
        pdfs: int,
        chunks: int,
        dlq_count: int,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self.update_job(
            job_id,
            status=IngestStatus.COMPLETED,
            progress=100,
            stage="completed",
            pdfs=pdfs,
            chunks=chunks,
            dlq_count=dlq_count,
            completed_at=now,
        )

    def mark_failed(self, job_id: str, error: str) -> None:
        now = datetime.now(UTC).isoformat()
        self.update_job(
            job_id,
            status=IngestStatus.FAILED,
            stage="failed",
            error=error,
            failed_at=now,
        )

    def _save(self, job_id: str, payload: dict[str, Any]) -> None:
        self._redis.setex(
            self._key(job_id),
            self.settings.ingest_job_ttl_seconds,
            json.dumps(payload, default=str),
        )
