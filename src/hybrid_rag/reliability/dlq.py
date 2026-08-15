import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

import redis

from hybrid_rag.config.settings import Settings, get_settings
from hybrid_rag.utils.logging import logger


class DLQManager:
    REDIS_KEY = "dlq:ingestion"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.storage_dir.mkdir(parents=True, exist_ok=True)
        self._redis = redis.from_url(self.settings.redis_url, decode_responses=True)
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.settings.dlq_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS failed_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    error TEXT NOT NULL,
                    retries INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )

    def push(self, payload: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        record = {**payload, "timestamp": now}
        try:
            self._redis.rpush(self.REDIS_KEY, json.dumps(record))
        except Exception as exc:
            logger.warning("Redis DLQ unavailable, using SQLite: %s", exc)
            with sqlite3.connect(self.settings.dlq_db_path) as conn:
                conn.execute(
                    "INSERT INTO failed_jobs (payload, error, created_at) VALUES (?, ?, ?)",
                    (json.dumps(record), payload.get("error", "unknown"), now),
                )

    def count(self) -> int:
        try:
            return int(self._redis.llen(self.REDIS_KEY))
        except Exception:
            with sqlite3.connect(self.settings.dlq_db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM failed_jobs").fetchone()
                return int(row[0]) if row else 0

    def list_failed(self) -> list[dict[str, Any]]:
        try:
            items = self._redis.lrange(self.REDIS_KEY, 0, -1)
            return [json.loads(item) for item in items]
        except Exception:
            with sqlite3.connect(self.settings.dlq_db_path) as conn:
                rows = conn.execute(
                    "SELECT payload, error, created_at FROM failed_jobs ORDER BY id DESC"
                ).fetchall()
            return [json.loads(row[0]) | {"error": row[1], "created_at": row[2]} for row in rows]

    def pop_for_retry(self) -> dict[str, Any] | None:
        try:
            raw = self._redis.lpop(self.REDIS_KEY)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        with sqlite3.connect(self.settings.dlq_db_path) as conn:
            row = conn.execute(
                "SELECT id, payload FROM failed_jobs ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            conn.execute("DELETE FROM failed_jobs WHERE id = ?", (row[0],))
            return json.loads(row[1])
