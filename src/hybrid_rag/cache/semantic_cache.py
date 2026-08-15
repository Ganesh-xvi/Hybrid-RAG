import json
import uuid
from typing import Any

import numpy as np
import redis

from hybrid_rag.config.settings import Settings, get_settings
from hybrid_rag.utils.embeddings import embed_query


class SemanticCache:
    INDEX_KEY = "cache:index"
    ENTRY_PREFIX = "cache:entry:"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._redis = redis.from_url(self.settings.redis_url, decode_responses=True)

    def get(self, query: str) -> dict[str, Any] | None:
        if not self.settings.cache_enabled:
            return None

        query_embedding = np.array(embed_query(query, self.settings), dtype=np.float32)
        query_embedding /= np.linalg.norm(query_embedding) + 1e-10

        for entry_id in self._redis.smembers(self.INDEX_KEY):
            raw = self._redis.get(f"{self.ENTRY_PREFIX}{entry_id}")
            if not raw:
                self._redis.srem(self.INDEX_KEY, entry_id)
                continue
            entry = json.loads(raw)
            cached_embedding = np.array(entry["embedding"], dtype=np.float32)
            cached_embedding /= np.linalg.norm(cached_embedding) + 1e-10
            similarity = float(np.dot(query_embedding, cached_embedding))
            if similarity >= self.settings.cache_similarity_threshold:
                return {
                    "answer": entry["answer"],
                    "sources": entry.get("sources", []),
                    "cache_hit": True,
                    "similarity": similarity,
                }
        return None

    def set(self, query: str, answer: str, sources: list[dict[str, Any]]) -> None:
        if not self.settings.cache_enabled:
            return

        entry_id = str(uuid.uuid4())
        embedding = embed_query(query, self.settings)
        payload = {
            "query": query,
            "embedding": embedding,
            "answer": answer,
            "sources": sources,
        }
        self._redis.setex(
            f"{self.ENTRY_PREFIX}{entry_id}",
            self.settings.cache_ttl_seconds,
            json.dumps(payload),
        )
        self._redis.sadd(self.INDEX_KEY, entry_id)

    def ping(self) -> bool:
        try:
            return bool(self._redis.ping())
        except Exception:
            return False
