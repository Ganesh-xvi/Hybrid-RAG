import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from hybrid_rag.cache.semantic_cache import SemanticCache
from hybrid_rag.config.settings import Settings, get_settings
from hybrid_rag.core.prompts import build_prompt
from hybrid_rag.retrieval.hybrid_retriever import hybrid_search
from hybrid_rag.retrieval.naive import naive_search
from hybrid_rag.retrieval.reranker import reorder_for_attention, rerank


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False
    latency_ms: int = 0


class RAGChain:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache = SemanticCache(self.settings)
        self.llm = ChatGroq(
            api_key=self.settings.groq_api_key,
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
        )

    async def ainvoke(
        self,
        query: str,
        mode: str = "hybrid_prefilter",
    ) -> RAGResponse:
        start = time.perf_counter()

        cached = self.cache.get(query)
        if cached:
            return RAGResponse(
                answer=cached["answer"],
                sources=cached.get("sources", []),
                cache_hit=True,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )

        docs = self._retrieve(query, mode)
        reranked = rerank(query, docs, settings=self.settings)
        ordered = reorder_for_attention(reranked)
        messages = build_prompt(query, ordered)
        response = await self.llm.ainvoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        sources = self._extract_sources(ordered)

        self.cache.set(query, answer, sources)
        return RAGResponse(
            answer=answer,
            sources=sources,
            cache_hit=False,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    def _retrieve(self, query: str, mode: str) -> list[Document]:
        if mode == "naive":
            return naive_search(query, k=self.settings.naive_k, settings=self.settings)
        if mode == "hybrid":
            return hybrid_search(query, use_prefilter=False, settings=self.settings)
        return hybrid_search(query, use_prefilter=True, settings=self.settings)

    @staticmethod
    def _extract_sources(documents: list[Document]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen: set[str] = set()
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id", "")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            sources.append(
                {
                    "company": doc.metadata.get("company"),
                    "page": doc.metadata.get("page"),
                    "source": doc.metadata.get("source"),
                }
            )
        return sources
