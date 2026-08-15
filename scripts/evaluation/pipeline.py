"""Per-config RAG pipeline for evaluation (cache optional, mode-aware retrieval)."""

from __future__ import annotations

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

EVAL_MODES = ("naive", "hybrid_rerank", "hybrid_rerank_prefilter")


@dataclass
class EvalRunResult:
    question_id: int
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    mode: str
    cache_hit: bool = False
    latency_ms: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)


class EvaluationPipeline:
    def __init__(self, settings: Settings | None = None, use_cache: bool = False) -> None:
        self.settings = settings or get_settings()
        self.use_cache = use_cache
        self.cache = SemanticCache(self.settings) if use_cache else None
        self.llm = ChatGroq(
            api_key=self.settings.groq_api_key,
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
        )

    async def ainvoke(
        self,
        question_id: int,
        question: str,
        ground_truth: str,
        mode: str,
    ) -> EvalRunResult:
        if mode not in EVAL_MODES:
            raise ValueError(f"Unknown eval mode: {mode}. Choose from {EVAL_MODES}")

        start = time.perf_counter()

        if self.use_cache and self.cache:
            cached = self.cache.get(question)
            if cached:
                return EvalRunResult(
                    question_id=question_id,
                    question=question,
                    answer=cached["answer"],
                    contexts=[],
                    ground_truth=ground_truth,
                    mode=mode,
                    cache_hit=True,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    sources=cached.get("sources", []),
                )

        docs = self._retrieve(question, mode)
        if mode != "naive":
            docs = rerank(question, docs, settings=self.settings)
            docs = reorder_for_attention(docs)

        messages = build_prompt(question, docs)
        response = await self.llm.ainvoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        contexts = [doc.page_content for doc in docs]
        sources = self._extract_sources(docs)

        if self.use_cache and self.cache:
            self.cache.set(question, answer, sources)

        return EvalRunResult(
            question_id=question_id,
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            mode=mode,
            cache_hit=False,
            latency_ms=int((time.perf_counter() - start) * 1000),
            sources=sources,
        )

    def _retrieve(self, query: str, mode: str) -> list[Document]:
        if mode == "naive":
            return naive_search(query, settings=self.settings)
        if mode == "hybrid_rerank":
            return hybrid_search(query, use_prefilter=False, settings=self.settings)
        return hybrid_search(query, use_prefilter=True, settings=self.settings)

    @staticmethod
    def _extract_sources(documents: list[Document]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen: set[str] = set()
        for doc in documents:
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            if chunk_id and chunk_id in seen:
                continue
            if chunk_id:
                seen.add(chunk_id)
            sources.append(
                {
                    "company": doc.metadata.get("company"),
                    "page": doc.metadata.get("page"),
                    "source": doc.metadata.get("source"),
                }
            )
        return sources
