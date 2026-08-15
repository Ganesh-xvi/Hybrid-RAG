"""RAGAS metric computation for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybrid_rag.config.settings import Settings, get_settings
from scripts.evaluation.pipeline import EvalRunResult
from hybrid_rag.utils.embeddings import get_dense_embeddings
from hybrid_rag.utils.logging import logger


@dataclass
class RagasScores:
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
        }


def _require_ragas() -> Any:
    try:
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise ImportError(
            "RAGAS is required for evaluation. Install with:\n"
            "  uv sync --extra eval\n"
            "or: pip install -e '.[eval]'"
        ) from exc

    return {
        "evaluate": evaluate,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "metrics": [context_precision, context_recall, faithfulness, answer_relevancy],
    }


def compute_ragas_scores(
    results: list[EvalRunResult],
    settings: Settings | None = None,
) -> RagasScores:
    if not results:
        return RagasScores(0.0, 0.0, 0.0, 0.0)

    ragas = _require_ragas()
    cfg = settings or get_settings()

    from datasets import Dataset
    from langchain_groq import ChatGroq

    dataset = Dataset.from_dict(
        {
            "question": [r.question for r in results],
            "answer": [r.answer for r in results],
            "contexts": [r.contexts for r in results],
            "ground_truth": [r.ground_truth for r in results],
        }
    )

    llm = ragas["LangchainLLMWrapper"](
        ChatGroq(
            api_key=cfg.groq_api_key,
            model=cfg.llm_model,
            temperature=0,
        )
    )
    embeddings = ragas["LangchainEmbeddingsWrapper"](get_dense_embeddings(cfg))

    logger.info("Running RAGAS metrics on %s samples...", len(results))
    evaluation = ragas["evaluate"](
        dataset,
        metrics=ragas["metrics"],
        llm=llm,
        embeddings=embeddings,
    )

    df = evaluation.to_pandas()
    return RagasScores(
        context_precision=_mean_column(df, "context_precision"),
        context_recall=_mean_column(df, "context_recall"),
        faithfulness=_mean_column(df, "faithfulness"),
        answer_relevancy=_mean_column(df, "answer_relevancy"),
    )


def _mean_column(df: Any, column: str) -> float:
    if column not in df.columns:
        return 0.0
    series = df[column].dropna()
    if series.empty:
        return 0.0
    return float(series.mean())
