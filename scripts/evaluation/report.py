"""Markdown report generation for evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from scripts.evaluation.metrics import RagasScores


@dataclass
class ConfigResult:
    name: str
    ragas: RagasScores
    avg_latency_ms: float
    cache_hit_rate: float
    question_count: int


def render_markdown(results: list[ConfigResult], questions_path: str) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Evaluation Results",
        "",
        f"_Generated: {now}_",
        "",
        f"**Questions file:** `{questions_path}`",
        "",
        "## Comparison",
        "",
        "| Config | context_precision | context_recall | faithfulness | "
        "answer_relevancy | avg_latency_ms | cache_hit_rate |",
        "|--------|-------------------|----------------|--------------|"
        "------------------|----------------|----------------|",
    ]

    for row in results:
        lines.append(
            f"| {row.name} "
            f"| {row.ragas.context_precision:.3f} "
            f"| {row.ragas.context_recall:.3f} "
            f"| {row.ragas.faithfulness:.3f} "
            f"| {row.ragas.answer_relevancy:.3f} "
            f"| {row.avg_latency_ms:.0f} "
            f"| {row.cache_hit_rate:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Config descriptions",
            "",
            "| Config | Retrieval | Rerank | Pre-filter |",
            "|--------|-----------|--------|------------|",
            "| `naive` | Dense only (Qdrant) | No | No |",
            "| `hybrid_rerank` | Dense + sparse (RRF) | Yes | No |",
            "| `hybrid_rerank_prefilter` | Dense + sparse (RRF) | Yes | Yes |",
            "",
            "## Notes",
            "",
            "- **avg_latency_ms** — mean end-to-end latency per question (LLM + retrieval).",
            "- **cache_hit_rate** — measured on a second pass with semantic cache enabled.",
            "- RAGAS scores use Groq (LLM) and Ollama (embeddings) from your `.env`.",
            "",
        ]
    )
    return "\n".join(lines)
