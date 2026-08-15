"""Async evaluation orchestrator across retrieval configs."""

from __future__ import annotations

import asyncio
from pathlib import Path

from hybrid_rag.config.settings import Settings, get_settings
from scripts.evaluation.dataset import EvalQuestion, load_questions
from scripts.evaluation.metrics import compute_ragas_scores
from scripts.evaluation.pipeline import EVAL_MODES, EvaluationPipeline
from scripts.evaluation.report import ConfigResult, render_markdown
from hybrid_rag.utils.logging import logger

DEFAULT_CONFIGS: tuple[str, ...] = EVAL_MODES


async def _run_questions(
    pipeline: EvaluationPipeline,
    questions: list[EvalQuestion],
    mode: str,
    concurrency: int,
) -> list:
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(question: EvalQuestion):
        async with semaphore:
            return await pipeline.ainvoke(
                question_id=question.id,
                question=question.question,
                ground_truth=question.ground_truth,
                mode=mode,
            )

    return await asyncio.gather(*[_one(q) for q in questions])


async def _evaluate_config(
    config_name: str,
    questions: list[EvalQuestion],
    settings: Settings,
    concurrency: int,
    measure_cache: bool,
) -> ConfigResult:
    logger.info("Evaluating config=%s (%s questions)", config_name, len(questions))

    cold_pipeline = EvaluationPipeline(settings, use_cache=False)
    cold_results = await _run_questions(cold_pipeline, questions, config_name, concurrency)
    avg_latency = sum(r.latency_ms for r in cold_results) / max(len(cold_results), 1)

    ragas = compute_ragas_scores(cold_results, settings)

    cache_hit_rate = 0.0
    if measure_cache and settings.cache_enabled:
        logger.info("Cache benchmark (2nd pass) for config=%s", config_name)
        warm_pipeline = EvaluationPipeline(settings, use_cache=True)
        await _run_questions(warm_pipeline, questions, config_name, concurrency)
        warm_results = await _run_questions(warm_pipeline, questions, config_name, concurrency)
        hits = sum(1 for r in warm_results if r.cache_hit)
        cache_hit_rate = hits / max(len(warm_results), 1)

    return ConfigResult(
        name=config_name,
        ragas=ragas,
        avg_latency_ms=avg_latency,
        cache_hit_rate=cache_hit_rate,
        question_count=len(questions),
    )


async def run_evaluation(
    output_path: Path | None = None,
    questions_path: Path | None = None,
    configs: tuple[str, ...] | None = None,
    concurrency: int | None = None,
    measure_cache: bool = True,
    settings: Settings | None = None,
) -> str:
    cfg = settings or get_settings()
    questions_file = questions_path or (cfg.data_dir / "eval_questions.json")
    target = output_path or Path("evaluation_results.md")
    selected_configs = configs or DEFAULT_CONFIGS
    parallel = concurrency or cfg.eval_concurrency

    for name in selected_configs:
        if name not in EVAL_MODES:
            raise ValueError(f"Unknown config '{name}'. Valid: {EVAL_MODES}")

    if not cfg.groq_api_key:
        raise ValueError("GROQ_API_KEY is required for evaluation.")

    questions = load_questions(questions_file, cfg)
    logger.info(
        "Starting evaluation | questions=%s configs=%s concurrency=%s",
        len(questions),
        selected_configs,
        parallel,
    )

    config_results: list[ConfigResult] = []
    for config_name in selected_configs:
        result = await _evaluate_config(
            config_name=config_name,
            questions=questions,
            settings=cfg,
            concurrency=parallel,
            measure_cache=measure_cache,
        )
        config_results.append(result)
        logger.info(
            "Finished %s | precision=%.3f latency=%.0fms cache=%.1f%%",
            config_name,
            result.ragas.context_precision,
            result.avg_latency_ms,
            result.cache_hit_rate * 100,
        )

    markdown = render_markdown(config_results, str(questions_file))
    target.write_text(markdown, encoding="utf-8")
    logger.info("Wrote evaluation report to %s", target.resolve())
    return str(target.resolve())
