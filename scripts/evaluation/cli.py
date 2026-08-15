"""CLI entrypoint for Hybrid RAG evaluation."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from hybrid_rag.config.settings import get_settings
from scripts.evaluation.pipeline import EVAL_MODES
from scripts.evaluation.runner import run_evaluation
from hybrid_rag.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="hybrid-rag-eval",
        description="Run RAGAS evaluation across naive / hybrid / hybrid+prefilter configs.",
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=settings.data_dir / "eval_questions.json",
        help="Path to eval questions JSON (default: data/eval_questions.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("evaluation_results.md"),
        help="Output markdown report path (default: evaluation_results.md)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(EVAL_MODES),
        help=f"Comma-separated configs (default: all). Choices: {', '.join(EVAL_MODES)}",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=settings.eval_concurrency,
        help=f"Max concurrent questions (default: {settings.eval_concurrency})",
    )
    parser.add_argument(
        "--skip-cache-benchmark",
        action="store_true",
        help="Skip second-pass cache hit rate measurement",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        help="Log level for evaluation run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.log_level = str(args.log_level).upper()
    setup_logging(settings)

    configs = tuple(c.strip() for c in args.configs.split(",") if c.strip())
    invalid = [c for c in configs if c not in EVAL_MODES]
    if invalid:
        parser.error(f"Invalid config(s): {invalid}. Choose from {EVAL_MODES}")

    try:
        output = asyncio.run(
            run_evaluation(
                output_path=args.output,
                questions_path=args.questions_path,
                configs=configs,
                concurrency=args.concurrency,
                measure_cache=not args.skip_cache_benchmark,
                settings=settings,
            )
        )
        print(f"Evaluation complete → {output}")
        return 0
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
