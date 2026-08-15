"""Load and validate evaluation question datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hybrid_rag.config.settings import Settings, get_settings


@dataclass(frozen=True)
class EvalQuestion:
    id: int
    type: str
    question: str
    ground_truth: str


def default_questions_path(settings: Settings | None = None) -> Path:
    cfg = settings or get_settings()
    return cfg.data_dir / "eval_questions.json"


def load_questions(path: Path | None = None, settings: Settings | None = None) -> list[EvalQuestion]:
    file_path = path or default_questions_path(settings)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Evaluation questions not found: {file_path.resolve()}\n"
            "Add data/eval_questions.json or pass --questions-path."
        )

    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty JSON array in {file_path}")

    questions: list[EvalQuestion] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid question entry at index {index - 1}: expected object")
        questions.append(
            EvalQuestion(
                id=int(item.get("id", index)),
                type=str(item.get("type", "unknown")),
                question=str(item["question"]).strip(),
                ground_truth=str(item.get("ground_truth", "")).strip(),
            )
        )
    return questions
