"""Run the bundled task set and print a score the README can quote."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from briefwright.config import Settings
from briefwright.eval.metrics import score_run
from briefwright.service import run_research


def load_tasks() -> list[dict]:
    path = Path(__file__).with_name("tasks.json")
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval(settings: Settings | None = None) -> dict:
    """Score every bundled task. Memory is isolated so earlier briefs cannot leak in."""

    base = settings or Settings()
    with TemporaryDirectory() as directory:
        isolated = base.model_copy(
            update={"memory_path": str(Path(directory) / "memory.json"), "auto_approve": True}
        )
        rows = []
        for task in load_tasks():
            result = run_research(task["query"], settings=isolated, auto_approve=True)
            rows.append(score_run(task, result))
    macro = round(sum(row["composite"] for row in rows) / len(rows), 4) if rows else 0.0
    return {
        "backend": isolated.llm_backend,
        "model": "mock-research-v1" if isolated.llm_backend == "mock" else isolated.openai_model,
        "n_tasks": len(rows),
        "macro_score": macro,
        "tasks": rows,
    }


def main() -> int:
    print(json.dumps(run_eval(), indent=2))
    return 0
