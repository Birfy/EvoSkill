"""CLI entry point for generic offline skill evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.evaluator import OfflineSkillEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run three-layer skill evaluation")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Task JSON/JSONL file, or a directory containing tasks.json. Optional if trajectories contain task ids.",
    )
    parser.add_argument(
        "--keypoints-dir",
        default=None,
        help="Optional directory of <task_id>.json files with required_actions for static outcome checks.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Deprecated compatibility shortcut for a directory with tasks.json and eval_keypoints/.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="Condition as name:trajectory_jsonl:skills_dir. May be repeated.",
    )
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Anthropic model for alignment scoring")
    parser.add_argument("--output", required=True, help="Output summary JSON path")
    args = parser.parse_args()

    tasks = args.tasks
    keypoints_dir = args.keypoints_dir
    if args.benchmark:
        benchmark = Path(args.benchmark)
        tasks = tasks or str(benchmark / "tasks.json")
        keypoints_dir = keypoints_dir or str(benchmark / "eval_keypoints")

    evaluator = OfflineSkillEvaluator(
        tasks_path=tasks,
        model=args.model,
        keypoints_dir=keypoints_dir,
    )
    summaries = []
    for raw in args.condition:
        name, trajectory_path, skills_dir = raw.split(":", 2)
        summaries.append(
            evaluator.run_full_evaluation(
                skill_library_path=skills_dir,
                trajectory_path=trajectory_path,
                condition_name=name,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"conditions": summaries}, indent=2))


if __name__ == "__main__":
    main()
