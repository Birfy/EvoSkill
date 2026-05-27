"""CLI entry point for SkillLearnBench-style evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.evaluator import SkillLearnBenchEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run three-layer skill evaluation")
    parser.add_argument("--benchmark", required=True, help="Benchmark directory with tasks.json")
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="Condition as name:trajectory_jsonl:skills_dir. May be repeated.",
    )
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Anthropic model for alignment scoring")
    parser.add_argument("--output", required=True, help="Output summary JSON path")
    args = parser.parse_args()

    evaluator = SkillLearnBenchEvaluator(args.benchmark, model=args.model)
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
