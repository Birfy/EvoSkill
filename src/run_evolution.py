"""CLI entry point for offline skill evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.skill_evolver.pipeline import SkillEvolutionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline skill evolution")
    parser.add_argument("--trajectories", required=True, help="Trajectory JSONL path or directory")
    parser.add_argument("--skills", required=True, help="Skill library directory")
    parser.add_argument("--rounds", type=int, default=5, help="Number of evolution rounds")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Anthropic model for evolution/scoring")
    parser.add_argument("--output", default=None, help="Optional JSON summary path")
    args = parser.parse_args()

    pipeline = SkillEvolutionPipeline(
        skill_library_path=args.skills,
        trajectory_path=args.trajectories,
        n_rounds=args.rounds,
        model=args.model,
    )
    result = pipeline.run()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
