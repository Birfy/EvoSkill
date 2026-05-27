#!/usr/bin/env python3
"""Collect agent trajectories from a dataset and save to disk.

Run this ONCE to generate a trajectory file, then pass --trajectories_dir
to run_loop.py so the evolution loop never needs to re-run the agent.

Usage:
    python scripts/collect_trajectories.py \\
        --dataset .dataset/new_runs_base/solved_dataset.csv \\
        --output_dir .evoskill/trajectories \\
        --concurrency 4

The output is a JSONL file (.evoskill/trajectories/trajectories.jsonl).
Each line is a JSON-serialised StoredTrajectory.
"""

import asyncio
import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm.asyncio import tqdm_asyncio

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import Agent, set_sdk
from src.harness.sdk_config import get_sdk
from src.agent_profiles import base_agent_options, make_base_agent_options
from src.agent_profiles.base_agent.base_agent import make_base_agent_options_from_task
from src.agent_profiles.skill_generator import get_project_root
from src.schemas import AgentResponse
from src.schemas.trajectory import StoredTrajectory


class CollectSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,
        title="Collect agent trajectories",
    )

    dataset: str = Field(
        description="Path to dataset CSV with question/ground_truth/category columns",
    )
    output_dir: str = Field(
        default=".evoskill/trajectories",
        description="Directory to write trajectories.jsonl",
    )
    concurrency: int = Field(
        default=4,
        description="Max concurrent agent runs",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model override for base agent (e.g. 'opus', 'sonnet', 'haiku')",
    )
    sdk: Literal["claude", "opencode", "codex", "goose", "openhands"] = Field(
        default="claude",
        description="Agent SDK to use",
    )
    question_column: str = Field(
        default="question",
        description="CSV column name for questions",
    )
    ground_truth_column: str = Field(
        default="ground_truth",
        description="CSV column name for expected answers",
    )
    category_column: str = Field(
        default="category",
        description="CSV column name for categories (added as 'default' if missing)",
    )
    uid_column: Optional[str] = Field(
        default=None,
        description="Optional stable row id column used for resume bookkeeping",
    )
    project_root: Optional[str] = Field(
        default=None,
        description="Optional project root for harness workspace resolution",
    )
    data_dir: list[str] = Field(
        default_factory=list,
        description="Optional data directory mounted/readable by harness. Repeat for multiple dirs.",
    )
    task_prompt_file: Optional[str] = Field(
        default=None,
        description="Optional system prompt file to use instead of the default base prompt",
    )
    resume: bool = Field(
        default=True,
        description="Resume from existing trajectories.jsonl in output_dir",
    )


def load_dataset(settings: CollectSettings) -> list[tuple[str, str, str, str]]:
    """Load CSV and return list of (uid, question, ground_truth, category)."""
    data = pd.read_csv(settings.dataset)

    renames: dict[str, str] = {}
    if settings.question_column != "question":
        renames[settings.question_column] = "question"
    if settings.ground_truth_column != "ground_truth":
        renames[settings.ground_truth_column] = "ground_truth"
    if renames:
        data.rename(columns=renames, inplace=True)

    if settings.category_column in data.columns and settings.category_column != "category":
        data.rename(columns={settings.category_column: "category"}, inplace=True)
    elif "category" not in data.columns:
        data["category"] = "default"

    uid_col = settings.uid_column
    if uid_col and uid_col in data.columns:
        data["_trajectory_uid"] = data[uid_col].astype(str)
    elif "uid" in data.columns:
        data["_trajectory_uid"] = data["uid"].astype(str)
    else:
        data["_trajectory_uid"] = [str(i) for i in range(len(data))]

    data = data.dropna(subset=["question", "ground_truth"])
    return [
        (
            str(row["_trajectory_uid"]),
            str(row["question"]),
            str(row["ground_truth"]),
            str(row["category"]),
        )
        for _, row in data.iterrows()
    ]


def load_done_questions(path: Path) -> set[str]:
    """Return questions already present in a StoredTrajectory JSONL file."""
    if not path.exists():
        return set()
    done: set[str] = set()
    for stored in StoredTrajectory.load_jsonl(path):
        done.add(stored.question)
    return done


async def collect(settings: CollectSettings) -> list[StoredTrajectory]:
    """Run the base agent on every sample and return StoredTrajectory list."""
    set_sdk(settings.sdk)

    all_data = load_dataset(settings)
    output_path = Path(settings.output_dir) / "trajectories.jsonl"
    done_questions = load_done_questions(output_path) if settings.resume else set()
    pending = [row for row in all_data if row[1] not in done_questions]
    print(f"Loaded {len(all_data)} samples from {settings.dataset}")
    print(f"Existing={len(done_questions)} pending={len(pending)} output={output_path}")

    if settings.task_prompt_file:
        task_prompt = Path(settings.task_prompt_file).read_text().strip()
        base_options = make_base_agent_options_from_task(
            task_prompt,
            model=settings.model,
            data_dirs=settings.data_dir or None,
            project_root=settings.project_root,
        )
    else:
        base_options = (
            make_base_agent_options(
                model=settings.model,
                data_dirs=settings.data_dir or None,
                project_root=settings.project_root,
            )
            if settings.model or settings.data_dir or settings.project_root
            else base_agent_options
        )
    agent: Agent[AgentResponse] = Agent(base_options, AgentResponse)

    semaphore = asyncio.Semaphore(settings.concurrency)
    lock = asyncio.Lock()
    total_cost = 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def run_one(uid: str, question: str, ground_truth: str, category: str) -> StoredTrajectory:
        nonlocal total_cost
        async with semaphore:
            try:
                trace = await agent.run(question)
            except Exception as e:
                print(f"\n[WARN] Agent failed on '{question[:60]}...': {e}")
                # Return a failed trajectory so the record is preserved
                from src.harness.agent import AgentTrace
                trace = AgentTrace(
                    duration_ms=0,
                    total_cost_usd=0.0,
                    num_turns=0,
                    usage={},
                    result="",
                    is_error=True,
                    parse_error=str(e),
                    messages=[],
                )

            total_cost += trace.total_cost_usd
            agent_answer = (
                trace.output.final_answer
                if trace.output and trace.output.final_answer
                else "[PARSE FAILED]"
            )
            return StoredTrajectory.from_trace(
                trace=trace,
                question=question,
                ground_truth=ground_truth,
                category=category,
                agent_answer=agent_answer,
                task_id=uid,
                domain=category,
            )

    async def run_and_save(uid: str, question: str, ground_truth: str, category: str) -> StoredTrajectory:
        trajectory = await run_one(uid, question, ground_truth, category)
        async with lock:
            with output_path.open("a") as f:
                f.write(trajectory.model_dump_json() + "\n")
            print(f"[saved] {uid} cost=${trajectory.trace_total_cost_usd:.4f}")
        return trajectory

    tasks = [run_and_save(uid, q, gt, cat) for uid, q, gt, cat in pending]
    trajectories: list[StoredTrajectory] = await tqdm_asyncio.gather(
        *tasks, desc="Collecting trajectories"
    ) if tasks else []

    print(f"\nTotal inference cost: ${total_cost:.4f}")
    return trajectories


async def main(settings: CollectSettings) -> None:
    await collect(settings)

    output_dir = Path(settings.output_dir)
    output_path = output_dir / "trajectories.jsonl"
    trajectories = StoredTrajectory.load_jsonl(output_path)

    # Print summary
    n_ok = sum(1 for t in trajectories if not t.trace_is_error and t.agent_answer != "[PARSE FAILED]")
    categories = sorted({t.category for t in trajectories})
    print(f"Saved {len(trajectories)} trajectories → {output_path}")
    print(f"  Parsed OK: {n_ok}/{len(trajectories)}")
    print(f"  Categories ({len(categories)}): {', '.join(categories)}")


if __name__ == "__main__":
    settings = CollectSettings()
    asyncio.run(main(settings))
