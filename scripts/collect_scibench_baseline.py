#!/usr/bin/env python3
"""Collect baseline trajectories for SciBench chemistry tasks."""

import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import Agent, set_sdk
from src.schemas import AgentResponse
from src.schemas.trajectory import StoredTrajectory


TASK_PROMPT_FILE = Path("examples/scibench/.evoskill/task.md")
DATASET_CSV = Path("examples/scibench/.evoskill/tasks.csv")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,
    )
    output_dir: str = Field(description="Directory to write trajectories.jsonl")
    concurrency: int = Field(default=4, description="Max concurrent agent runs")
    resume: bool = Field(default=True, description="Skip already-completed task IDs")
    limit: int = Field(default=0, description="Max tasks to run (0 = all)")
    model: str = Field(default="deepseek-v4-flash", description="Model to use")
    timeout: int = Field(default=300, description="Timeout per task in seconds")
    category: Optional[str] = Field(default=None, description="Filter by category")
    project_root: str = Field(default="examples/scibench", description="Project root whose .claude/skills the agent loads")


def _extract_answer(text: str) -> str:
    """Extract numeric answer from agent output."""
    # Prefer explicit ANSWER: pattern
    m = re.search(r"ANSWER:\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fall back to last number on last line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        nums = re.findall(r"[+-]?\d+\.?\d*(?:e[+-]?\d+)?", lines[-1].replace(",", ""))
        if nums:
            return nums[-1]
    return ""


def _score(agent_answer: str, ground_truth: str) -> float:
    """Score using 1% relative tolerance (SciBench standard)."""
    extracted = _extract_answer(agent_answer)
    if not extracted:
        return 0.0
    try:
        pred = float(extracted)
        gt = float(str(ground_truth).strip().lstrip("+"))
    except ValueError:
        return 0.0
    if abs(gt) < 1e-10:
        return 1.0 if abs(pred) < 1e-10 else 0.0
    return 1.0 if abs(pred - gt) / abs(gt) <= 0.01 else 0.0


async def run_one(
    task_id: str,
    question: str,
    ground_truth: str,
    category: str,
    unit: str,
    task_prompt: str,
    options,
    sem: asyncio.Semaphore,
    out_path: Path,
) -> None:
    async with sem:
        full_question = f"{task_prompt}\n\n---\n\n{question}"
        agent = Agent(options, AgentResponse, timeout_seconds=3600)
        t0 = time.time()
        try:
            trace = await agent.run(full_question)
        except Exception as e:
            print(f"  [ERROR] {task_id}: {e}")
            trace = None
        elapsed = time.time() - t0

        agent_answer = ""
        if trace and trace.output:
            agent_answer = trace.output.final_answer or ""

        score = _score(agent_answer, ground_truth)
        status = "PASS" if score >= 0.5 else "FAIL"

        stored = StoredTrajectory.from_trace(
            trace=trace,
            question=question,
            ground_truth=ground_truth,
            category=category,
            agent_answer=agent_answer,
            task_id=task_id,
        )
        with open(out_path, "a") as f:
            f.write(stored.model_dump_json() + "\n")

        print(f"  [{status}] {task_id}  score={score:.3f}  gt={ground_truth}  ans={_extract_answer(agent_answer)}  ({elapsed:.0f}s)")


async def main() -> None:
    cfg = Settings()
    set_sdk("claude")

    tasks = []
    with open(DATASET_CSV) as f:
        for row in csv.DictReader(f):
            if cfg.category and row["category"] != cfg.category:
                continue
            tasks.append(row)

    if cfg.limit > 0:
        tasks = tasks[: cfg.limit]

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "trajectories.jsonl"

    done_ids: set[str] = set()
    if cfg.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line).get("task_id", ""))
                except json.JSONDecodeError:
                    pass
        if done_ids:
            print(f"Resuming — {len(done_ids)} tasks already done.")

    pending = [t for t in tasks if t["task_id"] not in done_ids]
    print(f"Tasks total: {len(tasks)} | Pending: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    task_prompt = TASK_PROMPT_FILE.read_text().strip() if TASK_PROMPT_FILE.exists() else ""

    from src.agent_profiles.base_agent.base_agent import make_base_agent_options_from_task
    options = make_base_agent_options_from_task(
        task_prompt,
        model=cfg.model,
        project_root=Path(cfg.project_root),
        data_dirs=[],
    )

    sem = asyncio.Semaphore(cfg.concurrency)
    await tqdm_asyncio.gather(
        *[
            run_one(
                task_id=t["task_id"],
                question=t["question"],
                ground_truth=t["ground_truth"],
                category=t["category"],
                unit=t.get("unit", ""),
                task_prompt=task_prompt,
                options=options,
                sem=sem,
                out_path=out_path,
            )
            for t in pending
        ],
        desc="SciBench Chemistry",
    )

    # Final summary
    results = []
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            tid = d.get("task_id", "")
            row = next((t for t in tasks if t["task_id"] == tid), None)
            if row:
                s = _score(d.get("agent_answer", ""), row["ground_truth"])
                results.append(s >= 0.5)
        except Exception:
            pass

    total = len(results)
    passed = sum(results)
    print(f"\nDone. Pass rate: {passed}/{total} = {passed/total:.1%}" if total else "\nNo results.")


if __name__ == "__main__":
    asyncio.run(main())
