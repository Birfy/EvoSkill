#!/usr/bin/env python3
"""Collect baseline trajectories for SciSkillBench tasks.

Usage:
    set -a; source examples/skillsbench/.evoskill/.env.deepseek.local; set +a
    export EVOSKILL_NO_GIT=1
    export PATH="$(pwd)/.venv/bin:$PATH"

    # Level 0 (procedural queries) baseline:
    .venv/bin/python3 scripts/collect_sciskillbench_baseline.py \\
        --output_dir examples/sciskillbench/.evoskill/trajectories/baseline-level0 \\
        --level 0

    # Level 1 (abstract queries):
    .venv/bin/python3 scripts/collect_sciskillbench_baseline.py \\
        --output_dir examples/sciskillbench/.evoskill/trajectories/baseline-level1 \\
        --level 1

    # All levels combined:
    .venv/bin/python3 scripts/collect_sciskillbench_baseline.py \\
        --output_dir examples/sciskillbench/.evoskill/trajectories/baseline-all
"""

import asyncio
import csv
import json
import os
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


TASK_PROMPT_FILE = Path("examples/sciskillbench/.evoskill/task.md")
DATASET_CSV = Path("examples/sciskillbench/.evoskill/tasks_no_orca.csv")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,
    )

    output_dir: str = Field(
        description="Directory to write trajectories.jsonl",
    )
    level: Optional[str] = Field(
        default=None,
        description="Query level to run: '0' (procedural), '1' (abstract), or None for all",
    )
    concurrency: int = Field(
        default=2,
        description="Max concurrent agent runs",
    )
    resume: bool = Field(
        default=True,
        description="Skip already-completed task IDs",
    )
    limit: int = Field(
        default=0,
        description="Max tasks to run (0 = all)",
    )
    model: str = Field(
        default="deepseek-v4-flash",
        description="Model to use for agent",
    )
    timeout: int = Field(
        default=300,
        description="Timeout per task in seconds",
    )


def _score_sciskillbench(agent_answer: str, ground_truth_json: str, tolerance_json: str) -> float:
    """Score a SciSkillBench answer against expected value with tolerance."""
    if not agent_answer or not agent_answer.strip():
        return 0.0

    try:
        expected = json.loads(ground_truth_json)
        tolerance = json.loads(tolerance_json)
    except Exception:
        return 0.0

    return _compare_values(agent_answer, expected, tolerance)


def _compare_values(predicted_text: str, expected, tolerance) -> float:
    """Compare predicted text to expected value with given tolerance."""
    import re

    # Try to extract the answer from common patterns
    # Look for "ANSWER: <value>" or "Final answer: <value>" etc.
    patterns = [
        r"ANSWER:\s*(.+?)(?:\n|$)",
        r"final answer[:\s]+(.+?)(?:\n|$)",
        r"result[:\s]+(.+?)(?:\n|$)",
        r"output[:\s]+(.+?)(?:\n|$)",
    ]

    extracted = None
    for pat in patterns:
        m = re.search(pat, predicted_text, re.IGNORECASE)
        if m:
            extracted = m.group(1).strip()
            break

    if extracted is None:
        # Fall back to last non-empty line
        lines = [l.strip() for l in predicted_text.strip().splitlines() if l.strip()]
        extracted = lines[-1] if lines else ""

    if not extracted:
        return 0.0

    # Resolve tolerance: treat non-numeric strings as exact/semantic match
    def _numeric_tol(tol) -> float | None:
        if isinstance(tol, (int, float)):
            return float(tol)
        if isinstance(tol, str):
            try:
                return float(tol)
            except (ValueError, TypeError):
                return None  # exact/semantic match
        return None

    if isinstance(expected, (int, float)):
        tol_val = _numeric_tol(tolerance)
        if tol_val is None:
            # exact match for numeric
            return 1.0 if abs(float(extracted.replace(",", "").split()[-1] if extracted else "nan") - float(expected)) < 1e-9 else 0.0
        return _score_numeric(extracted, float(expected), tol_val)
    elif isinstance(expected, list):
        return _score_list(extracted, expected, tolerance)
    elif isinstance(expected, str):
        # exact_match or semantic_match: check substring/containment
        tol_str = str(tolerance).lower() if tolerance else ""
        if "semantic" in tol_str:
            # partial match ok
            return 1.0 if expected.strip().lower() in extracted.lower() or extracted.lower() in expected.strip().lower() else 0.0
        return 1.0 if extracted.strip().lower() == expected.strip().lower() else 0.0
    return 0.0


def _score_numeric(text: str, expected: float, tolerance: float) -> float:
    """Extract and compare a single number."""
    import re
    nums = re.findall(r"-?\d+\.?\d*(?:e[+-]?\d+)?", text.replace(",", ""))
    if not nums:
        return 0.0
    # Try the last number (often the final result)
    for num_str in reversed(nums):
        try:
            val = float(num_str)
            if abs(expected) > 1e-10:
                if abs(val - expected) <= tolerance:
                    return 1.0
            else:
                if abs(val - expected) <= tolerance:
                    return 1.0
        except ValueError:
            continue
    return 0.0


def _score_list(text: str, expected: list, tolerance) -> float:
    """Score a list answer by checking if expected items appear in the text."""
    import re
    scores = []
    for i, item in enumerate(expected):
        tol = tolerance[i] if isinstance(tolerance, list) and i < len(tolerance) else tolerance
        tol_f = float(tol) if isinstance(tol, (int, float)) or (isinstance(tol, str) and tol.replace('.','').replace('-','').replace('e','').replace('+','').isdigit()) else 0.01
        if isinstance(item, (int, float)):
            s = _score_numeric(text, float(item), tol_f)
            scores.append(s)
        elif isinstance(item, str):
            s = 1.0 if item.lower() in text.lower() else 0.0
            scores.append(s)
        elif isinstance(item, list):
            s = _score_list(text, item, tol)
            scores.append(s)
    return sum(scores) / len(scores) if scores else 0.0


async def run_one(
    task_id: str,
    question: str,
    ground_truth: str,
    category: str,
    tolerance: str,
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

        score = _score_sciskillbench(agent_answer, ground_truth, tolerance)
        passed = score >= 0.5

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

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {task_id}  score={score:.3f}  ({elapsed:.0f}s)")


async def main() -> None:
    cfg = Settings()

    # Load tasks
    tasks = []
    with open(DATASET_CSV) as f:
        for row in csv.DictReader(f):
            if cfg.level and row["level"] != cfg.level:
                continue
            tasks.append(row)

    if cfg.limit > 0:
        tasks = tasks[: cfg.limit]

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "trajectories.jsonl"

    # Resume: skip done task IDs
    done_ids: set[str] = set()
    if cfg.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    done_ids.add(data.get("task_id", ""))
                except json.JSONDecodeError:
                    pass
        if done_ids:
            print(f"Resuming — {len(done_ids)} tasks already done.")

    pending = [t for t in tasks if t["task_id"] not in done_ids]
    print(f"Tasks total: {len(tasks)} | Pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    # Load task prompt
    task_prompt = TASK_PROMPT_FILE.read_text().strip() if TASK_PROMPT_FILE.exists() else ""

    # Build agent options
    set_sdk("claude")
    from src.agent_profiles.base_agent.base_agent import make_base_agent_options_from_task
    if task_prompt:
        options = make_base_agent_options_from_task(
            task_prompt,
            model=cfg.model,
            project_root=Path("examples/sciskillbench"),
            data_dirs=[],
        )
    else:
        from src.agent_profiles import make_base_agent_options
        options = make_base_agent_options(
            model=cfg.model,
            project_root=Path("examples/sciskillbench"),
        )

    sem = asyncio.Semaphore(cfg.concurrency)

    await tqdm_asyncio.gather(
        *[
            run_one(
                task_id=t["task_id"],
                question=t["question"],
                ground_truth=t["ground_truth"],
                category=t["category"],
                tolerance=t["tolerance"],
                task_prompt=task_prompt,
                options=options,
                sem=sem,
                out_path=out_path,
            )
            for t in pending
        ],
        desc="SciSkillBench tasks",
    )

    # Final summary
    results = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ans = data.get("agent_answer", "")
                gt = data.get("ground_truth", "null")
                tid = data.get("task_id", "")
                # Get tolerance from CSV
                row = next((t for t in tasks if t["task_id"] == tid), None)
                tol = row["tolerance"] if row else "0.01"
                score = _score_sciskillbench(ans, gt, tol)
                results.append(score >= 0.5)
            except Exception:
                pass

    total = len(results)
    passed = sum(results)
    print(f"\nDone. Pass rate: {passed}/{total} = {passed/total:.1%}" if total else "\nNo results.")


if __name__ == "__main__":
    asyncio.run(main())
