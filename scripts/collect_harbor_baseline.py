#!/usr/bin/env python3
"""Collect baseline trajectories from SkillsBench (Harbor) tasks.

Usage:
    set -a; source examples/skillsbench/.evoskill/.env.deepseek.local; set +a
    export EVOSKILL_NO_GIT=1
    export DOCKER_HOST=unix:///run/user/1000/podman/podman.sock  # Podman on Alibaba Cloud
    export PATH="$(pwd)/.venv/bin:$PATH"  # harbor CLI lives in venv

    # With per-task curated skills (default — each task gets only its own skills):
    .venv/bin/python3 scripts/collect_harbor_baseline.py \\
        --output_dir examples/skillsbench/.evoskill/trajectories/curated-skills-baseline

    # With a shared global skills dir (all skills for every task):
    .venv/bin/python3 scripts/collect_harbor_baseline.py \\
        --output_dir examples/skillsbench/.evoskill/trajectories/curated-skills-baseline \\
        --skills_dir examples/skillsbench/.claude/skills

    # Without skills:
    .venv/bin/python3 scripts/collect_harbor_baseline.py \\
        --output_dir examples/skillsbench/.evoskill/trajectories/no-skills-baseline \\
        --skills_dir ""
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.harbor_loader import load_harbor_tasks, TASK_PATH_INDEX
from src.cli.config import load_config
from src.evaluation.harbor_scorer import harbor_reward_scorer
from src.harness.harbor import HarborAgent
from src.schemas.trajectory import StoredTrajectory


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,
    )

    config: str = Field(
        default="examples/skillsbench/.evoskill/config.toml",
        description="Path to EvoSkill config.toml (resolved relative to repo root)",
    )
    output_dir: str = Field(
        description="Directory to write trajectories.jsonl",
    )
    skills_dir: str = Field(
        default="",
        description=(
            "Global skills dir to mount for ALL tasks (empty = use per-task skills "
            "from task_dir/environment/skills/, or no skills if that dir is absent)"
        ),
    )
    concurrency: int = Field(
        default=1,
        description="Max concurrent harbor jobs",
    )
    resume: bool = Field(
        default=True,
        description="Skip already-completed task IDs found in output_dir",
    )
    limit: int = Field(
        default=0,
        description="Max tasks to run (0 = all)",
    )


def _build_agent_env_args() -> list[str]:
    """Return --ae KEY=VALUE pairs for ANTHROPIC_* and CLAUDE_CODE_* env vars."""
    passthrough = [
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "CLAUDE_CODE_EFFORT_LEVEL",
    ]
    args = []
    for key in passthrough:
        val = os.environ.get(key)
        if val:
            args.extend(["--ae", f"{key}={val}"])
    return args


async def main() -> None:
    cfg_settings = Settings()
    cfg = load_config(config_path=Path(cfg_settings.config))

    # Load task list — populates TASK_PATH_INDEX
    df = load_harbor_tasks(cfg)
    tasks = list(zip(df["question"], df["ground_truth"], df["category"]))

    if cfg_settings.limit > 0:
        tasks = tasks[: cfg_settings.limit]

    output_dir = Path(cfg_settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "trajectories.jsonl"

    # Load already-completed task IDs for resume
    done_ids: set[str] = set()
    if cfg_settings.resume and out_path.exists():
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
        print(f"Resuming — {len(done_ids)} tasks already done.")

    pending = [(q, gt, cat) for q, gt, cat in tasks if q not in done_ids]
    print(f"Tasks total: {len(tasks)} | Pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    global_skills_dir = Path(cfg_settings.skills_dir) if cfg_settings.skills_dir else None
    extra_args = _build_agent_env_args()

    def _make_agent(skills_source: Path) -> HarborAgent:
        return HarborAgent(
            project_root=Path("."),
            skills_source_dir=skills_source,
            inner_agent=cfg.harbor.inner_agent,
            inner_model=cfg.harbor.inner_model,
            env=cfg.harbor.env,
            n_concurrent=cfg.harbor.n_concurrent,
            timeout_seconds=cfg.harness.timeout_seconds * cfg.harbor.timeout_multiplier,
            max_retries=cfg.harness.max_retries,
            container_skills_path=cfg.harbor.container_skills_path,
            timeout_multiplier=cfg.harbor.timeout_multiplier,
            extra_args=extra_args,
        )

    # Pre-build a shared agent when using a global skills dir; otherwise agents
    # are created per-task to pick up the task-specific environment/skills/ dir.
    shared_agent = _make_agent(global_skills_dir) if global_skills_dir else None

    # Bugswarm tasks use 8GB+ images — run them one at a time to avoid disk exhaustion
    BUGSWARM_TASKS = {"tasks/fix-build-agentops", "tasks/fix-build-google-auto"}
    sem = asyncio.Semaphore(cfg_settings.concurrency)
    sem_bugswarm = asyncio.Semaphore(1)

    async def run_one(task_id: str, ground_truth: str, category: str) -> None:
        limiter = sem_bugswarm if task_id in BUGSWARM_TASKS else sem
        async with limiter:
            if shared_agent is not None:
                agent = shared_agent
            else:
                # Per-task skills: use task_dir/environment/skills/ if present.
                task_dir = TASK_PATH_INDEX.get(task_id)
                per_task_skills = (
                    (task_dir / "environment" / "skills")
                    if task_dir is not None
                    else None
                )
                if per_task_skills and per_task_skills.is_dir():
                    agent = _make_agent(per_task_skills)
                else:
                    agent = _make_agent(Path("/nonexistent"))
            trace = await agent.run(task_id)
            score = harbor_reward_scorer("", trace.output.final_answer if trace.output else "", "")
            passed = score >= 0.5

            agent_answer = trace.output.final_answer if trace.output else ""
            stored = StoredTrajectory.from_trace(
                trace=trace,
                question=task_id,
                ground_truth=ground_truth,
                category=category,
                agent_answer=agent_answer or "",
                task_id=task_id,
            )

            # Populate steps from harbor's trajectory.json (written by populate_context_post_run)
            try:
                envelope = json.loads(agent_answer) if agent_answer else {}
                job_dir = Path(envelope["job_dir"]) if envelope.get("job_dir") else None
                if job_dir and job_dir.is_dir():
                    traj_files = list(job_dir.rglob("agent/trajectory.json"))
                    if traj_files:
                        with open(traj_files[0]) as tf:
                            harbor_traj = json.load(tf)
                        stored.steps = harbor_traj.get("steps", [])
                        if harbor_traj.get("final_metrics", {}).get("total_cost_usd"):
                            stored.trace_total_cost_usd = harbor_traj["final_metrics"]["total_cost_usd"]
            except Exception:
                pass

            with open(out_path, "a") as f:
                f.write(stored.model_dump_json() + "\n")

            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {task_id}  reward={score:.3f}")
            # With concurrency=1, safe to remove all stopped containers and all images
            subprocess.run(["podman", "rm", "-a", "-f"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["podman", "rmi", "-a", "-f"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["podman", "volume", "prune", "-f"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    await tqdm_asyncio.gather(
        *[run_one(q, gt, cat) for q, gt, cat in pending],
        desc="Harbor tasks",
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
                score = harbor_reward_scorer("", ans, "")
                results.append(score >= 0.5)
            except Exception:
                pass

    total = len(results)
    passed = sum(results)
    print(f"\nDone. Pass rate: {passed}/{total} = {passed/total:.1%}" if total else "\nNo results.")


if __name__ == "__main__":
    asyncio.run(main())
