#!/usr/bin/env python3
"""
End-to-end test: OfficeQA trajectory collection + LLM judge evolution.

Run from examples/officeqa/:
    python ../../scripts/test_officeqa_llm_judge.py

Uses .evoskill/config.toml (harness + dataset config) but forces
use_llm_judge=True and skips the val-set re-evaluation step.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

# Ensure EvoSkill root is importable regardless of CWD
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.cli.config import load_config
from src.cli.shared import load_and_split, make_scorer
from src.harness import Agent, set_sdk
from src.agent_profiles.base_agent.base_agent import make_base_agent_options_from_task
from src.agent_profiles.skill_generator.skill_generator import make_skill_generator_options
from src.agent_profiles.skill_proposer.skill_proposer import make_skill_proposer_options
from src.agent_profiles.prompt_generator.prompt_generator import make_prompt_generator_options
from src.agent_profiles.prompt_proposer.prompt_proposer import make_prompt_proposer_options
from src.loop import SelfImprovingLoop, LoopAgents, LoopConfig
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
    StoredTrajectory,
)
from tqdm.asyncio import tqdm_asyncio


# ── helpers ──────────────────────────────────────────────────────────────────

def ensure_git_repo(project_root: Path) -> None:
    """Initialise a fresh git repo if needed (ProgramManager requires git)."""
    git_dir = project_root / ".git"
    if git_dir.exists():
        print("  [git] Using existing repo")
        return
    subprocess.run(["git", "init", "-q"], cwd=project_root, check=True)
    subprocess.run(["git", "add", "."], cwd=project_root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial commit (test)"],
        cwd=project_root,
        check=True,
    )
    print("  [git] Initialised fresh repo")


async def collect_trajectories(
    cfg,
    base_factory,
    all_data: list[tuple[str, str, str]],
    concurrency: int = 2,
) -> list[StoredTrajectory]:
    """Run base agent on every sample and return StoredTrajectory list."""
    agent: Agent[AgentResponse] = Agent(
        base_factory,
        AgentResponse,
        timeout_seconds=cfg.harness.timeout_seconds,
        max_retries=cfg.harness.max_retries,
    )

    sem = asyncio.Semaphore(concurrency)
    total_cost = 0.0

    async def run_one(question: str, ground_truth: str, category: str) -> StoredTrajectory:
        nonlocal total_cost
        async with sem:
            try:
                trace = await agent.run(question)
            except Exception as e:
                print(f"\n  [WARN] Agent failed: {e}")
                from src.harness.agent import AgentTrace
                trace = AgentTrace(
                    duration_ms=0, total_cost_usd=0.0, num_turns=0, usage={},
                    result="", is_error=True, parse_error=str(e), messages=[],
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
            )

    tasks = [run_one(q, gt, cat) for q, gt, cat in all_data]
    results: list[StoredTrajectory] = await tqdm_asyncio.gather(
        *tasks, desc="  Collecting trajectories"
    )
    print(f"  Trajectory cost: ${total_cost:.4f}")
    return results


# ── main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    project_root = Path.cwd().resolve()
    config_path = project_root / ".evoskill" / "config.toml"

    if not config_path.exists():
        print(f"ERROR: Not in an EvoSkill project directory (missing {config_path})")
        sys.exit(1)

    print(f"\n=== OfficeQA LLM Judge Test ===")
    print(f"  Project: {project_root}")

    cfg = load_config()

    # Override model to gpt-5.4-nano (matches existing officeqa configs)
    MODEL = "openai/gpt-5.4-nano"
    cfg.harness.model = MODEL
    # Force openhands SDK for OpenAI models
    cfg.harness.name = "openhands"

    set_sdk(cfg.harness.name)
    print(f"  SDK:     {cfg.harness.name}")
    print(f"  Model:   {cfg.harness.model}")

    # ── git setup ────────────────────────────────────────────────────────────
    ensure_git_repo(project_root)

    # ── dataset ──────────────────────────────────────────────────────────────
    train_pools, val_data = load_and_split(cfg)
    scorer = make_scorer(cfg)

    # Flatten all data (no train/val distinction for trajectory collection)
    seen: set = set()
    all_data: list[tuple[str, str, str]] = []
    for cat, pool in train_pools.items():
        for q, a in pool:
            key = (q, a, cat)
            if key not in seen:
                seen.add(key)
                all_data.append(key)
    for q, a, cat in val_data:
        key = (q, a, cat)
        if key not in seen:
            seen.add(key)
            all_data.append(key)

    print(f"\n[1/3] Dataset: {len(all_data)} samples total")
    cats = sorted({cat for _, _, cat in all_data})
    print(f"  Categories: {', '.join(cats)}")

    # ── base agent factory ───────────────────────────────────────────────────
    base_factory = make_base_agent_options_from_task(
        cfg.task_description,
        model=cfg.harness.model,
        data_dirs=cfg.harness.data_dirs,
        project_root=cfg.project_root,
    )

    # ── Step 1: Collect trajectories ─────────────────────────────────────────
    traj_dir = project_root / ".evoskill" / "trajectories"
    traj_file = traj_dir / "trajectories.jsonl"

    if traj_file.exists():
        print(f"\n[1/3] Trajectories already exist at {traj_file} — loading...")
        stored = StoredTrajectory.load_jsonl(traj_file)
        print(f"  Loaded {len(stored)} trajectories")
    else:
        print(f"\n[1/3] Collecting trajectories ({len(all_data)} samples, concurrency={cfg.evolution.concurrency})...")
        stored = await collect_trajectories(cfg, base_factory, all_data, cfg.evolution.concurrency)
        StoredTrajectory.save_jsonl(stored, traj_file)
        print(f"  Saved → {traj_file}")

    # Summary
    n_ok = sum(1 for t in stored if not t.trace_is_error and t.agent_answer != "[PARSE FAILED]")
    n_pass = sum(
        1 for t in stored
        if scorer(t.question, t.agent_answer.strip().lower(), t.ground_truth.strip().lower()) >= 0.8
    )
    print(f"  Parsed OK: {n_ok}/{len(stored)}, Passed: {n_pass}/{len(stored)}")

    # ── Step 2: Load preloaded trajectories ──────────────────────────────────
    print(f"\n[2/3] Converting to AgentTrace tuples...")
    preloaded = SelfImprovingLoop.load_trajectories_from_dir(traj_dir)
    print(f"  {len(preloaded)} trajectories ready for evolution loop")

    # ── Step 3: Run evolution with LLM judge ─────────────────────────────────
    print(f"\n[3/3] Running skill evolution with LLM judge...")
    print(f"  Judge: auto (will use {cfg.harness.name} provider, small model)")

    agents = LoopAgents(
        base=Agent(
            base_factory,
            AgentResponse,
            timeout_seconds=cfg.harness.timeout_seconds,
            max_retries=cfg.harness.max_retries,
        ),
        skill_proposer=Agent(
            make_skill_proposer_options(project_root=cfg.project_root, model=cfg.harness.model),
            SkillProposerResponse,
            timeout_seconds=cfg.harness.timeout_seconds,
            max_retries=cfg.harness.max_retries,
        ),
        prompt_proposer=Agent(
            make_prompt_proposer_options(project_root=cfg.project_root, model=cfg.harness.model),
            PromptProposerResponse,
            timeout_seconds=cfg.harness.timeout_seconds,
            max_retries=cfg.harness.max_retries,
        ),
        skill_generator=Agent(
            make_skill_generator_options(project_root=cfg.project_root, model=cfg.harness.model),
            ToolGeneratorResponse,
            timeout_seconds=cfg.harness.timeout_seconds,
            max_retries=cfg.harness.max_retries,
        ),
        prompt_generator=Agent(
            make_prompt_generator_options(project_root=cfg.project_root, model=cfg.harness.model),
            PromptGeneratorResponse,
            timeout_seconds=cfg.harness.timeout_seconds,
            max_retries=cfg.harness.max_retries,
        ),
    )

    loop_config = LoopConfig(
        max_iterations=cfg.evolution.iterations,
        frontier_size=cfg.evolution.frontier_size,
        no_improvement_limit=cfg.evolution.no_improvement_limit,
        concurrency=cfg.evolution.concurrency,
        evolution_mode=cfg.evolution.mode,
        failure_sample_count=cfg.evolution.failure_samples,
        categories_per_batch=cfg.evolution.failure_samples,
        use_llm_judge=True,
        judge_model=None,   # auto: openai provider → gpt-4o-mini
    )

    manager = ProgramManager(cwd=cfg.project_root)

    loop = SelfImprovingLoop(
        loop_config,
        agents,
        manager,
        train_pools,
        val_data,
        scorer=scorer,
        preloaded_trajectories=preloaded,
    )

    result = await loop.run()

    print(f"\n=== Done ===")
    print(f"  Best program : {result.best_program}")
    print(f"  Best score   : {result.best_score:.4f}")
    print(f"  Iterations   : {result.iterations_completed}")
    print(f"  Total cost   : ${result.total_cost_usd:.4f}")
    print(f"  Frontier     : {result.frontier}")


if __name__ == "__main__":
    asyncio.run(main())
