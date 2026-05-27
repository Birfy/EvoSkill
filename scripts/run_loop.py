#!/usr/bin/env python3
"""Run self-improving agent loop."""

import asyncio
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
import yaml
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.harness import Agent, set_sdk
from src.agent_profiles import (
    make_base_agent_options,
    make_skill_proposer_options,
    make_prompt_proposer_options,
    make_skill_generator_options,
    make_prompt_generator_options,
)
from src.agent_profiles.skill_generator import get_project_root
from src.registry import ProgramConfig, ProgramManager
from src.schemas import (
    AgentResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)


class LocalProgramManager:
    """Filesystem-only program manager for offline evolution on dirty worktrees."""

    PROGRAM_FILE = ".claude/program.yaml"

    def __init__(self, cwd: str | Path | None = None):
        self.cwd = Path(cwd or Path.cwd()).resolve()
        self.root = self.cwd / ".evoskill" / "local_programs"
        self.programs_dir = self.root / "programs"
        self.frontier_path = self.root / "frontier.json"
        self.state_path = self.root / "state.json"
        self.programs_dir.mkdir(parents=True, exist_ok=True)

    def create_program(
        self,
        name: str,
        config: ProgramConfig,
        parent: str | None = None,
    ) -> str:
        if name in self.list_programs():
            raise ValueError(
                f"Local program already exists: {name}. "
                "Use --continue_loop true to continue an existing run, or reset local state first."
            )
        if parent:
            self.switch_to(parent)
        program_dir = self._program_dir(name)
        program_dir.mkdir(parents=True, exist_ok=True)
        self._write_program_config(name, config)
        self._write_current_config(config)
        self._snapshot_current_claude(name)
        self._write_state(current=name)
        return f"program/{name}"

    def reset(self) -> None:
        """Remove all local program snapshots/frontier state for a fresh run."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.programs_dir.mkdir(parents=True, exist_ok=True)

    def switch_to(self, name: str) -> None:
        if name not in self.list_programs():
            raise ValueError(f"Unknown local program: {name}")
        self._restore_claude_snapshot(name)
        self._write_state(current=name)

    def get_current(self) -> ProgramConfig:
        current = self._read_state().get("current")
        if current and current in self.list_programs():
            return self._read_program_config(current)
        return self._read_current_config()

    def list_programs(self) -> list[str]:
        if not self.programs_dir.exists():
            return []
        return sorted(p.name for p in self.programs_dir.iterdir() if p.is_dir())

    def get_frontier(self) -> list[str]:
        return [name for name, _score in self.get_frontier_with_scores()]

    def get_frontier_with_scores(self) -> list[tuple[str, float]]:
        frontier = self._read_frontier()
        scored: list[tuple[str, float]] = []
        for name in frontier:
            if name not in self.list_programs():
                continue
            score = self._read_program_config(name).get_score()
            if score is not None:
                scored.append((name, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def select_from_frontier(self, strategy: str, iteration: int = 0) -> str | None:
        scored = self.get_frontier_with_scores()
        if not scored:
            return None
        if strategy == "random":
            return random.choice(scored)[0]
        if strategy == "round_robin":
            return scored[iteration % len(scored)][0]
        return scored[0][0]

    def get_best_from_frontier(self) -> str | None:
        scored = self.get_frontier_with_scores()
        return scored[0][0] if scored else None

    def update_frontier(self, name: str, score: float, max_size: int = 5) -> bool:
        config = self._read_program_config(name).with_score(score)
        self._write_program_config(name, config)
        if self._read_state().get("current") == name:
            self._write_current_config(config)

        frontier = self._read_frontier()
        if name not in frontier:
            frontier.append(name)
        frontier = [
            n
            for n, _s in sorted(
                (
                    (n, self._read_program_config(n).get_score())
                    for n in frontier
                    if n in self.list_programs()
                ),
                key=lambda item: item[1] if item[1] is not None else -1.0,
                reverse=True,
            )
            if self._read_program_config(n).get_score() is not None
        ]
        kept = frontier[:max_size]
        self._write_frontier(kept)
        return name in kept

    def discard(self, name: str) -> None:
        current = self._read_state().get("current")
        if current == name:
            replacement = self.get_best_from_frontier()
            if replacement == name:
                remaining = [n for n in self.list_programs() if n != name]
                replacement = remaining[0] if remaining else None
            if replacement:
                self.switch_to(replacement)

        program_dir = self._program_dir(name)
        if program_dir.exists():
            shutil.rmtree(program_dir)
        self._write_frontier([n for n in self._read_frontier() if n != name])

    def commit(self, message: str | None = None) -> bool:
        current = self._read_state().get("current")
        if not current:
            return False
        self._snapshot_current_claude(current)
        return True

    def _program_dir(self, name: str) -> Path:
        return self.programs_dir / name

    def _program_config_path(self, name: str) -> Path:
        return self._program_dir(name) / "program.yaml"

    def _program_claude_dir(self, name: str) -> Path:
        return self._program_dir(name) / ".claude"

    def _write_program_config(self, name: str, config: ProgramConfig) -> None:
        path = self._program_config_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(config.model_dump(), sort_keys=False), encoding="utf-8")

    def _read_program_config(self, name: str) -> ProgramConfig:
        data = yaml.safe_load(self._program_config_path(name).read_text(encoding="utf-8"))
        return ProgramConfig.model_validate(data)

    def _write_current_config(self, config: ProgramConfig) -> None:
        path = self.cwd / self.PROGRAM_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(config.model_dump(), sort_keys=False), encoding="utf-8")

    def _read_current_config(self) -> ProgramConfig:
        data = yaml.safe_load((self.cwd / self.PROGRAM_FILE).read_text(encoding="utf-8"))
        return ProgramConfig.model_validate(data)

    def _snapshot_current_claude(self, name: str) -> None:
        src = self.cwd / ".claude"
        dst = self._program_claude_dir(name)
        if dst.exists():
            shutil.rmtree(dst)
        if src.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))

    def _restore_claude_snapshot(self, name: str) -> None:
        src = self._program_claude_dir(name)
        dst = self.cwd / ".claude"
        if dst.exists():
            shutil.rmtree(dst)
        if src.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        config = self._read_program_config(name)
        self._write_current_config(config)

    def _read_frontier(self) -> list[str]:
        if not self.frontier_path.exists():
            return []
        return json.loads(self.frontier_path.read_text(encoding="utf-8"))

    def _write_frontier(self, frontier: list[str]) -> None:
        self.frontier_path.parent.mkdir(parents=True, exist_ok=True)
        self.frontier_path.write_text(json.dumps(frontier, indent=2), encoding="utf-8")

    def _read_state(self) -> dict[str, str]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, **state: str) -> None:
        data = {**self._read_state(), **state}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class LoopSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,
        title="Run self-improving agent loop",
    )

    mode: Literal["skill_only", "prompt_only"] = Field(
        default="skill_only",
        description="Evolution mode: 'skill_only' or 'prompt_only'",
    )
    max_iterations: int = Field(
        default=20, description="Maximum number of improvement iterations"
    )
    frontier_size: int = Field(
        default=3, description="Number of top-performing programs to keep"
    )
    no_improvement_limit: int = Field(
        default=5, description="Stop after this many iterations without improvement"
    )
    concurrency: int = Field(default=4, description="Number of concurrent evaluations")
    failure_samples: int = Field(
        default=3,
        description="Number of samples to test per iteration for pattern detection",
    )
    cache: bool = Field(default=True, description="Enable run caching")
    reset_feedback: bool = Field(
        default=True, description="Reset feedback history on start"
    )
    continue_loop: bool = Field(
        default=False,
        description="Continue from existing frontier/branch instead of starting fresh",
    )
    dataset: Optional[str] = Field(
        default=".dataset/new_runs_base/solved_dataset.csv",
        description="Path to dataset CSV with category column",
    )
    train_ratio: float = Field(
        default=0.18, description="Fraction of each category for training"
    )
    val_ratio: float = Field(
        default=0.12, description="Fraction of each category for validation"
    )
    val_count: Optional[int] = Field(
        default=None, description="Override total validation count"
    )
    model: Optional[str] = Field(
        default=None, description="Model for base agent (opus, sonnet, haiku)"
    )
    sdk: Literal["claude", "opencode", "codex", "goose", "openhands"] = Field(
        default="claude",
        description="SDK to use: 'claude', 'opencode', 'codex', or 'goose'",
    )
    use_llm_judge: bool = Field(
        default=False,
        description=(
            "Enable LLM Judge mode: run all trajectories once upfront (no train/val split), "
            "then use a lightweight LLM judge call to evaluate skill changes instead of "
            "re-running the full agent on a validation set."
        ),
    )
    judge_model: Optional[str] = Field(
        default=None,
        description=(
            "Model for judge calls (direct API, no agent). "
            "Defaults to a small model for the active SDK provider "
            "(Anthropic → claude-haiku-4-5-20251001, OpenAI → gpt-4o-mini)."
        ),
    )
    judge_scoring: Literal["elo", "average", "bradley_terry"] = Field(
        default="elo",
        description=(
            "LLM judge scoring mode. 'elo' treats each judged failure as a "
            "pairwise challenger-vs-baseline match; 'bradley_terry' refits a "
            "global pairwise-rating league over all tree nodes; 'average' uses "
            "the older mean fixed-failure confidence."
        ),
    )
    judge_elo_k: float = Field(
        default=128.0,
        description="Elo K-factor used by --judge_scoring elo.",
    )
    judge_elo_scale: float = Field(
        default=400.0,
        description="Elo logistic scale used by --judge_scoring elo.",
    )
    judge_log_details: bool = Field(
        default=True,
        description="Print per-sample judge JSON, match scores, and Elo updates.",
    )
    judge_input_cost_per_1m: Optional[float] = Field(
        default=None,
        description=(
            "Optional direct judge input-token price in USD per 1M tokens. "
            "When omitted, known OpenAI model prices are used when available."
        ),
    )
    judge_output_cost_per_1m: Optional[float] = Field(
        default=None,
        description=(
            "Optional direct judge output-token price in USD per 1M tokens. "
            "When omitted, known OpenAI model prices are used when available."
        ),
    )
    puct_c: float = Field(
        default=0.5,
        description="PUCT exploration constant for LLM-judge run-loop tree search.",
    )
    puct_max_depth: int = Field(
        default=3,
        description="Maximum depth for LLM-judge PUCT program tree search.",
    )
    puct_children_per_node: int = Field(
        default=2,
        description="Maximum children to expand from each PUCT program node.",
    )
    children_per_iteration: int = Field(
        default=2,
        description=(
            "Number of candidate child programs to generate from the selected "
            "PUCT parent per iteration, capped by remaining puct_children_per_node slots."
        ),
    )
    puct_default_prior: float = Field(
        default=0.5,
        description=(
            "Default PUCT policy prior for newly generated children. This is "
            "kept independent from judge scores so judge value only affects Q."
        ),
    )
    trajectories_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory containing trajectories.jsonl produced by collect_trajectories.py. "
            "When set, the loop loads pre-collected trajectories and skips all agent inference "
            "(requires --use_llm_judge). "
            "Generate with: python scripts/collect_trajectories.py --dataset <csv> --output_dir <dir>"
        ),
    )
    project_root: Optional[str] = Field(
        default=None,
        description=(
            "Project root containing .claude/.evoskill. "
            "Defaults to auto-detection from cwd."
        ),
    )


def stratified_split(
    data: pd.DataFrame, train_ratio: float = 0.18, val_ratio: float = 0.12
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    """Split data ensuring each category has at least 1 in both train and validation.

    Args:
        data: DataFrame with 'question', 'ground_truth', 'category' columns.
        train_ratio: Fraction of each category to use for training.
        val_ratio: Fraction of each category to use for validation.

    Returns:
        train_pools: Dict mapping category -> list of (question, answer) tuples.
        val_data: List of (question, answer, category) tuples for validation.
    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) cannot exceed 1.0"
        )

    # Drop rows with missing categories
    data = data.dropna(subset=["category"])
    categories = data["category"].unique()
    train_pools: dict[str, list[tuple[str, str]]] = {}
    val_data: list[tuple[str, str, str]] = []

    for cat in categories:
        cat_data = data[data["category"] == cat].sample(frac=1, random_state=42)
        n_train = max(1, int(len(cat_data) * train_ratio))
        n_val = max(1, int(len(cat_data) * val_ratio))

        # Train comes first, then validation
        train_pools[cat] = [
            (row.question, row.ground_truth)
            for _, row in cat_data.head(n_train).iterrows()
        ]
        val_data.extend(
            [
                (row.question, row.ground_truth, cat)
                for _, row in cat_data.iloc[n_train : n_train + n_val].iterrows()
            ]
        )

    return train_pools, val_data


def train_pools_from_trajectories(
    trajectories: list[tuple[Any, str, str, str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Build the minimal train pool required by the loop from stored trajectories."""
    train_pools: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for _trace, question, _agent_answer, ground_truth, category in trajectories:
        key = (question, ground_truth, category)
        if key in seen:
            continue
        seen.add(key)
        train_pools.setdefault(category, []).append((question, ground_truth))
    return train_pools


async def main(settings: LoopSettings):
    # Set SDK based on CLI argument
    set_sdk(settings.sdk)

    # Load pre-collected trajectories early so they can also act as the
    # training pool for compatibility wrappers that do not have a dataset CSV.
    preloaded_trajectories = None
    if settings.trajectories_dir:
        if not settings.use_llm_judge:
            raise ValueError("--trajectories_dir requires --use_llm_judge")
        print(f"Loading trajectories from {settings.trajectories_dir}...")
        preloaded_trajectories = SelfImprovingLoop.load_trajectories_from_dir(
            settings.trajectories_dir
        )
        print(f"Loaded {len(preloaded_trajectories)} trajectories")

    if settings.dataset and Path(settings.dataset).exists():
        data = pd.read_csv(settings.dataset)
        train_pools, val_data = stratified_split(
            data, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio
        )
        dataset_label = settings.dataset
    elif preloaded_trajectories is not None:
        train_pools = train_pools_from_trajectories(preloaded_trajectories)
        val_data = []
        dataset_label = "derived from pre-collected trajectories"
    else:
        dataset = settings.dataset or "<not set>"
        raise FileNotFoundError(
            f"Dataset not found: {dataset}. Provide --dataset or --trajectories_dir with --use_llm_judge."
        )

    # Print category distribution
    categories = list(train_pools.keys())
    total_train = sum(len(pool) for pool in train_pools.values())
    print(f"Dataset: {dataset_label}")
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(
        f"Training pools: {', '.join(f'{cat}: {len(pool)}' for cat, pool in train_pools.items())}"
    )
    print(f"Total training samples: {total_train}")
    print(
        f"Validation samples: {len(val_data)} ({settings.val_ratio:.0%} per category, min 1 each)"
    )
    if val_data:
        print(
            f"Split ratios: train={settings.train_ratio:.0%}, val={settings.val_ratio:.0%} (remaining {1 - settings.train_ratio - settings.val_ratio:.0%} unused)"
        )

    # Use custom model for base agent if specified
    project_root = (
        str(Path(settings.project_root).resolve())
        if settings.project_root
        else get_project_root()
    )
    base_options = make_base_agent_options(
        model=settings.model,
        project_root=project_root,
    )

    agents = LoopAgents(
        base=Agent(base_options, AgentResponse),
        skill_proposer=Agent(
            make_skill_proposer_options(project_root=project_root, model=settings.model),
            SkillProposerResponse,
        ),
        prompt_proposer=Agent(
            make_prompt_proposer_options(project_root=project_root, model=settings.model),
            PromptProposerResponse,
        ),
        skill_generator=Agent(
            make_skill_generator_options(project_root=project_root, model=settings.model),
            ToolGeneratorResponse,
        ),
        prompt_generator=Agent(
            make_prompt_generator_options(project_root=project_root, model=settings.model),
            PromptGeneratorResponse,
        ),
    )
    if os.getenv("EVOSKILL_NO_GIT") == "1":
        print("Using LocalProgramManager (EVOSKILL_NO_GIT=1)")
        manager = LocalProgramManager(cwd=project_root)
    else:
        manager = ProgramManager(cwd=project_root)

    config = LoopConfig(
        max_iterations=settings.max_iterations,
        frontier_size=settings.frontier_size,
        no_improvement_limit=settings.no_improvement_limit,
        concurrency=settings.concurrency,
        evolution_mode=settings.mode,
        failure_sample_count=settings.failure_samples,
        categories_per_batch=settings.failure_samples,  # Sample from N different categories
        cache_enabled=settings.cache,
        reset_feedback=settings.reset_feedback,
        continue_mode=settings.continue_loop,
        use_llm_judge=settings.use_llm_judge,
        judge_model=settings.judge_model,
        judge_scoring=settings.judge_scoring,
        judge_elo_k=settings.judge_elo_k,
        judge_elo_scale=settings.judge_elo_scale,
        judge_log_details=settings.judge_log_details,
        judge_input_cost_per_1m=settings.judge_input_cost_per_1m,
        judge_output_cost_per_1m=settings.judge_output_cost_per_1m,
        puct_c=settings.puct_c,
        puct_max_depth=settings.puct_max_depth,
        puct_children_per_node=settings.puct_children_per_node,
        children_per_iteration=settings.children_per_iteration,
        puct_default_prior=settings.puct_default_prior,
    )

    model_info = f", model={settings.model}" if settings.model else ""
    traj_info = f", trajectories={settings.trajectories_dir}" if settings.trajectories_dir else ""
    print(f"Running loop with evolution_mode={settings.mode}{model_info}{traj_info}")
    loop = SelfImprovingLoop(
        config,
        agents,
        manager,
        train_pools,
        val_data,
        preloaded_trajectories=preloaded_trajectories,
    )
    result = await loop.run()

    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")
    print(
        "Cost: "
        f"total=${result.total_cost_usd:.4f}, "
        f"trajectory=${result.trajectory_cost_usd:.4f}, "
        f"preloaded=${result.preloaded_trajectory_cost_usd:.4f}, "
        f"evolution=${result.evolution_agent_cost_usd:.4f}, "
        f"judge=${result.judge_cost_usd:.4f}, "
        f"judge_tokens={result.judge_total_tokens} "
        f"(in={result.judge_prompt_tokens}, out={result.judge_completion_tokens})"
    )
    return result


if __name__ == "__main__":
    settings = LoopSettings()
    asyncio.run(main(settings))
