"""Self-improving agent loop runner."""

import asyncio
import difflib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from src.harness import Agent, AgentTrace, is_claude_sdk, is_opencode_sdk, is_openhands_sdk, is_goose_sdk, is_codex_sdk
from src.cache import RunCache, CacheConfig
from src.registry.sdk_utils import options_to_config


def _log(phase: str, message: str = "", indent: int = 0) -> None:
    """Print a structured log message.

    Args:
        phase: Phase marker (e.g., "INIT", "ITER 1/5", "DONE") or empty for continuation.
        message: The message to display.
        indent: Indentation level (each level = 2 spaces).
    """
    prefix = "  " * indent
    if phase:
        print(f"\n{prefix}[{phase}] {message}")
    else:
        print(f"{prefix}{message}")


def _score_multi_tolerance(question: str, predicted: str, ground_truth: str) -> float:
    """Score answer using weighted average across tolerance levels.

    Weights favor stricter tolerances: weight = 1 / (1 + 20 * tolerance)
    This gives approximate weights:
      - 0.0%  tolerance: 1.00 (exact match, highest priority)
      - 1.0%  tolerance: 0.83
      - 2.5%  tolerance: 0.67
      - 5.0%  tolerance: 0.50
      - 10.0% tolerance: 0.33
    """
    if not str(predicted or "").strip():
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    for tol in TOLERANCE_LEVELS:
        weight = 1.0 / (1.0 + 20.0 * tol)
        score = score_answer(ground_truth, predicted, tol)
        weighted_sum += weight * score
        weight_total += weight
    return weighted_sum / weight_total


from src.evaluation import score_answer, evaluate_agent_parallel
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    ProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
    SkillProposerResponse,
    PromptProposerResponse,
)

from .config import LoopConfig
from .helpers import (
    build_answer_comparison_feedback,
    build_regression_success_feedback,
    build_proposer_query,
    build_skill_query,
    build_prompt_query,
    build_skill_query_from_skill_proposer,
    build_prompt_query_from_prompt_proposer,
    build_judge_query,
    build_skill_revision_query,
    append_feedback,
    read_feedback_history,
    update_prompt_file,
)


T = TypeVar("T")

TOLERANCE_LEVELS = [0.05, 0.01, 0.1, 0.0, 0.025]

# USD per 1M tokens for direct judge API cost estimation.
# These are intentionally conservative model-name mappings; explicit config
# overrides still win when a provider or account uses a different route.
OPENAI_JUDGE_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass
class MutationResult:
    """Generated child program plus proposer analysis used by later stages."""

    child_name: str
    proposal: str
    justification: str
    proposer_confidence: float
    root_cause_analysis: str = ""
    coverage_plan: str = ""
    regression_risks: str = ""
    skill_name: str = ""


@dataclass
class LoopAgents:
    """Container for the agents used in the loop."""

    base: Agent[AgentResponse]
    skill_proposer: Agent[SkillProposerResponse]
    prompt_proposer: Agent[PromptProposerResponse]
    skill_generator: Agent[ToolGeneratorResponse]
    prompt_generator: Agent[PromptGeneratorResponse]


@dataclass
class LoopResult:
    """Result of running the self-improving loop."""

    frontier: list[tuple[str, float]]
    best_program: str
    best_score: float
    iterations_completed: int
    total_cost_usd: float = 0.0
    trajectory_cost_usd: float = 0.0
    preloaded_trajectory_cost_usd: float = 0.0
    evolution_agent_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_total_tokens: int = 0


@dataclass
class JudgeMatchResult:
    """One LLM-judged pairwise match used by Elo scoring."""

    index: int
    category: str
    would_succeed: bool
    confidence: float
    match_score: float
    expected_before: float
    child_rating_after: float
    opponent_rating_after: float
    hypothetical_action: str
    reasoning: str
    raw_response: str
    valid: bool = True
    root_cause: str = ""
    skill_addresses_root_cause: float = 0.0
    proposer_root_cause_correct: float = 0.0
    failure_mechanism_encoding: float = 0.0
    executable_specificity: float = 0.0
    high_risk_blacklist: float = 0.0
    generalization_transfer: float = 0.0
    probability_of_success: float = 0.0
    parent_success_prob: float = 0.0
    candidate_success_prob: float = 0.0
    relative_advantage: float = 0.0
    remaining_blockers: list[str] = field(default_factory=list)


@dataclass
class JudgeResult:
    """Aggregated LLM judge result."""

    estimated_fix_rate: float
    average_match_score: float
    child_rating: float
    opponent_rating: float
    expected_win_rate: float
    matches: list[JudgeMatchResult]
    uncertainty: float = 0.0


@dataclass(frozen=True)
class BradleyTerryMatch:
    """One soft pairwise comparison for the global Bradley-Terry league."""

    player: str
    opponent: str
    score: float
    category: str = ""
    index: int = 0


@dataclass
class ProgramSearchNode:
    """PUCT node for run-loop program evolution."""

    name: str
    parent: "ProgramSearchNode | None"
    prior: float = 0.5
    score: float = 0.0
    visit_count: int = 0
    total_q: float = 0.0
    depth: int = 0
    discarded: bool = False
    children: list["ProgramSearchNode"] = field(default_factory=list)
    # Failures this node was judged on — inherited by children so comparisons
    # are always made on a superset of the parent's evaluation set.
    scoring_failures: list = field(default_factory=list)

    @property
    def q_value(self) -> float:
        return self.total_q / self.visit_count if self.visit_count else self.score

    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        exploration = c_puct * self.prior * math.sqrt(max(parent_visits, 1)) / (1 + self.visit_count)
        return self.q_value + exploration


class SelfImprovingLoop:
    """Self-improving agent loop with git-based versioning.

    This class encapsulates the self-improving loop where:
    1. Base agent attempts to answer questions
    2. Failures are passed to the proposer to suggest skills or prompt changes
    3. Skill/prompt generator creates the proposed changes
    4. New mutations are evaluated and added to frontier if improved
    5. Loop continues until threshold or max iterations
    """

    def __init__(
        self,
        config: LoopConfig,
        agents: LoopAgents,
        manager: ProgramManager,
        train_pools: dict[str, list[tuple[str, str]]],
        val_data: list[tuple[str, str, str]],
        scorer: Callable[[str, str, str], float] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        task_constraints: str = "",
        preloaded_trajectories: list[tuple[Any, ...]] | None = None,
    ):
        """Initialize the self-improving loop.

        Args:
            config: Loop configuration parameters.
            agents: Container with the 4 agents (base, proposer, skill_generator, prompt_generator).
            manager: ProgramManager for git-based versioning.
            train_pools: Dict mapping category -> list of (question, answer) tuples.
                Used for category-aware sampling in the evolution loop.
                Pass an empty dict when supplying preloaded_trajectories.
            val_data: Validation data as list of (question, answer, category) tuples.
                Unused in LLM judge mode when preloaded_trajectories is provided.
            scorer: Scoring function (question, predicted, ground_truth) -> float.
                    Defaults to _score_multi_tolerance for backward compatibility.
            preloaded_trajectories: Pre-collected trajectories as a list of
                (AgentTrace, question, agent_answer, ground_truth, category) tuples.
                When provided and use_llm_judge=True, the loop skips agent inference
                entirely and uses these trajectories directly for skill evolution.
                Each tuple is (AgentTrace, question, agent_answer, ground_truth, category, failure_type[, failure_feedback]).
        """
        self.config = config
        self.agents = agents
        self.manager = manager
        self.train_pools = train_pools
        self.val_data = val_data
        self.scorer = scorer or _score_multi_tolerance
        self.on_event = on_event
        self.task_constraints = task_constraints
        self._preloaded_trajectories = preloaded_trajectories

        # Round-robin sampling state — seeded from preloaded categories if available
        self._category_offset = 0
        # Independent rotation for the held-out judge pool (LLM-judge mode), so
        # judge-case selection never shares offsets with proposer sampling.
        self._judge_cat_offset: dict[str, int] = {}
        # Coverage-priority rotation: how many times each failure (keyed by
        # question) has been shown to the proposer. Least-shown failures are
        # picked first so successive iterations sweep distinct failures instead
        # of repeatedly re-proposing on the same few.
        self._failure_shown_count: dict[str, int] = {}
        if preloaded_trajectories:
            cats = sorted({entry[4] for entry in preloaded_trajectories})
            self._per_cat_offset: dict[str, int] = {cat: 0 for cat in cats}
            failure_types = sorted(
                {
                    self._trajectory_failure_type(entry)
                    for entry in preloaded_trajectories
                    if self._trajectory_failure_type(entry)
                }
            )
            self._per_failure_type_offset: dict[str, int] = {ft: 0 for ft in failure_types}
        else:
            self._per_cat_offset = {cat: 0 for cat in train_pools.keys()}
            self._per_failure_type_offset = {}

        # Paths
        self._project_root = Path(getattr(self.manager, "cwd", Path.cwd())).resolve()
        self._feedback_path = self._project_root / ".evoskill" / "feedback_history.md"
        self._prompt_path = (
            self._project_root / "src" / "agent_profiles" / "base_agent" / "prompt.txt"
        )

        # Initialize cache
        if config.cache_enabled:
            cache_config = CacheConfig(
                cache_dir=config.cache_dir,
                enabled=True,
                store_messages=config.cache_store_messages,
                cwd=self._project_root,
            )
            self.cache: RunCache | None = RunCache(cache_config)
        else:
            self.cache = None

        # Iteration offset for continue mode
        self._iteration_offset = 0

        # Checkpoint file for exact resume
        self._checkpoint_path = self._project_root / ".evoskill" / "loop_checkpoint.json"

        # Cost tracking
        self._total_cost: float = 0.0
        self._iter_cost: float = 0.0
        self._trajectory_cost_usd: float = 0.0
        self._preloaded_trajectory_cost_usd: float = 0.0
        self._evolution_agent_cost_usd: float = 0.0
        self._judge_cost_usd: float = 0.0
        self._judge_prompt_tokens: int = 0
        self._judge_completion_tokens: int = 0
        self._judge_total_tokens: int = 0

    def _emit(self, event: str, **data: Any) -> None:
        """Fire an event to the display callback if one is registered."""
        if self.on_event is not None:
            self.on_event(event, data)

    def _add_iteration_cost(self, amount: float, bucket: str) -> None:
        """Track a cost in the current iteration and in a named aggregate bucket."""
        amount = float(amount or 0.0)
        if amount <= 0.0:
            return
        self._iter_cost = getattr(self, "_iter_cost", 0.0) + amount
        if bucket == "trajectory":
            self._trajectory_cost_usd = getattr(self, "_trajectory_cost_usd", 0.0) + amount
        elif bucket == "evolution":
            self._evolution_agent_cost_usd = getattr(self, "_evolution_agent_cost_usd", 0.0) + amount
        elif bucket == "judge":
            self._judge_cost_usd = getattr(self, "_judge_cost_usd", 0.0) + amount

    def _add_preloaded_trajectory_cost(self, amount: float) -> None:
        """Track costs already embedded in loaded trajectories."""
        amount = float(amount or 0.0)
        if amount <= 0.0:
            return
        self._preloaded_trajectory_cost_usd += amount
        self._trajectory_cost_usd += amount
        self._total_cost += amount

    def _build_loop_result(
        self,
        frontier: list[tuple[str, float]],
        best_program: str,
        best_score: float,
        iterations_completed: int,
    ) -> LoopResult:
        return LoopResult(
            frontier=frontier,
            best_program=best_program,
            best_score=best_score,
            iterations_completed=iterations_completed,
            total_cost_usd=self._total_cost,
            trajectory_cost_usd=self._trajectory_cost_usd,
            preloaded_trajectory_cost_usd=self._preloaded_trajectory_cost_usd,
            evolution_agent_cost_usd=self._evolution_agent_cost_usd,
            judge_cost_usd=self._judge_cost_usd,
            judge_prompt_tokens=self._judge_prompt_tokens,
            judge_completion_tokens=self._judge_completion_tokens,
            judge_total_tokens=self._judge_total_tokens,
        )

    def _format_cost_breakdown(self) -> str:
        return (
            f"Total cost: ${self._total_cost:.4f} "
            f"(trajectory=${self._trajectory_cost_usd:.4f}, "
            f"preloaded=${self._preloaded_trajectory_cost_usd:.4f}, "
            f"evolution=${self._evolution_agent_cost_usd:.4f}, "
            f"judge=${self._judge_cost_usd:.4f}; "
            f"judge_tokens={self._judge_total_tokens} "
            f"in={self._judge_prompt_tokens} out={self._judge_completion_tokens})"
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """Save sampling state for exact resume.

        Args:
            iteration: The iteration number just completed.
        """
        checkpoint = {
            "iteration": iteration,
            "category_offset": self._category_offset,
            "per_cat_offset": self._per_cat_offset,
        }
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path.write_text(json.dumps(checkpoint, indent=2))

    def _load_checkpoint(self) -> int | None:
        """Load checkpoint if exists.

        Returns:
            Iteration number to resume from, or None if no checkpoint exists.
        """
        if not self._checkpoint_path.exists():
            return None
        try:
            checkpoint = json.loads(self._checkpoint_path.read_text())
            self._category_offset = checkpoint["category_offset"]
            self._per_cat_offset = checkpoint["per_cat_offset"]
            return checkpoint["iteration"]
        except (json.JSONDecodeError, KeyError) as e:
            _log("WARN", f"Invalid checkpoint file, ignoring: {e}")
            return None

    def _delete_checkpoint(self) -> None:
        """Delete checkpoint file if it exists."""
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()

    async def run(self) -> LoopResult:
        """Run the full self-improving loop.

        Returns:
            LoopResult with frontier, best program, and iteration count.
        """
        if self.config.use_llm_judge:
            return await self._run_with_llm_judge()

        # 0. Handle continue mode and feedback reset
        resume_iteration: int | None = None
        if not self.config.continue_mode:
            # Start fresh: reset feedback if configured
            if self.config.reset_feedback and self._feedback_path.exists():
                self._feedback_path.unlink()
            self._iteration_offset = 0
            # Delete any existing checkpoint on fresh start
            self._delete_checkpoint()
            reset_manager = getattr(self.manager, "reset", None)
            if callable(reset_manager):
                reset_manager()
                _log("INIT", "Reset local program state for fresh run")
        else:
            # Continue mode: keep feedback, find highest iteration number
            self._iteration_offset = self._get_highest_iteration()
            # Try to load checkpoint for exact sampling state resume
            resume_iteration = self._load_checkpoint()
            if resume_iteration is not None:
                _log("CONTINUE", f"Resuming from iteration {resume_iteration} with exact sampling state")
            else:
                _log("CONTINUE", f"Resuming from iteration {self._iteration_offset} (no checkpoint, sampling state reset)")

        # Get sorted list of categories for deterministic round-robin
        categories = sorted(self.train_pools.keys())

        # 1. Create and evaluate base program if needed (skip in continue mode with existing frontier)
        if self.config.continue_mode and self.manager.get_frontier():
            # Continue mode: use existing frontier, switch to best program
            best = self._select_parent()
            self.manager.switch_to(best)
            frontier_str = ", ".join(f"{n}:{s:.2f}" for n, s in self.manager.get_frontier_with_scores())
            _log("CONTINUE", f"Using existing frontier: [{frontier_str}]")
        else:
            await self._ensure_base_program()

        # 2. Main loop
        no_improvement_count = 0
        iteration_count = 0
        n_cats = len(categories)

        for i in range(self.config.max_iterations):
            iteration_count = i + 1
            actual_iteration = iteration_count + self._iteration_offset

            # Skip already-completed iterations when resuming with checkpoint
            if resume_iteration is not None and actual_iteration <= resume_iteration:
                continue

            # Select parent from frontier using configured strategy
            parent = self._select_parent(iteration_count)
            self.manager.switch_to(parent)
            self._iter_cost = 0.0  # Reset per-iteration cost
            _log(f"ITER {iteration_count}/{self.config.max_iterations}", f"Parent: {parent}")
            self._emit("iter_start", iteration=actual_iteration, total=self.config.max_iterations, parent=parent)

            # Round-robin sampling: pick samples_per_category from each of N categories (cycling)
            n_cats_this_iter = min(self.config.categories_per_batch, n_cats)

            test_samples: list[tuple[str, str, str]] = []
            sampled_cats: list[str] = []
            for j in range(n_cats_this_iter):
                cat_idx = (self._category_offset + j) % n_cats
                cat = categories[cat_idx]
                pool = self.train_pools[cat]

                # Take min(samples_per_category, pool_size) to handle small categories
                samples_to_take = min(self.config.samples_per_category, len(pool))

                for _ in range(samples_to_take):
                    sample_idx = self._per_cat_offset[cat] % len(pool)
                    question, answer = pool[sample_idx]
                    test_samples.append((question, answer, cat))
                    sampled_cats.append(cat)
                    self._per_cat_offset[cat] += 1

            self._category_offset += n_cats_this_iter

            _log("", f"  Testing {len(test_samples)} samples from categories: {', '.join(sampled_cats)}...")

            # Run all samples concurrently
            traces = await asyncio.gather(*[
                self.agents.base.run(question) for question, _, _ in test_samples
            ])
            self._add_iteration_cost(
                sum(t.total_cost_usd for t in traces),
                "trajectory",
            )

            # Collect failures
            failures: list[tuple[AgentTrace, str, str, str]] = []  # (trace, agent_answer, ground_truth, category)
            # Parallel (question, answer, category) for the failed samples, used as
            # the in-sample term when scoring the child (did it fix what it saw?).
            in_sample_samples: list[tuple[str, str, str]] = []
            for trace, (question, answer, category) in zip(traces, test_samples):
                agent_answer = (
                    trace.output.final_answer if trace.output and trace.output.final_answer else "[PARSE FAILED]"
                )
                avg_score = self.scorer(
                    question,
                    agent_answer.strip().lower(),
                    answer.strip().lower(),
                )
                threshold = self.config.failure_threshold
                status = "[OK]" if avg_score >= threshold else "[FAIL]"
                if self.on_event is None:
                    _log("", f"    {status} [{category}] {question[:40]}...")
                self._emit("sample", question=question, category=category, score=avg_score, passed=avg_score >= threshold)
                if avg_score < threshold:
                    failures.append((trace, agent_answer, answer, category))
                    in_sample_samples.append((question, answer, category))

            # Always propose if any failures exist
            if len(failures) == 0:
                _log("", f"  -> All samples passed, no proposal needed")
                continue

            _log("", f"  -> {len(failures)} failure(s), proposing improvement...")

            # Get parent's score for comparison
            parent_score = next(
                (score for name, score in self.manager.get_frontier_with_scores() if name == parent),
                0.0
            )

            # Run proposer with all failures (use actual iteration number with offset)
            mutation_result = await self._mutate_with_fallback(parent, failures, actual_iteration)

            if mutation_result is None:
                no_improvement_count += 1
            else:
                child_name = mutation_result.child_name
                proposal = mutation_result.proposal
                justification = mutation_result.justification

                # Evaluate child: blend in-sample fix rate with held-out generalization
                _log("", f"  -> Evaluating {child_name}...")
                child_score = await self._score_child(in_sample_samples)  # accumulates to self._iter_cost

                # Update frontier or discard
                added = self.manager.update_frontier(
                    child_name, child_score, max_size=self.config.frontier_size
                )

                if added:
                    _log("", f"  [OK] Added to frontier (score: {child_score:.4f})")
                    outcome = "improved" if child_score > parent_score else "kept"
                    no_improvement_count = 0
                else:
                    _log("", f"  [SKIP] Discarded (score: {child_score:.4f})")
                    outcome = "discarded"
                    self.manager.discard(child_name)
                    no_improvement_count += 1

                self._emit(
                    "eval_result",
                    child_name=child_name,
                    score=child_score,
                    parent_score=parent_score,
                    added=added,
                    frontier=self.manager.get_frontier_with_scores(),
                    n_skills=len(self._get_active_skills()),
                )

                # Record feedback with outcome for future proposers to learn from
                active_skills = self._get_active_skills()
                append_feedback(
                    self._feedback_path,
                    child_name,
                    proposal,
                    justification,
                    outcome=outcome,
                    score=child_score,
                    parent_score=parent_score,
                    active_skills=active_skills,
                )

            # Check early stopping
            if no_improvement_count >= self.config.no_improvement_limit:
                _log("STOP", f"No improvement for {self.config.no_improvement_limit} iterations")
                break

            # Print frontier status
            frontier_str = ", ".join(f"{n}:{s:.2f}" for n, s in self.manager.get_frontier_with_scores())
            _log("", f"  Frontier: [{frontier_str}]")

            # Report per-iteration and cumulative cost
            self._total_cost += self._iter_cost
            _log("COST", f"Iter {iteration_count} cost: ${self._iter_cost:.4f} | Running total: ${self._total_cost:.4f}")

            # Save checkpoint at end of each successful iteration
            self._save_checkpoint(actual_iteration)

        # 3. Return results
        frontier = self.manager.get_frontier_with_scores()
        best = self.manager.get_best_from_frontier()
        best_score = frontier[0][1] if frontier else 0.0

        _log("DONE", f"{iteration_count} iterations, best: {best or 'base'} ({best_score:.4f})")
        _log("COST", self._format_cost_breakdown())
        self._emit("loop_done", best=best or "base", best_score=best_score, iterations=iteration_count)

        return self._build_loop_result(
            frontier,
            best or "base",
            best_score,
            iteration_count,
        )

    async def _ensure_base_program(self) -> None:
        """Create and evaluate base program if it doesn't exist."""
        if "base" not in self.manager.list_programs():
            current_options = self.agents.base._get_options()
            base_config = options_to_config(current_options, "base")
            self.manager.create_program("base", base_config)
            _log("INIT", "Created base program")
        else:
            _log("INIT", "Using existing base program")

        # Evaluate and add base to frontier
        self.manager.switch_to("base")
        _log("", f"  -> Evaluating on {len(self.val_data)} samples...")
        self._iter_cost = 0.0
        base_score = await self._evaluate(self.val_data)
        self._total_cost += self._iter_cost
        self.manager.update_frontier(
            "base", base_score, max_size=self.config.frontier_size
        )
        _log("", f"  -> Base score: {base_score:.4f}")
        _log("", f"  -> Frontier: {self.manager.get_frontier()}")
        _log("COST", f"Base eval cost: ${self._iter_cost:.4f} | {self._format_cost_breakdown()}")
        self._emit("baseline", score=base_score, n_skills=len(self._get_active_skills()))

    def _generalization_samples(
        self, in_sample_samples: list[tuple[str, str, str]]
    ) -> list[tuple[str, str, str]]:
        """Pick held-out samples the proposer never saw this iteration.

        Prefers a dedicated validation split (``val_data``). When none exists
        (e.g. trajectory-preloaded runs put everything in the training pools),
        fall back to a deterministic slice of the training pools that excludes
        the in-sample failure questions, capped by ``generalization_sample_count``.
        """
        if self.val_data:
            return self.val_data

        seen = {q for q, _, _ in in_sample_samples}
        held_out: list[tuple[str, str, str]] = []
        cap = max(0, self.config.generalization_sample_count)
        if cap == 0:
            return []
        # Round-robin across categories for a balanced held-out batch.
        for cat in sorted(self.train_pools.keys()):
            for question, answer in self.train_pools[cat]:
                if question not in seen:
                    held_out.append((question, answer, cat))
        return held_out[:cap]

    async def _score_child(
        self, in_sample_samples: list[tuple[str, str, str]]
    ) -> float:
        """Score a freshly mutated child by blending two signals.

        ``child_score = w * in_sample_fix_rate + (1-w) * generalization_rate``

        - in-sample term: did the candidate fix the failures it was proposed for?
        - generalization term: does it hold up on unseen held-out samples?

        Degrades gracefully when one term is unavailable (e.g. no held-out data).
        """
        weight = min(1.0, max(0.0, self.config.in_sample_score_weight))

        in_sample_score = (
            await self._evaluate(in_sample_samples) if in_sample_samples else None
        )
        gen_samples = self._generalization_samples(in_sample_samples)
        gen_score = await self._evaluate(gen_samples) if gen_samples else None

        if in_sample_score is None and gen_score is None:
            _log("", "  [WARN] No samples to score child; defaulting to 0.0")
            return 0.0
        if gen_score is None:
            _log(
                "",
                f"  -> Child score (in-sample only, no held-out): {in_sample_score:.4f}",
            )
            return in_sample_score
        if in_sample_score is None:
            return gen_score

        blended = weight * in_sample_score + (1.0 - weight) * gen_score
        _log(
            "",
            (
                f"  -> Child score {blended:.4f} = {weight:.2f}*in_sample({in_sample_score:.4f})"
                f" + {1.0 - weight:.2f}*generalization({gen_score:.4f}; "
                f"n={len(gen_samples)})"
            ),
        )
        return blended

    async def _evaluate(self, data: list[tuple[str, str, str]]) -> float:
        """Evaluate base agent on data.

        Args:
            data: List of (question, answer, category) tuples.

        Returns:
            Accuracy score (0.0 to 1.0).
        """
        # Convert to (question, answer) format for evaluate_agent_parallel
        qa_data = [(q, a) for q, a, _ in data]
        results = await evaluate_agent_parallel(
            self.agents.base, qa_data, max_concurrent=self.config.concurrency, cache=self.cache
        )

        score = 0.0
        for result in results:
            if result.trace is not None:
                self._add_iteration_cost(result.trace.total_cost_usd, "trajectory")
            if result.trace is None or result.trace.output is None:
                continue  # Timeout/error/parse failed = 0 score
            score += self.scorer(
                result.question,
                result.trace.output.final_answer,
                result.ground_truth,
            )
        return score / len(results)

    async def _mutate(
        self,
        parent: str,
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str]],
        iteration: int | str,
        truncation_level: int = 0,
        diversity_hint: str = "",
        questions: list[str] | None = None,
    ) -> tuple[str, str, str, float] | None:
        """Run proposer and generator to create a mutation based on multiple failures.

        Args:
            parent: Name of the parent program.
            failures: List of (trace, agent_answer, ground_truth, category) tuples from failed attempts.
            iteration: Current iteration number.
            truncation_level: Context reduction level (0=full, 1=moderate, 2=aggressive).
            diversity_hint: Optional instruction used to diversify sibling children.
            questions: Optional parallel list of question texts for the proposer.

        Returns:
            MutationResult if created, None otherwise.
            if created, None otherwise.
        """
        # Calculate actual iteration number (with offset for continue mode)
        actual_iteration = (
            iteration + self._iteration_offset
            if isinstance(iteration, int)
            else iteration
        )

        # Run appropriate proposer based on evolution mode
        evolution_mode = self.config.evolution_mode
        _log("", f"  -> Running {evolution_mode.replace('_only', '')} proposer with {len(failures)} failures...")
        feedback_history = read_feedback_history(self._feedback_path)
        proposer_query = build_proposer_query(
            failures,
            feedback_history,
            evolution_mode,
            truncation_level,
            self.task_constraints,
            project_root=self._project_root,
            diversity_hint=diversity_hint,
            questions=questions,
            domain_hints=self.config.error_surface_hints,
        )

        if evolution_mode == "skill_only":
            proposer_trace = await self.agents.skill_proposer.run(proposer_query)
            self._add_iteration_cost(proposer_trace.total_cost_usd, "evolution")

            if proposer_trace.output is None:
                _log("", f"  [WARN] Skill proposer failed: {proposer_trace.parse_error}")
                return None

            proposer_output = proposer_trace.output
            proposed = proposer_output.proposed_skill
            justification = proposer_output.justification
            root_cause_analysis = getattr(proposer_output, "root_cause_analysis", "") or ""
            coverage_plan = getattr(proposer_output, "coverage_plan", "") or ""
            should_apply_when = getattr(proposer_output, "should_apply_when", "") or ""
            should_not_apply_when = getattr(proposer_output, "should_not_apply_when", "") or ""
            invariants_to_preserve = getattr(proposer_output, "invariants_to_preserve", "") or ""
            regression_risks = getattr(proposer_output, "regression_risks", "") or ""
            required_boundaries = {
                "should_apply_when": should_apply_when,
                "should_not_apply_when": should_not_apply_when,
                "invariants_to_preserve": invariants_to_preserve,
                "regression_risks": regression_risks,
            }
            missing_boundaries = [
                name for name, value in required_boundaries.items() if not value.strip()
            ]
            if missing_boundaries:
                _log(
                    "",
                    (
                        "  [WARN] Skill proposer missing boundary fields; "
                        f"skip generator: {', '.join(missing_boundaries)}"
                    ),
                )
                return None
            proposer_confidence = self._clamp01(
                getattr(proposer_output, "confidence", None),
                default=self.config.puct_default_prior,
            )
            action_type = proposer_output.action
            target_skill = proposer_output.target_skill

            action_label = f"edit:{target_skill}" if action_type == "edit" else "create"
            _log(
                "",
                (
                    f"  -> Proposal: skill ({action_label}, prior={proposer_confidence:.3f}) "
                    f"- {proposed[:50]}..."
                ),
            )
            self._emit("proposal", action=action_type, target_skill=target_skill, summary=proposed[:80])

            # Create child program branch
            child_name = f"iter-skill-{actual_iteration}"
            self.manager.switch_to(parent)
            parent_config = self.manager.get_current()
            child_config = parent_config.mutate(child_name)
            self.manager.create_program(child_name, child_config, parent=parent)

            # Generate skill - use different query for edit vs create
            if action_type == "edit" and target_skill:
                _log("", f"  -> Editing existing skill: {target_skill}...")
                skill_query = f"""EDIT existing skill: {target_skill}

Modifications needed:
{proposed}

Justification: {justification}

Root cause analysis from proposer:
{root_cause_analysis or "[not provided]"}

Failure coverage plan:
{coverage_plan or "[not provided]"}

Use this skill only when:
{should_apply_when}

Do not use this skill when:
{should_not_apply_when}

Invariants to preserve:
{invariants_to_preserve}

Regression risks / anti-regression guards:
{regression_risks or "[not provided]"}

Read the existing skill at .claude/skills/{target_skill}/SKILL.md
and modify it to add these capabilities. Preserve all existing content that is still relevant."""
            else:
                _log("", f"  -> Generating new skill...")
                skill_query = build_skill_query_from_skill_proposer(proposer_trace)

            skills_before = set(self._get_active_skills())
            # Capture pre-run content hash for edit-mode diagnostics
            _pre_skill_hash: str | None = None
            _changed = False
            if action_type == "edit" and target_skill:
                import hashlib
                _sp = self._project_root / ".claude" / "skills" / target_skill / "SKILL.md"
                if _sp.exists():
                    _pre_skill_hash = hashlib.md5(_sp.read_bytes()).hexdigest()
            skill_trace = await self.agents.skill_generator.run(skill_query)
            self._add_iteration_cost(skill_trace.total_cost_usd, "evolution")
            if skill_trace.is_error or skill_trace.parse_error:
                _log("", f"  [WARN] Skill generator parse error: {skill_trace.parse_error}")
            materialized_skill = None
            if skill_trace.output:
                materialized_skill = self._materialize_generated_skill(
                    skill_trace.output,
                    action_type=action_type,
                    target_skill=target_skill,
                    fallback_description=proposed,
                )
                if materialized_skill:
                    _log("", f"  -> Materialized SKILL.md: {materialized_skill}")
            skills_after = set(self._get_active_skills())
            new_skills = skills_after - skills_before
            created_skill = next(iter(new_skills)) if new_skills else materialized_skill

            if action_type == "edit" and target_skill:
                _sp = self._project_root / ".claude" / "skills" / target_skill / "SKILL.md"
                if _sp.exists():
                    import hashlib
                    _post_hash = hashlib.md5(_sp.read_bytes()).hexdigest()
                    _changed = _post_hash != _pre_skill_hash
                    _log("", f"  -> Skill file {'CHANGED' if _changed else 'UNCHANGED'}: {_sp}")
                else:
                    _log("", f"  -> Skill file NOT FOUND: {_sp}")

            if is_opencode_sdk() or is_openhands_sdk() or is_goose_sdk() or is_codex_sdk():
                from src.harness.opencode.skill_utils import (
                    DEFAULT_OPENHANDS_SKILL_TRIGGERS,
                    normalize_project_skill_frontmatter,
                )
                from src.harness.sdk_config import get_sdk
                skill_descriptions: dict[str, str] = {}
                if target_skill:
                    skill_descriptions[target_skill] = proposed
                if created_skill:
                    skill_descriptions[created_skill] = proposed
                compatibility = get_sdk()
                skill_triggers = (
                    {
                        skill_name: DEFAULT_OPENHANDS_SKILL_TRIGGERS
                        for skill_name in skill_descriptions
                    }
                    if compatibility == "openhands"
                    else None
                )
                normalize_project_skill_frontmatter(
                    self._project_root,
                    descriptions=skill_descriptions,
                    fallback_description=proposed,
                    compatibility=compatibility,
                    triggers_by_skill=skill_triggers,
                )

            if skill_trace.output:
                self._emit("skill_written", name=created_skill, action=action_type, target=target_skill)

            if action_type == "edit" and target_skill:
                expected_skill_path = (
                    self._project_root / ".claude" / "skills" / target_skill / "SKILL.md"
                )
                skill_valid = expected_skill_path.exists() and (
                    _pre_skill_hash is None or _changed
                )
                if not skill_valid:
                    _log(
                        "",
                        (
                            "  [WARN] Skill edit did not produce a changed SKILL.md; "
                            f"discarding {child_name}"
                        ),
                    )
                    self.manager.discard(child_name)
                    self.manager.switch_to(parent)
                    return None
            else:
                if not created_skill:
                    _log(
                        "",
                        (
                            "  [WARN] Skill generator did not create any SKILL.md; "
                            f"discarding {child_name}"
                        ),
                    )
                    self.manager.discard(child_name)
                    self.manager.switch_to(parent)
                    return None

        else:  # prompt_only
            proposer_trace = await self.agents.prompt_proposer.run(proposer_query)
            self._add_iteration_cost(proposer_trace.total_cost_usd, "evolution")

            if proposer_trace.output is None:
                _log("", f"  [WARN] Prompt proposer failed: {proposer_trace.parse_error}")
                return None

            proposed = proposer_trace.output.proposed_prompt_change
            justification = proposer_trace.output.justification
            root_cause_analysis = ""
            coverage_plan = ""
            regression_risks = ""
            proposer_confidence = self._clamp01(
                getattr(proposer_trace.output, "confidence", None),
                default=self.config.puct_default_prior,
            )
            _log("", f"  -> Proposal: prompt (prior={proposer_confidence:.3f}) - {proposed[:50]}...")

            # Create child program branch
            child_name = f"iter-prompt-{actual_iteration}"
            self.manager.switch_to(parent)
            parent_config = self.manager.get_current()
            original_prompt = parent_config.system_prompt
            child_config = parent_config.mutate(child_name)
            self.manager.create_program(child_name, child_config, parent=parent)

            # Generate optimized prompt
            _log("", f"  -> Generating optimized prompt...")
            prompt_query = build_prompt_query_from_prompt_proposer(
                proposer_trace, original_prompt
            )
            prompt_trace = await self.agents.prompt_generator.run(prompt_query)
            self._add_iteration_cost(prompt_trace.total_cost_usd, "evolution")
            if prompt_trace.output:
                update_prompt_file(
                    self._prompt_path, prompt_trace.output.optimized_prompt
                )

        # Commit changes
        self.manager.commit(f"{child_name}: {proposed[:50]}")

        # Resolve which skill was created/edited (used by the refine loop).
        if evolution_mode == "skill_only":
            resolved_skill_name = (
                target_skill
                if action_type == "edit" and target_skill
                else (created_skill or "")
            )
        else:
            resolved_skill_name = ""

        # Return mutation info (feedback will be written by caller with outcome)
        return MutationResult(
            child_name=child_name,
            proposal=proposed,
            justification=justification,
            proposer_confidence=proposer_confidence,
            root_cause_analysis=root_cause_analysis,
            coverage_plan=coverage_plan,
            regression_risks=regression_risks,
            skill_name=resolved_skill_name or "",
        )

    async def _mutate_with_fallback(
        self,
        parent: str,
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str]],
        iteration: int | str,
        diversity_hint: str = "",
        questions: list[str] | None = None,
    ) -> tuple[str, str, str, float] | None:
        """Try progressive truncation levels, then single-failure fallback.

        Args:
            parent: Name of the parent program.
            failures: List of (trace, agent_answer, ground_truth, category) tuples.
            iteration: Current iteration number.
            diversity_hint: Optional instruction used to diversify sibling children.
            questions: Optional parallel list of question texts for the proposer.

        Returns:
            MutationResult if created, None otherwise.
            if created, None otherwise.
        """
        max_level = self.config.proposer_max_truncation_level

        for truncation_level in range(max_level + 1):
            if truncation_level > 0:
                _log("", f"  -> Retrying with truncation level {truncation_level}...")

            result = await self._mutate(
                parent,
                failures,
                iteration,
                truncation_level,
                diversity_hint=diversity_hint,
                questions=questions,
            )
            if result is not None:
                return result

        # Final fallback: single failure focus (if enabled and multiple failures).
        # questions omitted here — aggressive truncation already strips context.
        if self.config.proposer_single_failure_fallback and len(failures) > 1:
            _log("", f"  -> All truncation levels failed, trying single-failure fallback...")
            single_failure = self._pick_shortest_failure(failures)
            result = await self._mutate(
                parent,
                [single_failure],
                iteration,
                truncation_level=max_level,
                diversity_hint=diversity_hint,
            )
            if result is not None:
                return result

        _log("", f"  [WARN] All proposer fallback attempts failed")
        return None

    def _pick_shortest_failure(
        self,
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str]],
    ) -> tuple[AgentTrace[AgentResponse], str, str, str]:
        """Pick the failure with the shortest trace for fallback.

        Args:
            failures: List of (trace, agent_answer, ground_truth, category) tuples.

        Returns:
            The failure tuple with the shortest trace summary.
        """
        # Estimate trace length by summarizing with default params
        shortest = failures[0]
        shortest_len = len(shortest[0].summarize())

        for failure in failures[1:]:
            length = len(failure[0].summarize())
            if length < shortest_len:
                shortest = failure
                shortest_len = length

        return shortest

    def _select_parent(self, iteration: int = 0) -> str:
        """Select a parent program from the frontier using the configured strategy.

        Args:
            iteration: Current iteration number (used by round_robin strategy).

        Returns:
            Program name to use as parent, or 'base' if frontier is empty.
        """
        selected = self.manager.select_from_frontier(
            self.config.selection_strategy, iteration
        )
        return selected if selected else "base"

    def _get_active_skills(self) -> list[str]:
        """Get list of currently active skills.

        Returns:
            List of skill names that have SKILL.md files.
        """
        skills_dir = self._project_root / ".claude" / "skills"
        active_skills = []
        if skills_dir.exists():
            for skill_dir in skills_dir.iterdir():
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                    active_skills.append(skill_dir.name)
        return sorted(active_skills)

    def _materialize_generated_skill(
        self,
        output: ToolGeneratorResponse,
        *,
        action_type: str,
        target_skill: str | None,
        fallback_description: str,
    ) -> str | None:
        """Write a generator-returned SKILL.md payload into the active program.

        Codex and other sandboxed executors may be able to read the workspace
        but not write it. In that case the generator returns the intended file
        content and the main EvoSkill process performs the write.
        """
        skill_markdown = str(getattr(output, "skill_markdown", "") or "").strip()
        generated_skill = str(getattr(output, "generated_skill", "") or "").strip()
        if not skill_markdown and generated_skill.lstrip().startswith("---"):
            skill_markdown = generated_skill

        skill_name = target_skill if action_type == "edit" and target_skill else None
        if not skill_name:
            skill_name = self._skill_name_from_path(str(getattr(output, "skill_path", "") or ""))
        if not skill_name:
            skill_name = self._skill_name_from_path(generated_skill)
        if not skill_name:
            skill_name = self._skill_name_from_markdown(skill_markdown)
        if not skill_name and self._is_valid_skill_name(generated_skill):
            skill_name = generated_skill
        if not skill_name or not self._is_valid_skill_name(skill_name):
            _log("", f"  [WARN] Generator returned invalid skill name/path: {skill_name!r}")
            return None
        if not skill_markdown:
            if generated_skill == skill_name:
                skill_markdown = (
                    "---\n"
                    f"name: {skill_name}\n"
                    f"description: {fallback_description}\n"
                    "---\n\n"
                    f"# {skill_name}\n\n"
                    f"{str(getattr(output, 'reasoning', '') or fallback_description).strip()}\n"
                )
            else:
                return None

        skill_path = self._project_root / ".claude" / "skills" / skill_name / "SKILL.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        if not skill_markdown.endswith("\n"):
            skill_markdown += "\n"
        skill_path.write_text(skill_markdown)

        from src.harness.opencode.skill_utils import ensure_skill_frontmatter

        ensure_skill_frontmatter(
            skill_path,
            description=fallback_description,
            compatibility=None,
        )
        if getattr(self.config, "validate_worked_examples", False) and (
            self._has_misleading_worked_example(skill_markdown)
        ):
            _log(
                "",
                (
                    f"  [WARN] {skill_name}: worked example may present a scaffold count "
                    "as the final answer; refine loop will attempt a revision"
                ),
            )
        return skill_name

    @staticmethod
    def _is_valid_skill_name(skill_name: str) -> bool:
        import re

        return bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", skill_name))

    @staticmethod
    def _skill_name_from_path(text: str) -> str | None:
        import re

        normalized = text.replace("\\", "/")
        match = re.search(
            r"(?:^|[`'\"( ])(?:\./)?\.claude/skills/([a-z0-9]+(?:-[a-z0-9]+)*)/SKILL\.md",
            normalized,
        )
        return match.group(1) if match else None

    @staticmethod
    def _skill_name_from_markdown(markdown: str) -> str | None:
        import re

        match = re.search(r"\A---\n(.*?)\n---", markdown, flags=re.DOTALL)
        if not match:
            return None
        name_match = re.search(r"(?m)^name:\s*['\"]?([a-z0-9]+(?:-[a-z0-9]+)*)['\"]?\s*$", match.group(1))
        return name_match.group(1) if name_match else None

    def _get_highest_iteration(self) -> int:
        """Find the highest iteration number across all iter-* branches.

        Returns:
            The highest iteration number found, or 0 if none exist.
        """
        programs = self.manager.list_programs()
        max_iter = 0
        for p in programs:
            # Match iter-skill-N or iter-prompt-N or iter-N
            if p.startswith("iter-"):
                parts = p.split("-")
                try:
                    num = int(parts[-1])
                    max_iter = max(max_iter, num)
                except ValueError:
                    pass
        return max_iter

    # ------------------------------------------------------------------ #
    # LLM Judge mode helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _trajectory_failure_type(entry: tuple[Any, ...]) -> str:
        return str(entry[5] or "").strip() if len(entry) > 5 else ""

    @staticmethod
    def _trajectory_failure_feedback(entry: tuple[Any, ...]) -> str:
        return str(entry[6] or "").strip() if len(entry) > 6 else ""

    @staticmethod
    def load_trajectories_from_dir(
        trajectories_dir: str | Path,
    ) -> list[tuple[Any, ...]]:
        """Load pre-collected trajectories from a directory produced by collect_trajectories.py.

        Args:
            trajectories_dir: Directory containing ``trajectories.jsonl``.

        Returns:
            List of (AgentTrace, question, agent_answer, ground_truth, category, failure_type[, failure_feedback]) tuples
            ready to pass as ``preloaded_trajectories`` to SelfImprovingLoop.__init__.
        """
        from src.schemas.trajectory import StoredTrajectory
        jsonl_path = Path(trajectories_dir) / "trajectories.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Trajectories file not found: {jsonl_path}")
        stored = StoredTrajectory.load_jsonl(jsonl_path)
        return [t.to_extended_tuple() for t in stored]

    def _get_all_data(self) -> list[tuple[str, str, str]]:
        """Combine train pools and val data into one flat, deduplicated list."""
        seen: set[tuple[str, str, str]] = set()
        result: list[tuple[str, str, str]] = []
        for cat, pool in self.train_pools.items():
            for q, a in pool:
                key = (q, a, cat)
                if key not in seen:
                    seen.add(key)
                    result.append(key)
        for q, a, cat in self.val_data:
            key = (q, a, cat)
            if key not in seen:
                seen.add(key)
                result.append(key)
        return result

    async def _collect_all_trajectories(
        self,
        all_data: list[tuple[str, str, str]],
    ) -> list[tuple[AgentTrace, str, str, str, str, str, str]]:
        """Run all samples concurrently and return extended trajectory tuples."""
        traces = await asyncio.gather(*[
            self.agents.base.run(q) for q, _, _ in all_data
        ])
        self._add_iteration_cost(
            sum(t.total_cost_usd for t in traces),
            "trajectory",
        )
        result = []
        for trace, (question, ground_truth, category) in zip(traces, all_data):
            agent_answer = (
                trace.output.final_answer
                if trace.output and trace.output.final_answer
                else "[PARSE FAILED]"
            )
            result.append((trace, question, agent_answer, ground_truth, category, "", ""))
        return result

    def _get_all_skills_content(self) -> str:
        """Read and concatenate all active SKILL.md files."""
        skills_dir = self._project_root / ".claude" / "skills"
        if not skills_dir.exists():
            return "No skills available."
        parts = []
        for skill_dir in sorted(skills_dir.iterdir()):
            skill_file = skill_dir / "SKILL.md"
            if skill_dir.is_dir() and skill_file.exists():
                parts.append(f"### Skill: {skill_dir.name}\n{skill_file.read_text()}")
        return "\n\n".join(parts) if parts else "No skills available."

    def _get_program_skills_content(self, program_name: str, restore_to: str | None = None) -> str:
        """Read active skills for a stored program and optionally restore another program."""
        try:
            self.manager.switch_to(program_name)
            return self._get_all_skills_content()
        finally:
            if restore_to is not None:
                self.manager.switch_to(restore_to)

    def _select_bt_anchor_nodes(self, parent: str, child_name: str) -> list[str]:
        """Choose Bradley-Terry anchors for a new child.

        Kept deliberately small (parent + current frontier best) to bound judge
        call volume: each extra anchor re-judges the entire scoring set. ``base``
        is NOT added per-child — it is already the fixed global anchor of the BT
        fit, so cross-generation comparability is preserved without paying for a
        base re-judge on every candidate. When the frontier best is the parent
        itself (or the frontier is empty), only the parent anchor is used.
        """
        candidates: list[str] = [parent]
        best = self.manager.get_best_from_frontier()
        if best:
            candidates.append(best)

        existing = set(self.manager.list_programs())
        anchors: list[str] = []
        for name in candidates:
            if name == child_name or name not in existing or name in anchors:
                continue
            anchors.append(name)
        return anchors

    @staticmethod
    def _duel_ci_halfwidth(scores: list[float], delta: float) -> float:
        """Normal CI half-width for a paired match-score mean.

        Variance-aware: when the judge's per-case scores cluster tightly (the
        common case here) the interval is narrow and a duel resolves in few
        cases. A std floor guards against a coincidental near-tie producing an
        overconfident decision at very small n. Distribution-free bounds
        (Hoeffding / empirical-Bernstein) are far too loose at n≈4 — their
        additive term alone exceeds the [0, 1] range — so a normal approximation
        on the mean is used instead. Returns a wide 1.0 with too little data.
        """
        from statistics import NormalDist

        n = len(scores)
        if n < 2:
            return 1.0
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / (n - 1)
        std = max(math.sqrt(var), 0.05)  # std floor against tiny-n overconfidence
        z = NormalDist().inv_cdf(1.0 - max(1e-6, min(0.5, delta)) / 2.0)
        return z * std / math.sqrt(n)

    def _duel_decision(self, result: "JudgeResult | None") -> str:
        """Classify a champion duel as 'win', 'loss', or 'close'.

        Uses the candidate-vs-anchor paired match scores: 0.5 is a draw. The
        child clearly wins when even the CI lower bound exceeds 0.5, clearly
        loses when the CI upper bound is below 0.5, else the duel is close.
        """
        if result is None:
            return "close"
        scores = [m.match_score for m in result.matches if getattr(m, "valid", True)]
        n = len(scores)
        if n < max(1, self.config.judge_duel_min_cases):
            return "close"
        mean = sum(scores) / n
        h = self._duel_ci_halfwidth(scores, self.config.judge_duel_delta)
        if mean + h < 0.5:
            return "loss"
        if mean - h > 0.5:
            return "win"
        return "close"

    def _random_frontier_anchors(self, exclude: set[str], n: int) -> list[str]:
        """Pick up to ``n`` random current frontier members not in ``exclude``."""
        if n <= 0:
            return []
        existing = set(self.manager.list_programs())
        members = [
            name
            for name, _score in self.manager.get_frontier_with_scores()
            if name not in exclude and name in existing
        ]
        if not members:
            return []
        import random as _random

        rng = _random.Random(f"stage2::{','.join(sorted(exclude))}::{len(members)}")
        rng.shuffle(members)
        return members[:n]

    @staticmethod
    def _build_child_diversity_hint(child_idx: int, sibling_proposals: list[str]) -> str:
        """Build proposer guidance so sibling children explore different fixes."""
        strategies = [
            (
                "For this child, prefer the smallest targeted edit to an existing "
                "relevant skill. Focus on the most direct reusable verification gap."
            ),
            (
                "For this child, propose an alternative root-cause hypothesis. "
                "Do not repeat the first child's target skill, checks, or workflow unless unavoidable."
            ),
            (
                "For this child, prefer a broader workflow-level capability or a new "
                "specialized skill if editing an existing skill would only patch symptoms."
            ),
            (
                "For this child, deliberately choose a different intervention type "
                "from earlier siblings, with different trigger conditions and verification steps."
            ),
        ]
        base_hint = strategies[min(child_idx, len(strategies) - 1)]
        if not sibling_proposals:
            return base_hint

        prior = "\n".join(f"- {proposal[:240]}" for proposal in sibling_proposals)
        return (
            f"{base_hint}\n\nAlready proposed sibling changes in this expansion:\n{prior}\n"
            "Your proposal must be materially different in target, root cause, or verification method."
        )

    @staticmethod
    def _merge_scoring_failures(
        parent_failures: list,
        new_failures: list,
        max_size: int = 40,
    ) -> list:
        """Merge parent's scoring failures with new batch, deduplicating by question.

        New failures are added first (all of them, up to max_size) so the child's
        improvement signal is never diluted.  Parent failures fill the remaining
        slots up to max_size for regression coverage, capped at max_size // 2 so a
        large parent history cannot crowd out the new batch.
        """
        seen: set[str] = set()
        merged: list = []
        # New batch first — always fully represented
        for f in new_failures:
            if len(merged) >= max_size:
                break
            q = f[1]  # question is index 1 in the 6-tuple
            if q not in seen:
                seen.add(q)
                merged.append(f)
        # Parent failures fill the rest for regression safety, capped at half max_size
        parent_cap = min(max_size // 2, max_size - len(merged))
        n_parent = 0
        for f in parent_failures:
            if n_parent >= parent_cap:
                break
            q = f[1]
            if q not in seen:
                seen.add(q)
                merged.append(f)
                n_parent += 1
        return merged

    def _sample_regression_successes(
        self,
        successes: list[tuple[Any, ...]],
        categories: list[str],
        count: int,
    ) -> list[tuple[Any, ...]]:
        """Sample already-passed trajectories for judge regression checks."""
        if not successes or count <= 0:
            return []

        batch: list[tuple[Any, ...]] = []
        seen: set[str] = set()
        categories = categories or sorted({entry[4] for entry in successes})
        if not categories:
            categories = ["default"]

        cat_idx = 0
        attempts = 0
        max_attempts = max(len(categories) * 2, count * 4)
        while len(batch) < count and attempts < max_attempts:
            cat = categories[cat_idx % len(categories)]
            cat_successes = [entry for entry in successes if entry[4] == cat]
            if cat_successes:
                offset = self._per_cat_offset.get(f"regression:{cat}", 0)
                entry = cat_successes[offset % len(cat_successes)]
                self._per_cat_offset[f"regression:{cat}"] = offset + 1
                question = entry[1]
                if question not in seen:
                    seen.add(question)
                    batch.append(entry)
            cat_idx += 1
            attempts += 1

        if len(batch) < count:
            for entry in successes:
                if len(batch) >= count:
                    break
                question = entry[1]
                if question not in seen:
                    seen.add(question)
                    batch.append(entry)
        return batch

    @staticmethod
    def _filter_bt_matches_for_active_players(
        matches: list["BradleyTerryMatch"],
        active_players: set[str],
    ) -> list["BradleyTerryMatch"]:
        """Return only matches where both player and opponent are active.

        Active = currently in frontier, just generated, or an anchor.
        Keeps the global bt_matches list intact for history; only the
        filtered view is passed to the BT fitter so old discarded nodes
        cannot shift ratings of current candidates.
        """
        return [m for m in matches if m.player in active_players and m.opponent in active_players]

    @staticmethod
    def _summarize_skill_content(content: str, max_chars: int = 3500) -> str:
        """Build a compact judge-facing summary of available skills."""
        if not content.strip():
            return "No skills available."

        lines: list[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("### Skill:") or line.startswith("#") or line.startswith("- "):
                lines.append(line)
            if sum(len(item) + 1 for item in lines) >= max_chars:
                break

        summary = "\n".join(lines) if lines else content.strip()
        if len(summary) > max_chars:
            return summary[:max_chars].rstrip() + "\n[truncated]"
        return summary

    @staticmethod
    def _diff_skill_content(parent_content: str, candidate_content: str, max_chars: int = 5000) -> str:
        """Return a bounded unified diff for judge context."""
        diff = "\n".join(
            difflib.unified_diff(
                parent_content.splitlines(),
                candidate_content.splitlines(),
                fromfile="parent_skills",
                tofile="candidate_skills",
                lineterm="",
            )
        )
        if not diff.strip():
            return "No textual skill diff detected."
        if len(diff) > max_chars:
            return diff[:max_chars].rstrip() + "\n[diff truncated]"
        return diff

    @staticmethod
    def _compact_judge_text(
        text: str,
        *,
        max_chars: int,
        max_line_chars: int = 1200,
    ) -> str:
        """Bound judge prompt fields and break long lines for Codex prompt chunking."""
        value = str(text or "").strip()
        if not value:
            return ""

        if len(value) > max_chars:
            head_chars = max_chars * 2 // 3
            tail_chars = max_chars - head_chars
            omitted = len(value) - head_chars - tail_chars
            value = (
                f"{value[:head_chars].rstrip()}\n"
                f"[... {omitted:,} chars omitted for judge prompt budget ...]\n"
                f"{value[-tail_chars:].lstrip()}"
            )

        wrapped_lines: list[str] = []
        for line in value.splitlines():
            if len(line) <= max_line_chars:
                wrapped_lines.append(line)
                continue
            start = 0
            while start < len(line):
                remaining = len(line) - start
                suffix = " [line continued]" if remaining > max_line_chars else ""
                chunk_chars = max_line_chars - len(suffix) if suffix else max_line_chars
                wrapped_lines.append(line[start : start + chunk_chars] + suffix)
                start += chunk_chars
        return "\n".join(wrapped_lines)

    def _summarize_trace_for_judge(self, trace: AgentTrace) -> str:
        """Create a compact trace summary for judge calls.

        AgentTrace.summarize only truncates parse-error traces. Judge prompts
        must always be bounded because successful Codex trajectories can contain
        very long result blocks or JSON lines that break Codex SDK chunking.
        """
        return self._compact_judge_text(
            trace.summarize(head_chars=3000, tail_chars=1500),
            max_chars=7000,
            max_line_chars=1200,
        )

    @staticmethod
    def _normalize_model_name(model: str | None) -> str:
        m = (model or "").strip().lower()
        for prefix in ("openai/", "anthropic/", "codex/"):
            if m.startswith(prefix):
                m = m[len(prefix):]
        return m

    def _generator_model_id(self) -> str:
        """Best-effort model id of the skill generator (the skill author)."""
        try:
            opts = self.agents.skill_generator._get_options()
        except Exception:
            return ""
        if isinstance(opts, dict):
            return str(opts.get("model_id") or opts.get("model") or "")
        return str(
            getattr(opts, "model_id", None)
            or getattr(opts, "model", None)
            or getattr(opts, "model_name", None)
            or ""
        )

    def _distinct_judge_model(self, provider: str) -> str:
        """A sensible judge model for ``provider`` that differs from the generator.

        Stays within the same provider (so credentials/SDK still work) but picks a
        smaller/other default, which both de-correlates self-enhancement bias and
        is cheaper (cf. panel-of-small-judges findings)."""
        return {
            "anthropic": "claude-haiku-4-5-20251001",
            "openai": "gpt-4o-mini",
            "codex": "gpt-5.1-codex-mini",
        }.get(provider, "gpt-4o-mini")

    def _detect_judge_provider_and_model(self) -> tuple[str, str]:
        """Return (provider, model) for judge calls.

        Enforces that the judge does not run on the same model that authored the
        skill (the generator) when ``judge_distinct_from_generator`` is set, to
        avoid self-enhancement bias (a model favouring its own outputs)."""
        provider, model = self._detect_judge_provider_and_model_raw()
        if self.config.judge_distinct_from_generator:
            gen = self._normalize_model_name(self._generator_model_id())
            if gen and self._normalize_model_name(model) == gen:
                alt = self._distinct_judge_model(provider)
                if self._normalize_model_name(alt) != gen:
                    _log(
                        "JUDGE",
                        (
                            f"Judge model ({model}) == generator model; switching judge to "
                            f"{alt} to avoid self-enhancement bias "
                            f"(set judge_distinct_from_generator=False to override)"
                        ),
                    )
                    model = alt
        return provider, model

    def _detect_judge_provider_and_model_raw(self) -> tuple[str, str]:
        """Return (provider, model) for judge calls based on the active SDK."""
        from src.harness.sdk_config import get_sdk
        sdk = get_sdk()
        if sdk == "claude":
            return "anthropic", self.config.judge_model or "claude-haiku-4-5-20251001"
        if sdk == "codex":
            judge_model = self.config.judge_model
            if judge_model and judge_model.startswith("openai/"):
                judge_model = judge_model.split("/", 1)[1]
            return "codex", judge_model or "gpt-5.1-codex-mini"
        if self.config.judge_model:
            judge_model = self.config.judge_model
            if judge_model.startswith("openai/"):
                return "openai", judge_model.split("/", 1)[1]
            if judge_model.startswith(("gpt-", "o", "chatgpt-")):
                return "openai", judge_model
            if judge_model.startswith("anthropic/"):
                return "anthropic", judge_model.split("/", 1)[1]
            if judge_model.startswith("claude"):
                return "anthropic", judge_model
        # For opencode / codex / openhands / goose, read provider from base agent options
        opts = self.agents.base._get_options()
        if isinstance(opts, dict):
            model_id = str(opts.get("model_id") or opts.get("model") or "")
        else:
            model_id = str(
                getattr(opts, "model_id", None)
                or getattr(opts, "model", None)
                or getattr(opts, "model_name", None)
                or ""
            )
        provider = model_id.split("/")[0] if "/" in model_id else "anthropic"
        model_name = model_id.split("/", 1)[1] if "/" in model_id else model_id
        if provider == "openai" or model_name.startswith(("gpt-", "o", "chatgpt-")):
            return "openai", self.config.judge_model or "gpt-4o-mini"
        else:
            return "anthropic", self.config.judge_model or "claude-haiku-4-5-20251001"

    async def _call_judge_api(self, prompt: str, provider: str, model: str) -> str:
        """Make a direct (non-agent) completion call to the judge model."""
        if provider == "codex":
            from src.harness.codex import executor as codex_executor

            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "root_cause": {"type": "string"},
                    "proposer_root_cause_correct": {"type": "number"},
                    "skill_addresses_root_cause": {"type": "number"},
                    "failure_mechanism_encoding": {"type": "number"},
                    "executable_specificity": {"type": "number"},
                    "high_risk_blacklist": {"type": "number"},
                    "generalization_transfer": {"type": "number"},
                    "set_a_success_prob": {"type": "number"},
                    "set_b_success_prob": {"type": "number"},
                    "b_over_a_score": {"type": "number"},
                    "confidence": {"type": "number"},
                    "remaining_blockers": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "root_cause",
                    "proposer_root_cause_correct",
                    "skill_addresses_root_cause",
                    "failure_mechanism_encoding",
                    "executable_specificity",
                    "high_risk_blacklist",
                    "generalization_transfer",
                    "set_a_success_prob",
                    "set_b_success_prob",
                    "b_over_a_score",
                    "confidence",
                    "remaining_blockers",
                    "reasoning",
                ],
            }
            messages = await codex_executor.execute_query(
                {
                    "system": "",
                    "model": model,
                    "working_directory": str(self._project_root),
                    "tools": [],
                    "output_schema": schema,
                },
                prompt,
            )
            turn = messages[0]
            self._record_judge_api_usage("codex", model, getattr(turn, "usage", None))
            text = str(getattr(turn, "final_response", "") or "").strip()
            if not text:
                raise RuntimeError("Codex judge returned empty final_response")
            return text

        from src.harness.provider_auth import ensure_provider_api_key
        api_key = ensure_provider_api_key(provider)
        if provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=api_key)
            resp = await client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            self._record_judge_api_usage(provider, model, getattr(resp, "usage", None))
            return resp.content[0].text
        if provider == "openai":
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if model.startswith(("gpt-5", "o1", "o3", "o4")):
                kwargs["max_completion_tokens"] = 512
            else:
                kwargs["max_tokens"] = 512
            resp = await client.chat.completions.create(**kwargs)
            self._record_judge_api_usage(provider, model, getattr(resp, "usage", None))
            return resp.choices[0].message.content or ""
        raise ValueError(f"Judge does not support provider: {provider!r}")

    @staticmethod
    def _usage_value(usage: Any, names: tuple[str, ...]) -> int:
        if usage is None:
            return 0
        for name in names:
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0
        return 0

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        return model.split("/", 1)[-1].strip().lower()

    @staticmethod
    def _default_judge_price_per_1m(provider: str, model: str) -> tuple[float | None, float | None]:
        """Return default direct judge pricing as USD per 1M input/output tokens."""
        if provider != "openai":
            return None, None
        normalized = SelfImprovingLoop._normalize_model_name(model)
        if normalized in OPENAI_JUDGE_PRICE_PER_1M:
            return OPENAI_JUDGE_PRICE_PER_1M[normalized]

        # Treat versioned nano/mini suffixes as their closest published family
        # price unless an explicit override is provided.
        if normalized.startswith("gpt-5.") and normalized.endswith("-nano"):
            return OPENAI_JUDGE_PRICE_PER_1M["gpt-5-nano"]
        if normalized.startswith("gpt-5.") and normalized.endswith("-mini"):
            return OPENAI_JUDGE_PRICE_PER_1M["gpt-5-mini"]
        if normalized.startswith("gpt-5.") and "-pro" not in normalized:
            return OPENAI_JUDGE_PRICE_PER_1M["gpt-5.4"]
        return None, None

    def _record_judge_api_usage(self, provider: str, model: str, usage: Any) -> None:
        """Record direct judge token usage and estimated cost."""
        input_tokens = self._usage_value(
            usage,
            ("prompt_tokens", "input_tokens", "cache_read_input_tokens"),
        )
        output_tokens = self._usage_value(
            usage,
            ("completion_tokens", "output_tokens"),
        )
        total_tokens = self._usage_value(usage, ("total_tokens",))
        if total_tokens <= 0:
            total_tokens = input_tokens + output_tokens

        self._judge_prompt_tokens = getattr(self, "_judge_prompt_tokens", 0) + input_tokens
        self._judge_completion_tokens = getattr(self, "_judge_completion_tokens", 0) + output_tokens
        self._judge_total_tokens = getattr(self, "_judge_total_tokens", 0) + total_tokens

        config = getattr(self, "config", None)
        input_rate = getattr(config, "judge_input_cost_per_1m", None)
        output_rate = getattr(config, "judge_output_cost_per_1m", None)
        if input_rate is None or output_rate is None:
            default_input_rate, default_output_rate = self._default_judge_price_per_1m(
                provider,
                model,
            )
            input_rate = default_input_rate if input_rate is None else input_rate
            output_rate = default_output_rate if output_rate is None else output_rate
        if input_rate is None or output_rate is None:
            return

        cost = (input_tokens * float(input_rate) + output_tokens * float(output_rate)) / 1_000_000.0
        self._add_iteration_cost(cost, "judge")
        if cost > 0 and self.config.judge_log_details:
            _log(
                "COST",
                (
                    f"Judge API {provider}/{model}: "
                    f"in={input_tokens} out={output_tokens} cost=${cost:.6f}"
                ),
            )

    @staticmethod
    def _clamp01(value: Any, default: float = 0.5) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _elo_expected(rating: float, opponent_rating: float, scale: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / scale))

    @staticmethod
    def _fit_bradley_terry_ratings(
        matches: list[BradleyTerryMatch],
        players: list[str],
        *,
        anchor: str = "base",
        initial_rating: float = 1500.0,
        scale: float = 400.0,
        iterations: int = 300,
        learning_rate: float = 0.08,
        l2: float = 0.01,
    ) -> dict[str, float]:
        """Fit global Bradley-Terry ratings from soft pairwise outcomes.

        The fitted parameter is a logit strength anchored at ``anchor``. It is
        converted back to Elo-like ratings with ``scale / ln(10)`` so existing
        Elo expected-win math can be reused.
        """
        names = sorted(set(players) | {m.player for m in matches} | {m.opponent for m in matches})
        if anchor not in names:
            names.append(anchor)
        theta = {name: 0.0 for name in names}

        if not matches:
            return {name: initial_rating for name in names}

        for _ in range(iterations):
            grad = {name: 0.0 for name in names}
            for match in matches:
                if match.player == match.opponent:
                    continue
                a = match.player
                b = match.opponent
                y = SelfImprovingLoop._clamp01(match.score, default=0.5)
                diff = theta[a] - theta[b]
                # Stable sigmoid.
                if diff >= 0:
                    exp_neg = math.exp(-diff)
                    p = 1.0 / (1.0 + exp_neg)
                else:
                    exp_pos = math.exp(diff)
                    p = exp_pos / (1.0 + exp_pos)
                residual = y - p
                grad[a] += residual
                grad[b] -= residual

            for name in names:
                if name == anchor:
                    continue
                grad[name] -= l2 * theta[name]
                theta[name] += learning_rate * grad[name] / max(1, len(matches))

            # Keep the rating frame stable and globally comparable.
            offset = theta.get(anchor, 0.0)
            for name in names:
                theta[name] -= offset
            theta[anchor] = 0.0

        factor = scale / math.log(10.0)
        return {name: initial_rating + theta[name] * factor for name in names}

    @staticmethod
    def _rating_to_score(
        rating: float,
        anchor_rating: float,
        base_score: float,
        scale: float,
    ) -> float:
        """Map a global rating advantage over base to the score space.

        Equal rating maps to the observed base score. Ratings above base consume
        the remaining headroom toward 1.0; ratings below base consume the
        downside room toward 0.0. This preserves BT ordering instead of
        collapsing all weaker-than-base nodes onto the base score.
        """
        expected_win_rate = SelfImprovingLoop._elo_expected(rating, anchor_rating, scale)
        centered = max(-1.0, min(1.0, 2.0 * (expected_win_rate - 0.5)))
        if centered >= 0.0:
            return base_score + centered * (1.0 - base_score)
        return base_score + centered * base_score

    @staticmethod
    def _judge_binary_to_match_score(would_succeed: bool, confidence: float) -> float:
        """Convert judge binary outcome + confidence to an Elo match score.

        Confidence controls distance from a draw:
        - true, 1.0  -> 1.0 challenger win
        - false, 1.0 -> 0.0 challenger loss
        - any, 0.0   -> 0.5 draw/uncertain
        """
        if would_succeed:
            return 0.5 + 0.5 * confidence
        return 0.5 - 0.5 * confidence

    @staticmethod
    def _parse_judge_bool(value: Any, *, default: bool = False) -> bool:
        """Parse judge booleans without treating non-empty strings as true."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "1"}:
                return True
            if normalized in {"false", "no", "n", "0"}:
                return False
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        return default

    @staticmethod
    def _judge_probability_to_match_score(
        probability_of_success: Any,
        skill_addresses_root_cause: Any,
    ) -> float:
        """Convert probabilistic judge estimates to an Elo match score."""
        p_success = SelfImprovingLoop._clamp01(probability_of_success, default=0.5)
        p_root = SelfImprovingLoop._clamp01(skill_addresses_root_cause, default=p_success)
        return max(0.0, min(1.0, 0.75 * p_success + 0.25 * p_root))

    @staticmethod
    def _judge_relative_to_match_score(
        data: dict[str, Any],
    ) -> tuple[float, float, float, float]:
        """Convert A/B judge probabilities into a pairwise match score.

        Returns (match_score, parent_prob, candidate_prob, relative_advantage).
        ``match_score`` is centered on 0.5 so Bradley-Terry receives a true
        pairwise comparison rather than an absolute candidate success estimate.
        """
        parent_prob = SelfImprovingLoop._clamp01(
            data.get("parent_success_prob"),
            default=0.5,
        )
        candidate_prob = SelfImprovingLoop._clamp01(
            data.get("candidate_success_prob"),
            default=SelfImprovingLoop._clamp01(
                data.get("probability_of_success"),
                default=0.5,
            ),
        )
        if "relative_advantage" in data:
            try:
                relative_advantage = float(data.get("relative_advantage"))
            except (TypeError, ValueError):
                relative_advantage = candidate_prob - parent_prob
        else:
            relative_advantage = candidate_prob - parent_prob
        relative_advantage = max(-1.0, min(1.0, relative_advantage))

        if "match_score" in data:
            match_score = SelfImprovingLoop._clamp01(data.get("match_score"), default=0.5)
        else:
            match_score = max(0.0, min(1.0, 0.5 + 0.5 * relative_advantage))
        return match_score, parent_prob, candidate_prob, relative_advantage

    @staticmethod
    def _judge_artifact_quality(data: dict[str, Any]) -> float:
        """Score whether a skill change has the concrete ingredients that predict utility.

        This is intentionally separate from the judge's A/B success estimate. It
        prevents polished but generic skills from receiving high pairwise credit
        unless they encode the failure mechanism, executable remedy, and
        regression blacklist.
        """
        mechanism = SelfImprovingLoop._clamp01(
            data.get("failure_mechanism_encoding"),
            default=SelfImprovingLoop._clamp01(data.get("skill_addresses_root_cause"), default=0.0),
        )
        executable = SelfImprovingLoop._clamp01(data.get("executable_specificity"), default=0.0)
        blacklist = SelfImprovingLoop._clamp01(data.get("high_risk_blacklist"), default=0.0)
        transfer = SelfImprovingLoop._clamp01(data.get("generalization_transfer"), default=0.0)
        return max(
            0.0,
            min(
                1.0,
                0.35 * mechanism
                + 0.30 * executable
                + 0.20 * blacklist
                + 0.15 * transfer,
            ),
        )

    @staticmethod
    def _apply_artifact_quality_gate(match_score: float, artifact_quality: float) -> float:
        """Cap candidate wins when the skill lacks concrete utility predictors."""
        match_score = max(0.0, min(1.0, match_score))
        artifact_quality = max(0.0, min(1.0, artifact_quality))
        if match_score <= 0.5:
            return match_score
        return 0.5 + (match_score - 0.5) * artifact_quality

    @staticmethod
    def _canonicalize_judge_orientation(
        data: dict[str, Any],
        candidate_slot: str,
    ) -> dict[str, Any]:
        """Translate neutral A/B judge output to canonical candidate-vs-parent keys.

        The judge sees two neutral "Skill Set A/B" with the candidate randomly
        placed in one slot; ``candidate_slot`` records which. This recovers the
        canonical ``candidate_success_prob`` / ``parent_success_prob`` /
        ``match_score`` (candidate over parent) by un-swapping, cancelling any
        fixed-slot position bias. Writes the canonical keys back into ``data``.
        """
        sa = SelfImprovingLoop._clamp01(data.get("set_a_success_prob"), default=0.5)
        sb = SelfImprovingLoop._clamp01(data.get("set_b_success_prob"), default=0.5)
        b_over_a = SelfImprovingLoop._clamp01(data.get("b_over_a_score"), default=0.5)
        if candidate_slot == "B":
            cand_prob, parent_prob, match = sb, sa, b_over_a
        else:  # candidate placed in Set A
            cand_prob, parent_prob, match = sa, sb, 1.0 - b_over_a
        data["candidate_success_prob"] = cand_prob
        data["parent_success_prob"] = parent_prob
        data["match_score"] = match
        data["relative_advantage"] = max(-1.0, min(1.0, cand_prob - parent_prob))
        data["would_succeed"] = match >= 0.5
        data["_candidate_slot"] = candidate_slot
        return data

    @staticmethod
    def _bt_player_uncertainty(
        player: str,
        matches: list[BradleyTerryMatch],
        judge_results: dict[str, JudgeResult] | None = None,
    ) -> float:
        """Estimate uncertainty for a BT player in [0, 1].

        Sparse comparisons and low judge confidence both increase uncertainty.
        """
        player_matches = [
            match
            for match in matches
            if match.player == player or match.opponent == player
        ]
        n_matches = len(player_matches)
        count_uncertainty = 1.0 / math.sqrt(max(1, n_matches))

        confidence_uncertainty = 0.0
        if judge_results and player in judge_results:
            confidences = [
                match.confidence
                for match in judge_results[player].matches
                if match.valid
            ]
            if confidences:
                confidence_uncertainty = 1.0 - (
                    sum(confidences) / len(confidences)
                )

        return max(count_uncertainty, confidence_uncertainty)

    def _rating_to_score_with_uncertainty(
        self,
        rating: float,
        anchor_rating: float,
        base_score: float,
        scale: float,
        uncertainty: float,
    ) -> float:
        """Map BT rating to score and subtract a configurable uncertainty penalty."""
        raw_score = self._rating_to_score(rating, anchor_rating, base_score, scale)
        penalty = max(0.0, self.config.judge_bt_uncertainty_penalty) * max(
            0.0,
            min(1.0, uncertainty),
        )
        return max(0.0, min(1.0, raw_score - penalty))

    def _proposal_policy_prior(self, proposer_confidence: float | None = None) -> float:
        """Return a judge-independent policy prior for PUCT expansion.

        Keep this separate from judge value to avoid double-counting judge
        scores in both Q and exploration.
        """
        if proposer_confidence is None:
            proposer_confidence = self.config.puct_default_prior
        return self._clamp01(proposer_confidence, default=self.config.puct_default_prior)

    def _build_program_search_tree(self, base_score: float) -> ProgramSearchNode:
        """Build an in-memory PUCT tree seeded from the current frontier."""
        root = ProgramSearchNode(
            name="base",
            parent=None,
            prior=1.0,  # policy prior for root: maximum, decoupled from value (score)
            score=base_score,
            visit_count=1,
            total_q=base_score,
            depth=0,
        )

        for name, score in self.manager.get_frontier_with_scores():
            if name == "base":
                continue
            child = ProgramSearchNode(
                name=name,
                parent=root,
                prior=self._proposal_policy_prior(),
                score=score,
                visit_count=1,
                total_q=score,
                depth=1,
            )
            root.children.append(child)
            root.visit_count += 1
            root.total_q += score
        return root

    def _select_puct_node(self, root: ProgramSearchNode) -> ProgramSearchNode:
        """Select a program node to expand using PUCT.

        Selection starts at the root and repeatedly follows the child with the
        highest PUCT score until it reaches a leaf or max depth. The selected
        leaf is then expanded if it still has child capacity. This keeps Q
        values and policy priors in the tree policy instead of filling every
        slot on the current node before descent.
        """
        node = root
        while True:
            live_children = [child for child in node.children if not child.discarded]

            if not live_children:
                if (
                    node.depth < self.config.puct_max_depth
                    and len(node.children) < self.config.puct_children_per_node
                ):
                    return node
                break

            if node.depth >= self.config.puct_max_depth:
                break

            node = max(
                live_children,
                key=lambda child: child.puct_score(self.config.puct_c, node.visit_count),
            )

        expandable = self._collect_expandable_puct_nodes(root)
        if not expandable:
            return root
        return max(
            expandable,
            key=lambda candidate: candidate.puct_score(
                self.config.puct_c,
                candidate.parent.visit_count if candidate.parent else candidate.visit_count,
            ),
        )

    def _collect_expandable_puct_nodes(self, root: ProgramSearchNode) -> list[ProgramSearchNode]:
        nodes: list[ProgramSearchNode] = []

        def visit(node: ProgramSearchNode) -> None:
            if (
                not node.discarded
                and node.depth < self.config.puct_max_depth
                and len(node.children) < self.config.puct_children_per_node
            ):
                nodes.append(node)
            for child in node.children:
                visit(child)

        visit(root)
        return nodes

    def _sample_proposal_failures(
        self,
        failures: list[tuple[Any, ...]],
        categories: list[str],
        failure_types: list[str] | None = None,
    ) -> list[tuple[Any, ...]]:
        """Sample a rotating batch for proposal generation.

        When failure_type labels are available (non-empty strings), sample
        within a single failure_type so the Proposer sees a homogeneous batch
        with the same root cause.  Falls back to category-based round-robin
        when no failure_type labels exist.
        """
        if not failures:
            return []

        # --- failure_type-aware path ---
        typed_failures = [f for f in failures if self._trajectory_failure_type(f)]
        if failure_types and typed_failures:
            types = failure_types
            n_types = len(types)
            ft_idx = self._category_offset % n_types
            ft = types[ft_idx]
            ft_failures = [
                f for f in typed_failures if self._trajectory_failure_type(f) == ft
            ]
            # Advance even if empty so we don't keep retrying the same type
            self._category_offset += 1
            if ft_failures:
                samples = min(
                    self.config.samples_per_category * self.config.categories_per_batch,
                    len(ft_failures),
                )
                return self._pick_least_shown(ft_failures, samples)
            # If the chosen type has no failures left, fall through to category path

        # --- category-based path (fallback / no failure_type labels) ---
        n_cats = len(categories)
        if n_cats == 0:
            return self._pick_least_shown(failures, self.config.samples_per_category)

        n_cats_this_iter = min(self.config.categories_per_batch, n_cats)
        batch: list[tuple[Any, ...]] = []
        for j in range(n_cats_this_iter):
            cat_idx = (self._category_offset + j) % n_cats
            cat = categories[cat_idx]
            cat_failures = [f for f in failures if f[4] == cat]
            samples_to_take = min(self.config.samples_per_category, len(cat_failures))
            batch.extend(self._pick_least_shown(cat_failures, samples_to_take))
        self._category_offset += n_cats_this_iter

        if not batch:
            return self._pick_least_shown(failures, self.config.samples_per_category)
        return batch

    def _pick_least_shown(
        self,
        pool: list[tuple[Any, ...]],
        n: int,
    ) -> list[tuple[Any, ...]]:
        """Pick ``n`` least-recently-shown failures from ``pool`` and mark them.

        A stable sort by per-question show-count keeps original order as the
        tiebreak, so the first call returns items [0, 1], the next returns the
        still-unshown [2, 3], and so on — sweeping all distinct failures before
        repeating any. Show-counts are incremented for the picked items.
        """
        if n <= 0 or not pool:
            return []
        ordered = sorted(pool, key=lambda f: self._failure_shown_count.get(str(f[1]), 0))
        picked = ordered[: min(n, len(ordered))]
        for entry in picked:
            key = str(entry[1])
            self._failure_shown_count[key] = self._failure_shown_count.get(key, 0) + 1
        return picked

    def _split_holdout(
        self,
        failures_ext: list[tuple[Any, ...]],
    ) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        """Partition failures (per category) into a proposer-visible pool and a
        disjoint judge-only pool.

        The proposer only ever sees the first pool, while candidates are scored
        on the second. This stops a proposal from being graded on the very cases
        it was written against (memorization / overfitting). Categories with a
        single failure are kept proposer-visible; if no held-out case exists at
        all, both pools fall back to the full set so scoring still has signal.
        """
        ratio = max(0.0, min(0.9, self.config.judge_holdout_ratio))
        if ratio <= 0.0 or len(failures_ext) < 2:
            return failures_ext, failures_ext

        import random as _random

        by_cat: dict[str, list[tuple[Any, ...]]] = {}
        for entry in failures_ext:
            by_cat.setdefault(entry[4], []).append(entry)

        proposer_pool: list[tuple[Any, ...]] = []
        judge_pool: list[tuple[Any, ...]] = []
        for cat in sorted(by_cat):
            items = list(by_cat[cat])
            _random.Random(f"holdout::{cat}").shuffle(items)
            if len(items) < 2:
                proposer_pool.extend(items)
                continue
            n_judge = max(1, round(len(items) * ratio))
            n_judge = min(n_judge, len(items) - 1)  # always leave >=1 to propose on
            judge_pool.extend(items[:n_judge])
            proposer_pool.extend(items[n_judge:])

        if not judge_pool:
            return failures_ext, failures_ext
        return proposer_pool, judge_pool

    def _sample_judge_failures(
        self,
        judge_pool: list[tuple[Any, ...]],
        batch_categories: list[str],
        count: int,
    ) -> list[tuple[Any, ...]]:
        """Rotating sample of held-out failures for judging, aligned to the
        categories the proposer batch targeted so the comparison stays on-topic.

        Uses an offset table separate from proposer sampling so the two never
        advance in lockstep.
        """
        if not judge_pool or count <= 0:
            return []
        by_cat: dict[str, list[tuple[Any, ...]]] = {}
        for entry in judge_pool:
            by_cat.setdefault(entry[4], []).append(entry)
        cats = [c for c in dict.fromkeys(batch_categories) if c in by_cat] or sorted(by_cat)

        selected: list[tuple[Any, ...]] = []
        seen: set[str] = set()
        i = 0
        guard = 0
        max_guard = count * 4 + len(judge_pool)
        while len(selected) < count and guard < max_guard:
            cat = cats[i % len(cats)]
            pool = by_cat[cat]
            off = self._judge_cat_offset.get(cat, 0)
            cand = pool[off % len(pool)]
            self._judge_cat_offset[cat] = off + 1
            key = str(cand[1])
            if key not in seen:
                seen.add(key)
                selected.append(cand)
            i += 1
            guard += 1
        return selected

    def _add_puct_child(
        self,
        parent: ProgramSearchNode,
        child_name: str,
        score: float,
        prior: float,
        discarded: bool = False,
    ) -> ProgramSearchNode:
        child = ProgramSearchNode(
            name=child_name,
            parent=parent,
            prior=max(0.05, min(1.0, prior)),
            score=score,
            depth=parent.depth + 1,
            discarded=discarded,
        )
        parent.children.append(child)
        self._backpropagate_puct(child, score)
        return child

    def _find_puct_node(
        self,
        root: ProgramSearchNode,
        name: str,
    ) -> ProgramSearchNode | None:
        if root.name == name:
            return root
        for child in root.children:
            found = self._find_puct_node(child, name)
            if found is not None:
                return found
        return None

    def _update_puct_node_score(
        self,
        root: ProgramSearchNode,
        name: str,
        score: float,
    ) -> None:
        """Update a node score and adjust backpropagated totals by the delta."""
        node = self._find_puct_node(root, name)
        if node is None:
            return
        delta = score - node.score
        if abs(delta) < 1e-12:
            return
        node.score = score
        current: ProgramSearchNode | None = node
        while current is not None:
            current.total_q += delta
            current = current.parent

    @staticmethod
    def _backpropagate_puct(node: ProgramSearchNode, value: float) -> None:
        current: ProgramSearchNode | None = node
        while current is not None:
            current.visit_count += 1
            current.total_q += value
            current = current.parent

    @staticmethod
    def _format_puct_path(node: ProgramSearchNode) -> str:
        path = []
        current: ProgramSearchNode | None = node
        while current is not None:
            path.append(current.name)
            current = current.parent
        return " -> ".join(reversed(path))

    async def _judge_skill_with_llm(
        self,
        failures_ext: list[tuple[Any, ...]],
        provider: str,
        model: str,
        child_name: str = "candidate",
        parent_name: str = "parent",
        parent_skill_summary: str = "",
        candidate_skill_summary: str = "",
        skill_diff: str = "",
        skill_diff_swapped: str = "",
        proposer_root_cause: str = "",
    ) -> JudgeResult:
        """Estimate failure-recovery quality with LLM-judged pairwise matches.

        Makes one lightweight API call per case (no agent re-run). The judge
        sees only the concrete skill diff/summaries — never the proposer's own
        rationale — so it cannot rubber-stamp the proposal's claims.
        Returns Elo ratings plus a calibrated failure-fix estimate in [0, 1].

        Position de-bias: the candidate is randomly assigned to neutral Skill Set
        A or B per case (``judge_randomize_orientation``) in a single call, and the
        canonical candidate-vs-parent score is recovered by un-swapping — turning a
        fixed-slot bias into zero-mean noise at no extra cost. ``judge_position_swap``
        instead judges both orientations and averages (2x calls) and takes
        precedence when set. The "Change Being Evaluated" shown to the judge is
        always the candidate's natural diff, independent of slot, so the
        root-cause-fit fields stay correct in either orientation.
        """
        import json
        import random as _random
        parent_skill_summary = self._compact_judge_text(parent_skill_summary, max_chars=2500)
        candidate_skill_summary = self._compact_judge_text(candidate_skill_summary, max_chars=2500)
        skill_diff = self._compact_judge_text(skill_diff, max_chars=3200)
        proposer_root_cause = self._compact_judge_text(proposer_root_cause, max_chars=1200)
        semaphore = asyncio.Semaphore(self.config.judge_concurrency)
        child_rating = self.config.judge_elo_initial_rating
        opponent_rating = self.config.judge_elo_initial_rating
        scale = self.config.judge_elo_scale
        k_factor = self.config.judge_elo_k

        async def call_orientation(
            trace_summary: str,
            q: str,
            ans: str,
            gt: str,
            case_type: str,
            case_feedback: str,
            *,
            a_summary: str,
            b_summary: str,
            diff: str,
        ) -> dict[str, Any] | None:
            """One judge API call for a single A/B orientation. Returns parsed
            JSON (with A in the parent slot, B in the candidate slot) or None on
            failure."""
            prompt = build_judge_query(
                trace_summary,
                q,
                ans,
                gt,
                parent_skill_summary=a_summary,
                candidate_skill_summary=b_summary,
                skill_diff=diff,
                case_type=case_type,
                case_feedback=case_feedback,
                proposer_root_cause=proposer_root_cause,
            )
            prompt = self._compact_judge_text(prompt, max_chars=20000, max_line_chars=1200)
            try:
                text = await asyncio.wait_for(
                    self._call_judge_api(prompt, provider, model),
                    timeout=self.config.judge_call_timeout_seconds,
                )
                text = text.strip()
                if text.startswith("```"):
                    text = "\n".join(text.split("\n")[1:])
                    text = text.rsplit("```", 1)[0].strip()
                data = json.loads(text)
                data["_raw_response"] = text
                return data
            except Exception as e:
                _log("WARN", f"  Judge call failed ({type(e).__name__}): {e}")
                return None

        def _failed_case(index: int, category: str, case_type: str, reason: str) -> dict[str, Any]:
            return {
                "_raw_response": "",
                "_index": index,
                "_category": f"{category}:regression" if case_type == "regression" else category,
                "would_succeed": False,
                "confidence": 0.0,
                "hypothetical_action": "",
                "reasoning": reason,
                "_valid": False,
            }

        async def judge_one(
            index: int,
            trace: AgentTrace,
            question: str,
            agent_answer: str,
            ground_truth: str,
            category: str,
            case_type: str,
            case_feedback: str,
        ) -> dict[str, Any]:
            async with semaphore:
                trace_summary = self._summarize_trace_for_judge(trace)
                q = self._compact_judge_text(question, max_chars=1600)
                ans = self._compact_judge_text(agent_answer, max_chars=1600)
                gt = self._compact_judge_text(ground_truth, max_chars=1200)
                fb = self._compact_judge_text(case_feedback, max_chars=2500)

                async def _orient(candidate_slot: str) -> dict[str, Any] | None:
                    """One call with the candidate placed in Set A or Set B.

                    The Change Being Evaluated is always the candidate's natural
                    diff regardless of slot; only the two summaries swap places.
                    """
                    if candidate_slot == "B":
                        a_sum, b_sum = parent_skill_summary, candidate_skill_summary
                    else:
                        a_sum, b_sum = candidate_skill_summary, parent_skill_summary
                    raw = await call_orientation(
                        trace_summary, q, ans, gt, case_type, fb,
                        a_summary=a_sum, b_summary=b_sum, diff=skill_diff,
                    )
                    if raw is None:
                        return None
                    return self._canonicalize_judge_orientation(raw, candidate_slot)

                if self.config.judge_position_swap:
                    # Two complementary orientations, averaged (2x calls).
                    d_b = await _orient("B")
                    d_a = await _orient("A")
                    if d_b is None and d_a is None:
                        return _failed_case(index, category, case_type, "Judge call failed (both orientations)")
                    if d_b is None:
                        data = d_a
                    elif d_a is None:
                        data = d_b
                    else:
                        data = d_b
                        match = 0.5 * (d_b["match_score"] + d_a["match_score"])
                        cp = 0.5 * (d_b["candidate_success_prob"] + d_a["candidate_success_prob"])
                        pp = 0.5 * (d_b["parent_success_prob"] + d_a["parent_success_prob"])
                        data["match_score"] = match
                        data["candidate_success_prob"] = cp
                        data["parent_success_prob"] = pp
                        data["relative_advantage"] = max(-1.0, min(1.0, cp - pp))
                        data["would_succeed"] = match >= 0.5
                        data["confidence"] = 0.5 * (
                            self._clamp01(d_b.get("confidence"), default=0.5)
                            + self._clamp01(d_a.get("confidence"), default=0.5)
                        )
                        for fld in (
                            "skill_addresses_root_cause",
                            "proposer_root_cause_correct",
                            "failure_mechanism_encoding",
                            "executable_specificity",
                            "high_risk_blacklist",
                            "generalization_transfer",
                        ):
                            data[fld] = 0.5 * (
                                self._clamp01(d_b.get(fld), default=0.0)
                                + self._clamp01(d_a.get(fld), default=0.0)
                            )
                        data["_position_averaged"] = True
                else:
                    # Single call; randomize the candidate's slot to de-bias.
                    if self.config.judge_randomize_orientation:
                        rng = _random.Random(f"orient::{child_name}::{index}")
                        candidate_slot = rng.choice(["A", "B"])
                    else:
                        candidate_slot = "B"
                    data = await _orient(candidate_slot)
                    if data is None:
                        return _failed_case(index, category, case_type, "Judge call failed")

                data["_index"] = index
                data["_category"] = (
                    f"{category}:regression" if case_type == "regression" else category
                )
                return data

        raw_results = []
        judge_tasks = []
        for i, entry in enumerate(failures_ext, start=1):
            t, q, ans, gt, cat = entry[:5]
            failure_type = self._trajectory_failure_type(entry)
            feedback = self._trajectory_failure_feedback(entry)
            case_type = "regression" if failure_type == "regression_pass" else "failure"
            judge_tasks.append(judge_one(i, t, q, ans, gt, cat, case_type, feedback))
        raw_results = await asyncio.gather(*judge_tasks)
        # Anchor ratings: all expected_before values are computed from the same
        # starting point so match ordering has no effect on the final rating.
        _initial_child_rating = child_rating
        _initial_opponent_rating = opponent_rating

        matches: list[JudgeMatchResult] = []
        direct_scores: list[float] = []
        for data in sorted(raw_results, key=lambda item: int(item.get("_index", 0))):
            is_valid = bool(data.get("_valid", True))
            if not is_valid:
                result = JudgeMatchResult(
                    index=int(data.get("_index", len(matches) + 1)),
                    category=str(data.get("_category", "")),
                    would_succeed=False,
                    confidence=0.0,
                    match_score=0.5,
                    expected_before=self._elo_expected(child_rating, opponent_rating, scale),
                    child_rating_after=child_rating,
                    opponent_rating_after=opponent_rating,
                    hypothetical_action=str(data.get("hypothetical_action", "")),
                    reasoning=str(data.get("reasoning", "")),
                    raw_response=str(data.get("_raw_response", "")),
                    valid=False,
                )
                matches.append(result)
                if self.config.judge_log_details:
                    _log(
                        "JUDGE MATCH",
                        (
                            f"{child_name} vs {parent_name} #{result.index} "
                            f"cat={result.category} invalid judge result; skipped"
                        ),
                    )
                    _log("", f"  reasoning: {result.reasoning}")
                continue

            confidence = self._clamp01(data.get("confidence"), default=0.5)
            would_succeed = self._parse_judge_bool(
                data.get("would_succeed"),
                default=False,
            )
            has_relative_score = (
                "match_score" in data
                or "parent_success_prob" in data
                or "candidate_success_prob" in data
                or "relative_advantage" in data
            )
            if has_relative_score:
                (
                    match_score,
                    parent_success_prob,
                    candidate_success_prob,
                    relative_advantage,
                ) = self._judge_relative_to_match_score(data)
            else:
                # No pairwise signal from the judge. Treat as a draw rather than
                # assuming an optimistic baseline (e.g. parent=0.5,
                # candidate=confidence), which systematically inflated every
                # child above its parent regardless of real merit.
                match_score = 0.5
                parent_success_prob = 0.5
                candidate_success_prob = 0.5
                relative_advantage = 0.0
            failure_mechanism_encoding = self._clamp01(
                data.get("failure_mechanism_encoding"),
                default=self._clamp01(data.get("skill_addresses_root_cause"), default=0.0),
            )
            executable_specificity = self._clamp01(data.get("executable_specificity"), default=0.0)
            high_risk_blacklist = self._clamp01(data.get("high_risk_blacklist"), default=0.0)
            generalization_transfer = self._clamp01(data.get("generalization_transfer"), default=0.5)
            artifact_quality = self._judge_artifact_quality(data)
            gated_match_score = self._apply_artifact_quality_gate(match_score, artifact_quality)
            if gated_match_score != match_score:
                match_score = gated_match_score
                relative_advantage = max(-1.0, min(1.0, 2.0 * (match_score - 0.5)))

            expected_before = self._elo_expected(_initial_child_rating, _initial_opponent_rating, scale)
            child_rating += k_factor * (match_score - expected_before)
            opponent_expected = 1.0 - expected_before
            opponent_rating += k_factor * ((1.0 - match_score) - opponent_expected)
            direct_scores.append(match_score)

            result = JudgeMatchResult(
                index=int(data.get("_index", len(matches) + 1)),
                category=str(data.get("_category", "")),
                would_succeed=would_succeed,
                confidence=confidence,
                match_score=match_score,
                expected_before=expected_before,
                child_rating_after=child_rating,
                opponent_rating_after=opponent_rating,
                hypothetical_action=str(
                    data.get("candidate_hypothetical_action")
                    or data.get("hypothetical_action", "")
                ),
                reasoning=str(data.get("reasoning", "")),
                raw_response=str(data.get("_raw_response", "")),
                valid=True,
                root_cause=str(data.get("root_cause", "")),
                skill_addresses_root_cause=self._clamp01(
                    data.get("skill_addresses_root_cause"),
                    default=0.0,
                ),
                proposer_root_cause_correct=self._clamp01(
                    data.get("proposer_root_cause_correct"),
                    default=0.0,
                ),
                failure_mechanism_encoding=failure_mechanism_encoding,
                executable_specificity=executable_specificity,
                high_risk_blacklist=high_risk_blacklist,
                generalization_transfer=generalization_transfer,
                probability_of_success=self._clamp01(
                    data.get("probability_of_success"),
                    default=candidate_success_prob,
                ),
                parent_success_prob=parent_success_prob,
                candidate_success_prob=candidate_success_prob,
                relative_advantage=relative_advantage,
                remaining_blockers=[
                    str(item) for item in data.get("remaining_blockers", [])[:5]
                ]
                if isinstance(data.get("remaining_blockers"), list)
                else [],
            )
            matches.append(result)

            if self.config.judge_log_details:
                _log(
                    "JUDGE MATCH",
                    (
                        f"{child_name} vs {parent_name} #{result.index} "
                        f"cat={result.category} outcome={result.match_score:.3f} "
                        f"expected={result.expected_before:.3f} "
                        f"rating={result.child_rating_after:.1f}/{result.opponent_rating_after:.1f} "
                        f"would_succeed={result.would_succeed} confidence={result.confidence:.3f} "
                        f"score_pass={result.match_score >= 0.5} "
                        f"p_parent={result.parent_success_prob:.3f} "
                        f"p_candidate={result.candidate_success_prob:.3f} "
                        f"rel_adv={result.relative_advantage:.3f} "
                        f"root_fit={result.skill_addresses_root_cause:.3f} "
                        f"rc_correct={result.proposer_root_cause_correct:.3f} "
                        f"mechanism={result.failure_mechanism_encoding:.3f} "
                        f"exec={result.executable_specificity:.3f} "
                        f"blacklist={result.high_risk_blacklist:.3f} "
                        f"transfer={result.generalization_transfer:.3f}"
                    ),
                )
                if result.root_cause:
                    _log("", f"  root_cause: {result.root_cause}")
                _log("", f"  action: {result.hypothetical_action}")
                if result.remaining_blockers:
                    _log("", f"  blockers: {', '.join(result.remaining_blockers)}")
                _log("", f"  reasoning: {result.reasoning}")
                _log("", f"  raw: {result.raw_response}")

        average_match_score = sum(direct_scores) / len(direct_scores) if direct_scores else 0.0
        if not direct_scores:
            reasons = "; ".join(match.reasoning for match in matches[:3])
            raise RuntimeError(f"All judge calls failed; cannot score candidate. {reasons}")
        expected_win_rate = self._elo_expected(child_rating, opponent_rating, scale)

        if self.config.judge_scoring == "average":
            estimated_fix_rate = average_match_score
        else:
            # Equal Elo means "no evidence of recovery over the failed baseline";
            # only rating above the opponent contributes to estimated recovery.
            estimated_fix_rate = max(0.0, min(1.0, 2.0 * (expected_win_rate - 0.5)))
        valid_confidences = [match.confidence for match in matches if match.valid]
        count_uncertainty = 1.0 / math.sqrt(max(1, len(direct_scores)))
        confidence_uncertainty = (
            1.0 - (sum(valid_confidences) / len(valid_confidences))
            if valid_confidences
            else 1.0
        )
        uncertainty = max(count_uncertainty, confidence_uncertainty)

        return JudgeResult(
            estimated_fix_rate=estimated_fix_rate,
            average_match_score=average_match_score,
            child_rating=child_rating,
            opponent_rating=opponent_rating,
            expected_win_rate=expected_win_rate,
            matches=matches,
            uncertainty=uncertainty,
        )

    @staticmethod
    def _summarize_judge_feedback(
        judge_result: "JudgeResult | None",
    ) -> tuple[str, list[str]]:
        """Distill a judge result into (root_cause, remaining_blockers) for the
        feedback history, so the next proposer sees *why* a proposal fell short
        rather than only its score.

        Blockers are drawn preferentially from the cases the candidate helped
        least (lowest match_score), since those are the unresolved failures the
        next iteration should target.
        """
        if judge_result is None or not judge_result.matches:
            return "", []

        matches = [m for m in judge_result.matches if getattr(m, "valid", True)]
        if not matches:
            matches = list(judge_result.matches)
        ordered = sorted(matches, key=lambda m: m.match_score)

        # Representative root cause: from the weakest-improvement case.
        root_cause = ""
        for m in ordered:
            if m.root_cause and m.root_cause.strip():
                root_cause = m.root_cause.strip()
                break

        # Union of blockers, weakest cases first, deduped (case-insensitive), capped.
        blockers: list[str] = []
        seen: set[str] = set()
        for m in ordered:
            for b in m.remaining_blockers:
                text = str(b).strip()
                key = text.lower()
                if text and key not in seen:
                    seen.add(key)
                    blockers.append(text)
                if len(blockers) >= 5:
                    return root_cause, blockers
        return root_cause, blockers

    @staticmethod
    def _mean_root_cause_fit(judge_result: "JudgeResult") -> float:
        """Mean judge-assessed root-cause-fit over valid matches in [0, 1]."""
        fits = [
            m.skill_addresses_root_cause
            for m in judge_result.matches
            if getattr(m, "valid", True)
        ]
        return sum(fits) / len(fits) if fits else 0.0

    @staticmethod
    def _weakest_case_reasoning(judge_result: "JudgeResult") -> str:
        """One-line judge reasoning from the case the candidate helped least."""
        valid = [m for m in judge_result.matches if getattr(m, "valid", True)]
        if not valid:
            return ""
        weakest = min(valid, key=lambda m: m.match_score)
        return weakest.reasoning or ""

    @staticmethod
    def _mean_generalization_transfer(judge_result: "JudgeResult") -> float:
        """Mean judge-estimated transferability over valid cases."""
        transfers = [
            max(0.0, min(1.0, getattr(m, "generalization_transfer", 0.5)))
            for m in judge_result.matches
            if getattr(m, "valid", True)
        ]
        return sum(transfers) / len(transfers) if transfers else 0.0

    @staticmethod
    def _has_misleading_worked_example(markdown: str) -> bool:
        """Detect a worked example that emits a scaffold count as the final answer.

        Flags the failure mode (e.g. UID0149) where a grid-size self-check such as
        ``ledger_count_check: 7 x 2 = 14`` shares its value with the example's
        stated ``Answer:`` / ``final_answer:`` — i.e. the agent would report the
        grid size instead of the predicate-satisfying subset count.
        """
        import re

        if not markdown:
            return False

        def _last_number(text: str) -> str | None:
            nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace(",", ""))
            return nums[-1] if nums else None

        scaffold_values: set[str] = set()
        answer_values: set[str] = set()
        scaffold_re = re.compile(
            r"(?im)^\s*(?:ledger_count_check|expected[\w ]*count|expected\b.*\btests?)\b.*$"
        )
        grid_re = re.compile(r"(?i)\b\d+\s*[x*×]\s*\d+\s*=\s*(\d[\d,]*)")
        answer_re = re.compile(r"(?im)^\s*(?:final_answer|answer)\s*[:=]\s*(.+)$")

        for line in markdown.splitlines():
            if scaffold_re.match(line):
                val = _last_number(line)
                if val is not None:
                    scaffold_values.add(val)
            for m in grid_re.finditer(line):
                scaffold_values.add(m.group(1).replace(",", ""))
            am = answer_re.match(line)
            if am:
                val = _last_number(am.group(1))
                if val is not None:
                    answer_values.add(val)

        return bool(scaffold_values & answer_values)

    def _snapshot_skill_files(self) -> dict[str, bytes]:
        """Capture current SKILL.md bytes so a failed revision can be rolled back."""
        skills_dir = self._project_root / ".claude" / "skills"
        snapshot: dict[str, bytes] = {}
        if skills_dir.exists():
            for path in skills_dir.rglob("SKILL.md"):
                snapshot[str(path)] = path.read_bytes()
        return snapshot

    def _restore_skill_files(self, snapshot: dict[str, bytes]) -> None:
        """Restore SKILL.md files captured by ``_snapshot_skill_files``."""
        for path_str, data in snapshot.items():
            path = Path(path_str)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    async def _refine_child_once(
        self,
        *,
        child_name: str,
        parent: str,
        skill_name: str,
        judge_result: "JudgeResult",
        scoring_cases_ext: list,
        judge_provider: str,
        judge_model: str,
        parent_skill_summary: str,
        parent_skills_content: str,
        proposer_root_cause: str,
        original_proposal: str,
    ) -> tuple["JudgeResult", str, str, str, str, bool]:
        """Revise the child's skill once from the judge's verdict, then re-judge.

        Returns (judge_result, candidate_skills_content, candidate_skill_summary,
        skill_diff, skill_diff_swapped, revised_kept). The pre-revision skill
        files and the original judge_result are restored whenever the revision
        does not improve the average match score vs the parent, so a refine can
        never lower a child's score.
        """
        def _artifacts() -> tuple[str, str, str, str]:
            content = self._get_all_skills_content()
            summary = self._summarize_skill_content(content)
            diff = self._diff_skill_content(parent_skills_content, content)
            diff_sw = self._diff_skill_content(content, parent_skills_content)
            return content, summary, diff, diff_sw

        skill_path = (
            self._project_root / ".claude" / "skills" / skill_name / "SKILL.md"
            if skill_name
            else None
        )
        if skill_path is None or not skill_path.exists():
            content, summary, diff, diff_sw = _artifacts()
            return judge_result, content, summary, diff, diff_sw, False

        judge_root_cause, judge_blockers = self._summarize_judge_feedback(judge_result)
        judge_reasoning = self._weakest_case_reasoning(judge_result)
        snapshot = self._snapshot_skill_files()
        revision_query = build_skill_revision_query(
            target_skill=skill_name,
            current_skill_markdown=skill_path.read_text(),
            proposer_root_cause=proposer_root_cause,
            original_proposal=original_proposal,
            judge_root_cause=judge_root_cause,
            judge_blockers=judge_blockers,
            judge_reasoning=judge_reasoning,
        )
        try:
            revise_trace = await self.agents.skill_generator.run(revision_query)
        except Exception as e:  # noqa: BLE001 — generator transport may raise broadly
            _log("WARN", f"  [REFINE] Generator failed ({type(e).__name__}: {e}); keeping original")
            content, summary, diff, diff_sw = _artifacts()
            return judge_result, content, summary, diff, diff_sw, False
        self._add_iteration_cost(revise_trace.total_cost_usd, "evolution")

        materialized = None
        if revise_trace.output:
            materialized = self._materialize_generated_skill(
                revise_trace.output,
                action_type="edit",
                target_skill=skill_name,
                fallback_description=original_proposal,
            )
        if not materialized:
            _log("", "  [REFINE] No revised SKILL.md produced; keeping original")
            self._restore_skill_files(snapshot)
            content, summary, diff, diff_sw = _artifacts()
            return judge_result, content, summary, diff, diff_sw, False

        rev_content, rev_summary, rev_diff, rev_diff_sw = _artifacts()
        try:
            rev_judge = await self._judge_skill_with_llm(
                scoring_cases_ext,
                judge_provider,
                judge_model,
                child_name=child_name,
                parent_name=parent,
                parent_skill_summary=parent_skill_summary,
                candidate_skill_summary=rev_summary,
                skill_diff=rev_diff,
                skill_diff_swapped=rev_diff_sw,
                proposer_root_cause=proposer_root_cause,
            )
        except RuntimeError as e:
            _log("WARN", f"  [REFINE] Re-judge failed ({e}); keeping original")
            self._restore_skill_files(snapshot)
            content, summary, diff, diff_sw = _artifacts()
            return judge_result, content, summary, diff, diff_sw, False

        if rev_judge.average_match_score > judge_result.average_match_score:
            # Persist the kept revision onto the child's program branch so a later
            # switch_to(child) (PUCT re-selection or final best extraction) does
            # not restore the pre-refine version from git.
            try:
                self.manager.commit(f"{child_name}: refine {skill_name} (judge feedback)")
            except Exception as e:  # noqa: BLE001 — commit backends vary
                _log("WARN", f"  [REFINE] Could not commit revision for {child_name}: {e}")
            _log(
                "REFINE",
                (
                    f"{child_name}: revision improved avg_match "
                    f"{judge_result.average_match_score:.3f} -> "
                    f"{rev_judge.average_match_score:.3f}; kept"
                ),
            )
            return rev_judge, rev_content, rev_summary, rev_diff, rev_diff_sw, True

        _log(
            "REFINE",
            (
                f"{child_name}: revision did not improve "
                f"({rev_judge.average_match_score:.3f} <= "
                f"{judge_result.average_match_score:.3f}); reverted"
            ),
        )
        self._restore_skill_files(snapshot)
        content, summary, diff, diff_sw = _artifacts()
        return judge_result, content, summary, diff, diff_sw, False

    async def _run_with_llm_judge(self) -> LoopResult:
        """LLM Judge mode: collect all trajectories once, then iterate on skill
        evolution using lightweight LLM judge calls instead of full agent re-evaluation.

        Key differences from the original run():
        - No train/val split: all data is run once at the start.
        - Skill quality is judged by asking an LLM to predict whether the new
          skill would have helped each failing trajectory.
        - The full agent is never re-run during the evolution loop.
        """
        # 0. Handle continue mode / feedback reset
        if not self.config.continue_mode:
            if self.config.reset_feedback and self._feedback_path.exists():
                self._feedback_path.unlink()
            self._iteration_offset = 0
            self._delete_checkpoint()
            reset_manager = getattr(self.manager, "reset", None)
            if callable(reset_manager):
                reset_manager()
                _log("INIT", "Reset local program state for fresh run")
        else:
            self._iteration_offset = self._get_highest_iteration()

        # 1. Ensure base program exists
        if "base" not in self.manager.list_programs():
            current_options = self.agents.base._get_options()
            base_config = options_to_config(current_options, "base")
            self.manager.create_program("base", base_config)
            _log("INIT", "Created base program")
        else:
            _log("INIT", "Using existing base program")
        self.manager.switch_to("base")

        # 2. Obtain trajectories — either from preloaded data or by running all samples now
        if self._preloaded_trajectories is not None:
            extended_traces = self._preloaded_trajectories
            _log("COLLECT", f"Using {len(extended_traces)} pre-collected trajectories (skipping agent inference)")
            preloaded_cost = sum(float(trace.total_cost_usd or 0.0) for trace, *_ in extended_traces)
            self._add_preloaded_trajectory_cost(preloaded_cost)
            _log("COST", f"Preloaded trajectory cost: ${preloaded_cost:.4f} | {self._format_cost_breakdown()}")
        else:
            all_data = self._get_all_data()
            _log("COLLECT", f"Running all {len(all_data)} samples upfront (no train/val split)...")
            self._iter_cost = 0.0
            extended_traces = await self._collect_all_trajectories(all_data)
            self._total_cost += self._iter_cost
            _log("COST", f"Trajectory collection: ${self._iter_cost:.4f} | {self._format_cost_breakdown()}")

        # 3. Compute base score and partition failures
        all_failures_ext: list[tuple[Any, ...]] = []
        all_successes_ext: list[tuple[Any, ...]] = []
        passed = 0
        for entry in extended_traces:
            trace, question, agent_answer, ground_truth, category = entry[:5]
            failure_type = self._trajectory_failure_type(entry)
            failure_feedback = self._trajectory_failure_feedback(entry)
            score = self.scorer(question, agent_answer.strip().lower(), ground_truth.strip().lower())
            if score >= self.config.failure_threshold:
                passed += 1
                all_successes_ext.append((
                    trace,
                    question,
                    agent_answer,
                    ground_truth,
                    category,
                    "regression_pass",
                    build_regression_success_feedback(
                        question,
                        agent_answer,
                        ground_truth,
                        domain_hints=self.config.error_surface_hints,
                    ),
                ))
            else:
                if not failure_feedback:
                    failure_feedback = build_answer_comparison_feedback(
                        question,
                        agent_answer,
                        ground_truth,
                        failure_type,
                        domain_hints=self.config.error_surface_hints,
                    )
                all_failures_ext.append((
                    trace,
                    question,
                    agent_answer,
                    ground_truth,
                    category,
                    failure_type,
                    failure_feedback,
                ))

        total = len(extended_traces)
        base_score = passed / total if total > 0 else 0.0
        _log(
            "COLLECT",
            f"Base score: {base_score:.4f} ({passed}/{total} passed, {len(all_failures_ext)} failures)",
        )
        # Hold out a disjoint slice of failures for judging so candidates are
        # never scored on the exact cases the proposer was shown.
        proposer_failures_ext, judge_failures_ext = self._split_holdout(all_failures_ext)
        if proposer_failures_ext is not judge_failures_ext:
            _log(
                "HOLDOUT",
                f"{len(proposer_failures_ext)} proposer-visible failures / "
                f"{len(judge_failures_ext)} held-out judge failures "
                f"(ratio={self.config.judge_holdout_ratio:.2f})",
            )
        else:
            _log("HOLDOUT", "Disabled or too few failures; proposer and judge share all cases")
        _log("JUDGE", "Judging on held-out failures + inherited parent set")
        self.manager.update_frontier("base", base_score, max_size=self.config.frontier_size)
        self._emit("baseline", score=base_score, n_skills=len(self._get_active_skills()))
        search_root = self._build_program_search_tree(base_score)
        bt_matches: list[BradleyTerryMatch] = []
        bt_players: set[str] = {"base"}
        bt_player_judge_results: dict[str, JudgeResult] = {}
        bt_ratings: dict[str, float] = {
            "base": self.config.judge_elo_initial_rating,
        }
        _log(
            "PUCT",
            (
                f"Enabled: c={self.config.puct_c}, max_depth={self.config.puct_max_depth}, "
                f"children_per_node={self.config.puct_children_per_node}, "
                f"children_per_iteration={self.config.children_per_iteration}"
            ),
        )

        # Determine judge provider/model once
        judge_provider, judge_model = self._detect_judge_provider_and_model()
        _log("JUDGE", f"Judge: {judge_provider} / {judge_model}")
        if self.config.judge_scoring == "bradley_terry":
            _log("JUDGE", "Scoring: global Bradley-Terry league over all generated nodes")
            _log(
                "JUDGE",
                f"BT uncertainty penalty: {self.config.judge_bt_uncertainty_penalty:.3f}",
            )

        # 4. Evolution loop
        # Categories come from trajectories when preloaded, else from train_pools
        if self._preloaded_trajectories is not None:
            categories = sorted({entry[4] for entry in extended_traces})
            failure_types = sorted(
                {
                    self._trajectory_failure_type(entry)
                    for entry in extended_traces
                    if self._trajectory_failure_type(entry)
                }
            )
        else:
            categories = sorted(self.train_pools.keys())
            failure_types = []
        n_cats = len(categories)
        no_improvement_count = 0
        iteration_count = 0

        for i in range(self.config.max_iterations):
            iteration_count = i + 1
            actual_iteration = iteration_count + self._iteration_offset

            parent_node = self._select_puct_node(search_root)
            parent = parent_node.name
            self.manager.switch_to(parent)
            self._iter_cost = 0.0
            _log(
                f"ITER {iteration_count}/{self.config.max_iterations}",
                (
                    f"Parent: {parent} | PUCT path: {self._format_puct_path(parent_node)} | "
                    f"q={parent_node.q_value:.4f}, visits={parent_node.visit_count}, "
                    f"children={len(parent_node.children)}"
                ),
            )
            self._emit("iter_start", iteration=actual_iteration, total=self.config.max_iterations, parent=parent)

            parent_score = parent_node.score
            parent_skills_content = self._get_all_skills_content()
            # Capture best frontier score before this iteration to detect genuine improvement.
            _frontier_before = self.manager.get_frontier_with_scores()
            best_score_before_iter = _frontier_before[0][1] if _frontier_before else base_score

            if not all_failures_ext:
                _log("", "  -> No failures remaining")
                break

            # The proposer only ever sees the proposer-visible pool.
            batch_failures_ext = self._sample_proposal_failures(
                proposer_failures_ext, categories, failure_types
            )
            # The judge scores on HELD-OUT failures the proposer never saw, drawn
            # from the same categories the proposal targeted. Inheriting the
            # parent's scoring set keeps child_score vs parent_score a fair
            # comparison on a consistent (held-out) evaluation set.
            batch_categories = [f[4] for f in batch_failures_ext]
            judge_batch_ext = self._sample_judge_failures(
                judge_failures_ext,
                batch_categories,
                count=max(len(batch_failures_ext), self.config.failure_sample_count),
            )
            # By default the judge scores only this batch's held-out failures
            # (plus a small regression sample below), not the parent's inherited
            # superset — inheriting was the dominant judge-call multiplier.
            inherited_failures = (
                [
                    entry
                    for entry in parent_node.scoring_failures
                    if self._trajectory_failure_type(entry) != "regression_pass"
                ]
                if self.config.judge_inherit_parent_failures
                else []
            )
            scoring_failures_ext = self._merge_scoring_failures(
                inherited_failures,
                judge_batch_ext,
                max_size=self.config.samples_per_category * self.config.categories_per_batch * 3,
            )
            regression_cases_ext = self._sample_regression_successes(
                all_successes_ext,
                categories,
                count=max(1, len(batch_failures_ext)),
            )
            regression_questions = {entry[1] for entry in regression_cases_ext}
            scoring_cases_ext = [
                entry
                for entry in scoring_failures_ext
                if entry[1] not in regression_questions
            ] + regression_cases_ext

            # Convert to legacy 4-tuple format for _mutate; keep questions separately
            # so the proposer sees the full question text alongside the trace.
            proposal_questions = [f[1] for f in batch_failures_ext]
            batch_failures = []
            for failure_entry in batch_failures_ext:
                t, q, ans, gt, cat = failure_entry[:5]
                ft = self._trajectory_failure_type(failure_entry)
                feedback = self._trajectory_failure_feedback(failure_entry)
                if not feedback:
                    feedback = build_answer_comparison_feedback(
                        q, ans, gt, ft,
                        domain_hints=self.config.error_surface_hints,
                    )
                batch_failures.append((t, ans, gt, cat, feedback))
            parent_questions = {f[1] for f in parent_node.scoring_failures}
            n_new = sum(1 for f in scoring_failures_ext if f[1] not in parent_questions)
            n_inherited = len(scoring_failures_ext) - n_new
            _log(
                "",
                (
                    f"  Using {len(batch_failures)} proposal failures; "
                    f"judging on {len(scoring_failures_ext)} failure cases "
                    f"({n_inherited} inherited from parent + {n_new} new) "
                    f"+ {len(regression_cases_ext)} regression cases..."
                ),
            )

            target_children = max(1, self.config.children_per_iteration)
            remaining_child_slots = max(
                0,
                self.config.puct_children_per_node - len(parent_node.children),
            )
            children_to_generate = min(target_children, remaining_child_slots)
            if children_to_generate <= 0:
                _log("", "  [SKIP] Selected parent has no remaining child slots")
                no_improvement_count += 1
                continue

            _log(
                "",
                (
                    f"  Expanding {children_to_generate}/{target_children} child "
                    f"candidate(s) from {parent}"
                ),
            )
            any_child_created = False
            sibling_proposals: list[str] = []

            for child_idx in range(children_to_generate):
                self.manager.switch_to(parent)
                child_iteration_id: int | str
                if target_children == 1:
                    child_iteration_id = actual_iteration
                else:
                    child_iteration_id = f"{actual_iteration}-{child_idx + 1}"

                _log(
                    "CHILD",
                    f"{child_idx + 1}/{children_to_generate} from parent {parent}",
                )
                diversity_hint = self._build_child_diversity_hint(child_idx, sibling_proposals)
                mutation_result = await self._mutate_with_fallback(
                    parent,
                    batch_failures,
                    child_iteration_id,
                    diversity_hint=diversity_hint,
                    questions=proposal_questions,
                )

                if mutation_result is None:
                    _log("", "  [WARN] Child generation failed")
                    continue

                any_child_created = True
                child_name = mutation_result.child_name
                proposal = mutation_result.proposal
                justification = mutation_result.justification
                proposer_confidence = mutation_result.proposer_confidence
                sibling_proposals.append(proposal)
                candidate_skills_content = self._get_all_skills_content()
                parent_skill_summary = self._summarize_skill_content(parent_skills_content)
                candidate_skill_summary = self._summarize_skill_content(candidate_skills_content)
                skill_diff = self._diff_skill_content(parent_skills_content, candidate_skills_content)
                skill_diff_swapped = self._diff_skill_content(candidate_skills_content, parent_skills_content)

                # Judge with LLM — no full agent re-run
                _log("", f"  -> Judging {child_name} via LLM ({judge_provider}/{judge_model})...")
                try:
                    judge_result = await self._judge_skill_with_llm(
                        scoring_cases_ext,
                        judge_provider,
                        judge_model,
                        child_name=child_name,
                        parent_name=parent,
                        parent_skill_summary=parent_skill_summary,
                        candidate_skill_summary=candidate_skill_summary,
                        skill_diff=skill_diff,
                        skill_diff_swapped=skill_diff_swapped,
                        proposer_root_cause=mutation_result.root_cause_analysis,
                    )
                except RuntimeError as e:
                    _log("WARN", f"  [WARN] All judge calls failed for {child_name}: {e}; skipping child")
                    continue

                # judge→generator refine loop: when the judge finds the candidate
                # does not address the true root cause (or does not beat parent),
                # feed its verdict back to the generator for one targeted revision
                # and re-judge, keeping the better-scoring version.
                refine_round = 0
                while (
                    self.config.refine_with_judge_feedback
                    and refine_round < self.config.refine_max_rounds
                ):
                    mean_root_fit = self._mean_root_cause_fit(judge_result)
                    mean_transfer = self._mean_generalization_transfer(judge_result)
                    misleading_example = (
                        self.config.validate_worked_examples
                        and self._has_misleading_worked_example(candidate_skills_content)
                    )
                    needs_refine = (
                        mean_root_fit < self.config.refine_root_cause_threshold
                        or mean_transfer < self.config.refine_generalization_threshold
                        or judge_result.average_match_score <= 0.5
                        or misleading_example
                    )
                    if not needs_refine:
                        break
                    if misleading_example:
                        _log(
                            "REFINE",
                            f"{child_name}: worked example emits a scaffold count as the answer "
                            "-> revising",
                        )
                    _log(
                        "REFINE",
                        (
                            f"{child_name}: root_fit={mean_root_fit:.3f} "
                            f"transfer={mean_transfer:.3f} "
                            f"avg_match={judge_result.average_match_score:.3f} "
                            f"-> revising (round {refine_round + 1}/{self.config.refine_max_rounds})"
                        ),
                    )
                    (
                        judge_result,
                        candidate_skills_content,
                        candidate_skill_summary,
                        skill_diff,
                        skill_diff_swapped,
                        revised_kept,
                    ) = await self._refine_child_once(
                        child_name=child_name,
                        parent=parent,
                        skill_name=mutation_result.skill_name,
                        judge_result=judge_result,
                        scoring_cases_ext=scoring_cases_ext,
                        judge_provider=judge_provider,
                        judge_model=judge_model,
                        parent_skill_summary=parent_skill_summary,
                        parent_skills_content=parent_skills_content,
                        proposer_root_cause=mutation_result.root_cause_analysis,
                        original_proposal=proposal,
                    )
                    refine_round += 1
                    if not revised_kept:
                        break

                # Estimate total score: Elo-estimated fix rate is the fraction of
                # failed samples the child is expected to recover. Scale it to the
                # observed base-score space.
                forced_discard = False
                if self.config.judge_scoring == "bradley_terry":
                    async def _judge_vs_anchor(anchor_name: str) -> "JudgeResult | None":
                        anchor_skills_content = self._get_program_skills_content(
                            anchor_name,
                            restore_to=child_name,
                        )
                        a_summary = self._summarize_skill_content(anchor_skills_content)
                        a_diff = self._diff_skill_content(anchor_skills_content, candidate_skills_content)
                        a_diff_sw = self._diff_skill_content(candidate_skills_content, anchor_skills_content)
                        _log("", f"  -> Judging {child_name} vs {anchor_name}...")
                        try:
                            return await self._judge_skill_with_llm(
                                scoring_cases_ext,
                                judge_provider,
                                judge_model,
                                child_name=child_name,
                                parent_name=anchor_name,
                                parent_skill_summary=a_summary,
                                candidate_skill_summary=candidate_skill_summary,
                                skill_diff=a_diff,
                                skill_diff_swapped=a_diff_sw,
                                proposer_root_cause=mutation_result.root_cause_analysis,
                            )
                        except RuntimeError as e:
                            _log("WARN", f"  [WARN] Anchor judge failed ({child_name} vs {anchor_name}): {e}; skipping")
                            return None

                    # The parent edge is free — already judged for the refine loop.
                    anchor_results: list[tuple[str, JudgeResult]] = [(parent, judge_result)]
                    used_anchors: set[str] = {parent}

                    if self.config.judge_champion_gate:
                        # Stage 1: duel the current frontier champion.
                        champion = self.manager.get_best_from_frontier() or "base"
                        if champion == child_name:
                            champion = "base"
                        if champion == parent:
                            champ_result: "JudgeResult | None" = judge_result
                        else:
                            champ_result = await _judge_vs_anchor(champion)
                            if champ_result is not None:
                                anchor_results.append((champion, champ_result))
                                used_anchors.add(champion)
                        decision = self._duel_decision(champ_result)
                        champ_match = (
                            f"{champ_result.average_match_score:.3f}"
                            if champ_result is not None
                            else "n/a"
                        )
                        _log(
                            "JUDGE",
                            f"{child_name} vs champion {champion}: duel={decision} "
                            f"(avg_match={champ_match})",
                        )
                        if decision == "loss":
                            # Clearly worse than the champion → eliminate after one duel.
                            forced_discard = True
                        elif decision == "close":
                            # Stage 2: a few random frontier members to disambiguate.
                            extra = self._random_frontier_anchors(
                                used_anchors | {child_name},
                                self.config.judge_stage2_anchors,
                            )
                            if extra:
                                _log("JUDGE", f"Close duel; extra frontier PKs: {', '.join(extra)}")
                            for anchor in extra:
                                r = await _judge_vs_anchor(anchor)
                                if r is not None:
                                    anchor_results.append((anchor, r))
                                    used_anchors.add(anchor)
                        # decision == "win": champion edge is decisive; no extra PKs.
                    else:
                        # Legacy fixed anchor set (parent + frontier best).
                        for anchor in self._select_bt_anchor_nodes(parent, child_name):
                            if anchor in used_anchors:
                                continue
                            r = await _judge_vs_anchor(anchor)
                            if r is not None:
                                anchor_results.append((anchor, r))
                                used_anchors.add(anchor)

                    anchors = sorted(used_anchors)
                    bt_players.update({child_name, *used_anchors})
                    bt_player_judge_results[child_name] = judge_result
                    new_bt_matches = [
                        BradleyTerryMatch(
                            player=child_name,
                            opponent=anchor,
                            score=match.match_score,
                            category=match.category,
                            index=match.index,
                        )
                        for anchor, result in anchor_results
                        for match in result.matches
                        if match.valid
                    ]
                    bt_matches.extend(new_bt_matches)
                    # Only fit over matches where both sides are currently
                    # active (frontier + new child + anchors).  Old discarded
                    # nodes stay in bt_matches for history but must not shift
                    # ratings of present candidates.
                    _active_players = (
                        {"base", child_name}
                        | {n for n, _ in self.manager.get_frontier_with_scores()}
                        | set(anchors)
                    )
                    _active_bt_matches = self._filter_bt_matches_for_active_players(
                        bt_matches, _active_players
                    )
                    bt_ratings = self._fit_bradley_terry_ratings(
                        _active_bt_matches,
                        sorted(_active_players),
                        anchor="base",
                        initial_rating=self.config.judge_elo_initial_rating,
                        scale=self.config.judge_elo_scale,
                    )
                    base_rating = bt_ratings.get("base", self.config.judge_elo_initial_rating)
                    child_rating = bt_ratings.get(child_name, self.config.judge_elo_initial_rating)
                    child_uncertainty = self._bt_player_uncertainty(
                        child_name,
                        _active_bt_matches,
                        bt_player_judge_results,
                    )
                    child_raw_score = self._rating_to_score(
                        child_rating,
                        base_rating,
                        base_score,
                        self.config.judge_elo_scale,
                    )
                    child_score = self._rating_to_score_with_uncertainty(
                        child_rating,
                        base_rating,
                        base_score,
                        self.config.judge_elo_scale,
                        child_uncertainty,
                    )
                    # Only update the frontier for the new child and nodes already in the
                    # frontier. Updating all BT-rated nodes would re-admit previously
                    # excluded programs whose relative rating rose due to new competitors.
                    existing_frontier_names = {name for name, _ in self.manager.get_frontier_with_scores()}
                    for rated_name, rating in bt_ratings.items():
                        if rated_name not in _active_players:
                            continue
                        rated_score = (
                            base_score
                            if rated_name == "base"
                            else self._rating_to_score_with_uncertainty(
                                rating,
                                base_rating,
                                base_score,
                                self.config.judge_elo_scale,
                                self._bt_player_uncertainty(
                                    rated_name,
                                    _active_bt_matches,
                                    bt_player_judge_results,
                                ),
                            )
                        )
                        # A clear-loss child is eliminated: keep it out of the
                        # frontier even if its rating lands acceptably.
                        if (rated_name == child_name and not forced_discard) or (
                            rated_name in existing_frontier_names
                        ):
                            self.manager.update_frontier(
                                rated_name,
                                rated_score,
                                max_size=self.config.frontier_size,
                            )
                        # Always sync PUCT node scores for accurate tree navigation.
                        self._update_puct_node_score(search_root, rated_name, rated_score)

                    expected_win = self._elo_expected(
                        child_rating,
                        base_rating,
                        self.config.judge_elo_scale,
                    )
                    avg_new_match = (
                        sum(m.score for m in new_bt_matches) / len(new_bt_matches)
                        if new_bt_matches
                        else 0.5
                    )
                    _log(
                        "",
                        (
                            f"  -> Judge BT: global_rating={child_rating:.1f} "
                            f"base_rating={base_rating:.1f}, expected_vs_base={expected_win:.3f}, "
                            f"avg_match={avg_new_match:.3f} "
                            f"matches={len(new_bt_matches)} anchors={len(anchors)} "
                            f"uncertainty={child_uncertainty:.3f} "
                            f"raw_score={child_raw_score:.4f} "
                            f"→ penalized score {child_score:.4f}"
                        ),
                    )
                else:
                    # Map the child-vs-parent rating advantage symmetrically so a
                    # child judged worse than its parent can score BELOW base. The
                    # old `base + fix_rate*(1-base)` floored every child at base and
                    # made the frontier rise monotonically by construction.
                    child_score = self._rating_to_score(
                        judge_result.child_rating,
                        judge_result.opponent_rating,
                        base_score,
                        self.config.judge_elo_scale,
                    )
                    _log(
                        "",
                        (
                            f"  -> Judge Elo: fix_rate={judge_result.estimated_fix_rate:.3f}, "
                            f"avg_match={judge_result.average_match_score:.3f}, "
                            f"rating={judge_result.child_rating:.1f}/{judge_result.opponent_rating:.1f}, "
                            f"expected_win={judge_result.expected_win_rate:.3f} "
                            f"→ estimated score {child_score:.4f}"
                        ),
                    )

                if forced_discard:
                    added = False
                    outcome = "duel_loss"
                    _log("", f"  [DUEL-LOSS] {child_name} lost to champion (score: {child_score:.4f}); eliminated")
                else:
                    added = self.manager.update_frontier(child_name, child_score, max_size=self.config.frontier_size)
                    if added:
                        outcome = "improved" if child_score > parent_score else "kept"
                        _log("", f"  [OK] Added to frontier (score: {child_score:.4f})")
                    else:
                        outcome = "not_frontier"
                        _log("", f"  [SKIP] Not added to frontier (score: {child_score:.4f}); kept for PUCT exploration")

                # Mark node as discarded when it lost its champion duel or scores
                # clearly below base, so PUCT stops revisiting a dead branch.
                is_discarded = forced_discard or child_score < base_score - 0.05
                child_node = self._add_puct_child(
                    parent_node,
                    child_name,
                    child_score,
                    prior=self._proposal_policy_prior(proposer_confidence),
                    discarded=is_discarded,
                )
                # Record which failures this child was scored on so its own
                # children will inherit them and comparisons remain consistent.
                child_node.scoring_failures = scoring_cases_ext
                if is_discarded:
                    _log("PUCT", f"Marked {child_name} as discarded (score {child_score:.4f} < base {base_score:.4f} - 0.05)")
                _log(
                    "PUCT",
                    (
                        f"Backpropagated {child_score:.4f} through {self._format_puct_path(child_node)}; "
                        f"prior={child_node.prior:.3f}, "
                        f"parent q={parent_node.q_value:.4f}, visits={parent_node.visit_count}"
                    ),
                )

                self._emit(
                    "eval_result",
                    child_name=child_name,
                    score=child_score,
                    parent_score=parent_score,
                    added=added,
                    frontier=self.manager.get_frontier_with_scores(),
                    n_skills=len(self._get_active_skills()),
                )
                judge_root_cause, judge_blockers = self._summarize_judge_feedback(judge_result)
                append_feedback(
                    self._feedback_path,
                    child_name,
                    proposal,
                    justification,
                    outcome=outcome,
                    score=child_score,
                    parent_score=parent_score,
                    active_skills=self._get_active_skills(),
                    root_cause=judge_root_cause or None,
                    remaining_blockers=judge_blockers or None,
                )

            _frontier_after = self.manager.get_frontier_with_scores()
            best_score_after_iter = _frontier_after[0][1] if _frontier_after else base_score
            if best_score_after_iter > best_score_before_iter:
                no_improvement_count = 0
            elif any_child_created:
                no_improvement_count += 1
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.config.no_improvement_limit:
                _log("STOP", f"No improvement for {self.config.no_improvement_limit} iterations")
                break

            frontier_str = ", ".join(f"{n}:{s:.2f}" for n, s in self.manager.get_frontier_with_scores())
            _log("", f"  Frontier: [{frontier_str}]")
            self._total_cost += self._iter_cost
            _log("COST", f"Iter {iteration_count} cost: ${self._iter_cost:.4f} | Running total: ${self._total_cost:.4f}")
            self._save_checkpoint(actual_iteration)

        # 5. Return results
        frontier = self.manager.get_frontier_with_scores()
        best = self.manager.get_best_from_frontier()
        best_score = frontier[0][1] if frontier else 0.0
        _log("DONE", f"{iteration_count} iterations, best: {best or 'base'} ({best_score:.4f})")
        _log("COST", self._format_cost_breakdown())
        self._emit("loop_done", best=best or "base", best_score=best_score, iterations=iteration_count)
        return self._build_loop_result(
            frontier,
            best or "base",
            best_score,
            iteration_count,
        )
