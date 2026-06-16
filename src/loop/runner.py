"""Self-improving agent loop runner."""

import asyncio
import difflib
import json
import math
import re
import shutil
import tempfile
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
    ToolGeneratorResponse,
    PromptGeneratorResponse,
    SkillProposerResponse,
    SkillEdit,
    SkillVerifierResponse,
    PromptProposerResponse,
)

from .config import LoopConfig
from .curriculum import CategoryBandit
from .playbook import apply_playbook_delta, update_bullet_counters
from .helpers import (
    build_answer_comparison_feedback,
    build_regression_success_feedback,
    build_proposer_query,
    build_skill_query_from_skill_proposer,
    build_prompt_query_from_prompt_proposer,
    build_judge_query,
    build_compact_judge_query,
    build_skill_revision_query,
    compact_relevant_text,
    append_feedback,
    read_feedback_history,
    consolidate_meta,
    read_meta,
    write_meta,
    update_prompt_file,
)


T = TypeVar("T")

TOLERANCE_LEVELS = [0.05, 0.01, 0.1, 0.0, 0.025]


def _no_improvement_stop(no_improvement_count: int, limit: int | None) -> bool:
    """Whether the no-improvement early stop should fire.

    A falsy ``limit`` (0 or None) disables early stopping entirely, so the search
    runs the full ``max_iterations`` budget. This matters for budget-based PUCT
    runs (e.g. a 50-node search) where the best score legitimately plateaus for
    many iterations while the tree is still being productively explored — a small
    default limit (5) would otherwise cut the run off after iteration 6.
    """
    return bool(limit) and no_improvement_count >= limit

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
    # Phase 1 (playbook mode): the skill whose playbook was edited and the ids of
    # bullets this proposal added or reinforced, so the loop can credit their
    # helpful counters once the held-out gate confirms an improvement.
    target_skill: str = ""
    credited_bullets: list[str] = field(default_factory=list)
    # Multi-skill proposals: one entry per skill created/edited by this proposal,
    # each {skill_name, target_skill, action, credited_bullets}. Includes the
    # primary edit; used to credit every touched skill's playbook on improvement.
    applied_skills: list[dict] = field(default_factory=list)


@dataclass
class LoopAgents:
    """Container for the agents used in the loop."""

    base: Agent[AgentResponse]
    skill_proposer: Agent[SkillProposerResponse]
    prompt_proposer: Agent[PromptProposerResponse]
    skill_generator: Agent[ToolGeneratorResponse]
    prompt_generator: Agent[PromptGeneratorResponse]
    # Optional information-isolated verifier (CoEvoSkills). When present and
    # config.use_skill_verifier is on, it independently authors the skill's test
    # cases so they are not gameable by the skill author. Default None keeps every
    # existing LoopAgents construction site valid.
    skill_verifier: "Agent[SkillVerifierResponse] | None" = None


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
    self_test_cost_usd: float = 0.0
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
    case_role: str = "other"
    case_weight: float = 1.0
    root_cause: str = ""
    case_relevance: float = 0.0
    skill_addresses_root_cause: float = 0.0
    proposer_root_cause_correct: float = 0.0
    failure_mechanism_encoding: float = 0.0
    executable_specificity: float = 0.0
    high_risk_blacklist: float = 0.0
    generalization_transfer: float = 0.0
    self_test_pass_rate: float = 0.0
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


@dataclass
class ExecutableSkillTestCaseResult:
    """Measured result for one generator-produced synthetic skill test."""

    skill_name: str
    test_name: str
    should_activate: bool
    activated: bool
    passed: bool
    reason: str
    agent_answer: str = ""
    missing_checks: list[str] = field(default_factory=list)
    missing_rejections: list[str] = field(default_factory=list)


@dataclass
class ExecutableSkillTestSuiteResult:
    """Measured executable test-suite result passed to the judge."""

    passed: int
    total: int
    pass_rate: float
    cases: list[ExecutableSkillTestCaseResult] = field(default_factory=list)

    def to_report(self, max_cases: int = 8) -> str:
        if self.total <= 0:
            return "No executable skill tests were run."
        lines = [
            "EXECUTABLE_SKILL_TEST_RESULTS",
            f"actual_pass_rate: {self.pass_rate:.3f} ({self.passed}/{self.total})",
        ]
        for case in self.cases[:max_cases]:
            status = "PASS" if case.passed else "FAIL"
            lines.append(
                (
                    f"- {status} {case.skill_name}/{case.test_name}: "
                    f"should_activate={case.should_activate} activated={case.activated}; "
                    f"reason={case.reason}"
                )
            )
            if case.missing_checks:
                lines.append(f"  missing_checks: {', '.join(case.missing_checks)}")
            if case.missing_rejections:
                lines.append(f"  missing_rejections: {', '.join(case.missing_rejections)}")
            if case.agent_answer:
                answer = case.agent_answer.replace("\n", " ")[:300]
                lines.append(f"  agent_answer: {answer}")
        if len(self.cases) > max_cases:
            lines.append(f"- ... {len(self.cases) - max_cases} more executable test cases omitted")
        return "\n".join(lines)


@dataclass(frozen=True)
class BradleyTerryMatch:
    """One soft pairwise comparison for the global Bradley-Terry league."""

    player: str
    opponent: str
    score: float
    category: str = ""
    index: int = 0
    weight: float = 1.0


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
    proposal: str = ""
    proposal_action: str = ""
    proposal_skill: str = ""

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
        # Evolution root for the LLM-judge loop: "base", or the seed program from
        # the pre-PUCT coverage sweep. Used as the BT gauge anchor and to exclude
        # base from head-to-head anchors.
        self._evolution_root: str = "base"
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

        # Phase 3: curriculum bandit over failure categories. Initialized from
        # whichever category set is known at construction time; updated with
        # proposal outcomes during the loop. Only consulted when
        # config.failure_selection == "bandit".
        self._bandit = CategoryBandit(
            list(self._per_cat_offset.keys()),
            epsilon=config.failure_bandit_epsilon,
            ema=config.failure_bandit_ema,
        )
        # Keys (failure_type or category) the bandit picked for the most recent
        # proposal batch, so the judge loop can credit them with the iteration's
        # realized gain. Set inside _sample_proposal_failures.
        self._last_bandit_keys: list[str] = []
        # Skill-induction-from-success scheduling: a fractional accumulator turns
        # success_induction_ratio into whole induction children over time, and an
        # offset rotates which successes are sampled.
        self._induction_credit: float = 0.0
        self._induction_offset: int = 0

        # Epoch slow/meta update: per-child outcome records accumulated across the
        # run, consolidated into a protected meta summary every meta_update_period
        # iterations (SkillOpt). In-memory; regenerated as the run continues.
        self._meta_records: list[dict[str, Any]] = []

        # Paths
        self._project_root = Path(getattr(self.manager, "cwd", Path.cwd())).resolve()
        self._feedback_path = self._project_root / ".evoskill" / "feedback_history.md"
        self._meta_path = self._project_root / ".evoskill" / "meta.md"
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
        self._self_test_cost_usd: float = 0.0
        self._judge_cost_usd: float = 0.0
        self._judge_prompt_tokens: int = 0
        self._judge_completion_tokens: int = 0
        self._judge_total_tokens: int = 0
        self._executable_skill_test_cache: dict[str, ExecutableSkillTestSuiteResult] = {}
        # Cache verifier-authored tests by SKILL.md content hash, so an unchanged
        # skill is not re-verified (saves a verifier LLM call per repeat).
        self._verifier_tests_cache: dict[str, list] = {}

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
        elif bucket == "self_test":
            self._self_test_cost_usd = getattr(self, "_self_test_cost_usd", 0.0) + amount
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
            self_test_cost_usd=self._self_test_cost_usd,
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
            f"self_test=${self._self_test_cost_usd:.4f}, "
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

            # Pick which categories to sample failures from this iteration.
            # Default: round-robin sweep. Bandit mode: weight by learning gain.
            n_cats_this_iter = min(self.config.categories_per_batch, n_cats)

            if self.config.failure_selection == "bandit":
                iter_cats = self._bandit.select(categories, n_cats_this_iter)
            else:
                iter_cats = [
                    categories[(self._category_offset + j) % n_cats]
                    for j in range(n_cats_this_iter)
                ]

            test_samples: list[tuple[str, str, str]] = []
            sampled_cats: list[str] = []
            for cat in iter_cats:
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

                # Update frontier or discard. QD mode also stores a per-category
                # coverage vector and keeps one elite per niche.
                if self.config.frontier_mode == "qd":
                    gen_samples = self._generalization_samples(in_sample_samples)
                    coverage = await self._evaluate_by_category(
                        gen_samples or in_sample_samples
                    )
                    added = self.manager.update_frontier_qd(
                        child_name, child_score, coverage
                    )
                else:
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

                # Phase 3: fold this outcome into the curriculum bandit so future
                # iterations weight categories by realized learning gain.
                accepted = added and child_score > parent_score
                delta = child_score - parent_score
                for cat in set(sampled_cats):
                    self._bandit.update(cat, accepted, delta)

                # Phase 1: an accepted (held-out-improving) child proves the
                # bullets it added were helpful. Persist the bump on the child
                # branch and COMMIT it — an uncommitted SKILL.md edit would be
                # stashed away (and lost) by the next branch switch. A discarded
                # child's branch, and its new bullets, are thrown away, so there
                # is nothing to debit; harmful credit is left to the explicit
                # `retire` op and future per-case attribution.
                if self.config.use_playbook and accepted:
                    bumped = self._credit_playbook(mutation_result)
                    if bumped:
                        self.manager.commit(
                            f"{child_name}: playbook +helpful x{bumped}"
                        )

            # Check early stopping (disabled when no_improvement_limit is 0/None)
            if _no_improvement_stop(no_improvement_count, self.config.no_improvement_limit):
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

    async def _evaluate_by_category(
        self, data: list[tuple[str, str, str]]
    ) -> dict[str, float]:
        """Per-category mean score over ``data`` (Phase 2 QD coverage vector).

        Reuses the run cache, so when called right after ``_evaluate`` on the
        same program tree it incurs no extra agent calls.
        """
        by_cat: dict[str, list[float]] = {}
        if not data:
            return {}
        qa_data = [(q, a) for q, a, _ in data]
        cats = [c for _, _, c in data]
        results = await evaluate_agent_parallel(
            self.agents.base, qa_data, max_concurrent=self.config.concurrency, cache=self.cache
        )
        for result, cat in zip(results, cats):
            if result.trace is None or result.trace.output is None:
                s = 0.0
            else:
                s = self.scorer(
                    result.question,
                    result.trace.output.final_answer,
                    result.ground_truth,
                )
            by_cat.setdefault(cat, []).append(s)
        return {c: (sum(v) / len(v)) for c, v in by_cat.items() if v}

    @staticmethod
    def _judge_coverage_vector(judge_result: "JudgeResult") -> dict[str, float]:
        """Per-category coverage vector from a child's judge matches (Phase 2 QD).

        Uses the mean valid match score per category as the niche descriptor:
        > 0.5 means the child beat its opponent on that category. The judging set
        already mixes held-out failures with regression (success-trajectory)
        cases, so this vector reflects both fixing and not-breaking per category.
        """
        by_cat: dict[str, list[float]] = {}
        for match in getattr(judge_result, "matches", []) or []:
            if not getattr(match, "valid", True):
                continue
            cat = getattr(match, "category", "") or "uncategorized"
            by_cat.setdefault(cat, []).append(match.match_score)
        return {c: sum(v) / len(v) for c, v in by_cat.items() if v}

    def _credit_playbook(self, mutation_result: "MutationResult") -> int:
        """Bump helpful counters for every skill this proposal touched.

        Iterates the multi-skill ``applied_skills`` (falling back to the singular
        fields). Returns the total number of bullet counters bumped, so the caller
        knows whether to commit. Must be called on the child's branch.
        """
        entries = mutation_result.applied_skills
        if not entries and mutation_result.target_skill and mutation_result.credited_bullets:
            entries = [{
                "skill_name": mutation_result.target_skill,
                "credited_bullets": mutation_result.credited_bullets,
            }]
        total = 0
        for entry in entries:
            bullets = entry.get("credited_bullets") or []
            skill = entry.get("skill_name") or entry.get("target_skill") or ""
            if skill and bullets:
                update_bullet_counters(self._project_root, skill, bullets, helpful=True)
                total += len(bullets)
        return total

    async def _generate_one_skill(
        self,
        edit: "SkillEdit",
        *,
        justification: str,
        root_cause_analysis: str,
        coverage_plan: str,
    ) -> dict | None:
        """Generate or edit ONE skill on the current (child) branch.

        Used to layer additional skills from a multi-skill proposal onto the same
        child program (the primary edit is handled inline by ``_mutate``). Returns
        ``{skill_name, target_skill, action, credited_bullets}`` or None if the
        edit produced nothing usable (the caller keeps the other edits).
        """
        import hashlib

        action_type = edit.action
        target_skill = edit.target_skill
        proposed = edit.proposed_skill

        if action_type == "edit" and target_skill:
            _log("", f"  -> [multi-skill] Editing existing skill: {target_skill}...")
            current_skill_path = (
                self._project_root / ".claude" / "skills" / target_skill / "SKILL.md"
            )
            current_skill_markdown = (
                current_skill_path.read_text(errors="replace")
                if current_skill_path.exists()
                else "[missing]"
            )
            skill_query = f"""EDIT existing skill: {target_skill}

Modifications needed:
{proposed}

Filesystem rule: the authoritative current skill is printed below. Do not read
any external files — especially do not read trajectory files (.jsonl), dataset
files (.csv), or any files under `.evoskill/trajectories/`. Do not search
`/home`, `/tmp`, `.codex`, historical runs, or unrelated repos. Use only:
`.claude/skills/{target_skill}/SKILL.md`.

Justification: {justification}

Root cause analysis from proposer:
{root_cause_analysis or "[not provided]"}

Failure coverage plan:
{coverage_plan or "[not provided]"}

Use this skill only when:
{edit.should_apply_when}

Do not use this skill when:
{edit.should_not_apply_when}

Invariants to preserve:
{edit.invariants_to_preserve}

Regression risks / anti-regression guards:
{edit.regression_risks or "[not provided]"}

## Current SKILL.md
```markdown
{current_skill_markdown}
```

Modify this SKILL.md to add these capabilities. Preserve all existing content
that is still relevant and return the complete revised file."""
        else:
            _log("", "  -> [multi-skill] Generating new skill...")
            skill_query = f"""Proposed tool or skill (high level description): {proposed}

Justification: {justification}

Root cause analysis from proposer:
{root_cause_analysis or "[not provided]"}

Failure coverage plan:
{coverage_plan or "[not provided]"}

Use this skill only when:
{edit.should_apply_when}

Do not use this skill when:
{edit.should_not_apply_when}

Invariants to preserve:
{edit.invariants_to_preserve}

Regression risks / anti-regression guards:
{edit.regression_risks or "[not provided]"}"""

        skills_before = set(self._get_active_skills())
        _pre_hash: str | None = None
        _pre_text: str | None = None
        if action_type == "edit" and target_skill:
            _sp = self._project_root / ".claude" / "skills" / target_skill / "SKILL.md"
            if _sp.exists():
                b = _sp.read_bytes()
                _pre_hash = hashlib.md5(b).hexdigest()
                _pre_text = b.decode("utf-8", errors="replace")

        skill_trace = await self.agents.skill_generator.run(skill_query)
        self._add_iteration_cost(skill_trace.total_cost_usd, "evolution")
        if skill_trace.is_error or skill_trace.parse_error:
            _log("", f"  [WARN] [multi-skill] generator parse error: {skill_trace.parse_error}")

        materialized_skill = None
        if skill_trace.output:
            materialized_skill = self._materialize_generated_skill(
                skill_trace.output,
                action_type=action_type,
                target_skill=target_skill,
                fallback_description=proposed,
            )
        skills_after = set(self._get_active_skills())
        new_skills = skills_after - skills_before
        created_skill = next(iter(new_skills)) if new_skills else materialized_skill

        # Validate (same guards as the primary edit), but on failure skip THIS
        # edit instead of discarding the whole child.
        if action_type == "edit" and target_skill:
            expected = self._project_root / ".claude" / "skills" / target_skill / "SKILL.md"
            changed = False
            if expected.exists():
                changed = hashlib.md5(expected.read_bytes()).hexdigest() != _pre_hash
            skill_valid = expected.exists() and (_pre_hash is None or changed)
            if skill_valid and _pre_text is not None and expected.exists():
                new_text = expected.read_text()
                old_len = len(_pre_text.strip())
                new_len = len(new_text.strip())
                ratio = max(0.0, min(1.0, self.config.skill_edit_min_retention_ratio))
                length_ok = old_len == 0 or new_len >= ratio * old_len
                dropped = self._section_headers(_pre_text) - self._section_headers(new_text)
                if not length_ok or dropped:
                    skill_valid = False
            if not skill_valid:
                _log("", f"  [WARN] [multi-skill] edit on {target_skill} invalid; skipping it")
                return None
            resolved = target_skill
        else:
            if not created_skill:
                _log("", "  [WARN] [multi-skill] generator produced no SKILL.md; skipping it")
                return None
            resolved = created_skill

        if is_opencode_sdk() or is_openhands_sdk() or is_goose_sdk() or is_codex_sdk():
            from src.harness.opencode.skill_utils import (
                DEFAULT_OPENHANDS_SKILL_TRIGGERS,
                normalize_project_skill_frontmatter,
            )
            from src.harness.sdk_config import get_sdk
            descriptions = {s: proposed for s in {target_skill, created_skill} if s}
            compatibility = get_sdk()
            triggers = (
                {s: DEFAULT_OPENHANDS_SKILL_TRIGGERS for s in descriptions}
                if compatibility == "openhands"
                else None
            )
            normalize_project_skill_frontmatter(
                self._project_root,
                descriptions=descriptions,
                fallback_description=proposed,
                compatibility=compatibility,
                triggers_by_skill=triggers,
            )

        credited_bullets: list[str] = []
        if self.config.use_playbook and resolved:
            ops = [op.model_dump() for op in (edit.bullet_ops or [])]
            if ops:
                credited_bullets = apply_playbook_delta(
                    self._project_root,
                    resolved,
                    ops,
                    dedup_threshold=self.config.playbook_dedup_threshold,
                    max_bullets=self.config.playbook_max_bullets,
                )
        _log("", f"  -> [multi-skill] applied {action_type} on {resolved}")
        return {
            "skill_name": resolved,
            "target_skill": target_skill if action_type == "edit" else "",
            "action": action_type,
            "credited_bullets": credited_bullets,
        }

    async def _mutate(
        self,
        parent: str,
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str]],
        iteration: int | str,
        truncation_level: int = 0,
        diversity_hint: str = "",
        questions: list[str] | None = None,
        induction: bool = False,
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
        source_label = "success-induction" if induction else "failure"
        _log(
            "",
            f"  -> Running {evolution_mode.replace('_only', '')} proposer "
            f"with {len(failures)} {source_label} sample(s)...",
        )
        feedback_history = read_feedback_history(self._feedback_path)
        # Prepend the protected epoch meta summary (longitudinal memory) so the
        # proposer sees consolidated guidance ahead of raw step-level history.
        meta_summary = read_meta(self._meta_path)
        if meta_summary.strip():
            feedback_history = f"{meta_summary}\n\n{feedback_history}"
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
            max_active_skills=self.config.max_active_skills,
            induction=induction,
            use_playbook=self.config.use_playbook,
        )
        _log(
            "PROPOSER",
            (
                f"query built mode={evolution_mode} samples={len(failures)} "
                f"chars={len(proposer_query)} truncation={truncation_level} "
                f"induction={induction}"
            ),
        )

        if evolution_mode == "skill_only":
            _log("PROPOSER", "skill proposer call start")
            proposer_trace = await self.agents.skill_proposer.run(proposer_query)
            _log(
                "PROPOSER",
                (
                    "skill proposer call done "
                    f"duration_ms={proposer_trace.duration_ms} "
                    f"turns={proposer_trace.num_turns} "
                    f"cost=${proposer_trace.total_cost_usd:.4f} "
                    f"is_error={proposer_trace.is_error} "
                    f"parse_error={proposer_trace.parse_error!r}"
                ),
            )
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
            action_type = proposer_output.action
            target_skill = proposer_output.target_skill
            # Edits modify an existing SKILL.md that already carries its boundary
            # sections, so the proposer need not re-supply them; only a create
            # (which must produce a complete new skill) requires all boundaries.
            required_boundaries = {
                "should_apply_when": should_apply_when,
                "should_not_apply_when": should_not_apply_when,
                "invariants_to_preserve": invariants_to_preserve,
                "regression_risks": regression_risks,
            }
            missing_boundaries = [
                name for name, value in required_boundaries.items() if not value.strip()
            ]
            if missing_boundaries and not (action_type == "edit" and target_skill):
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
            _log(
                "PROPOSER",
                (
                    f"output action={action_type} target={target_skill or '<new>'} "
                    f"confidence={proposer_confidence:.3f} "
                    f"proposal_chars={len(proposed)} "
                    f"root_cause_chars={len(root_cause_analysis)}"
                ),
            )

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
                current_skill_path = (
                    self._project_root / ".claude" / "skills" / target_skill / "SKILL.md"
                )
                current_skill_markdown = (
                    current_skill_path.read_text(errors="replace")
                    if current_skill_path.exists()
                    else "[missing]"
                )
                mechanism_label = (
                    "Success mechanism from proposer"
                    if induction
                    else "Root cause analysis from proposer"
                )
                coverage_label = (
                    "Success transfer plan"
                    if induction
                    else "Failure coverage plan"
                )
                skill_query = f"""EDIT existing skill: {target_skill}

Modifications needed:
{proposed.strip() or justification}

Filesystem rule: the authoritative current skill is printed below. Do not read
any external files — especially do not read trajectory files (.jsonl), dataset
files (.csv), or any files under `.evoskill/trajectories/`. Do not search
`/home`, `/tmp`, `.codex`, historical runs, or unrelated repos. Use only:
`.claude/skills/{target_skill}/SKILL.md`.

Justification: {justification}

{mechanism_label}:
{root_cause_analysis or "[not provided]"}

{coverage_label}:
{coverage_plan or "[not provided]"}

Use this skill only when:
{should_apply_when}

Do not use this skill when:
{should_not_apply_when}

Invariants to preserve:
{invariants_to_preserve}

Regression risks / anti-regression guards:
{regression_risks or "[not provided]"}

## Current SKILL.md
```markdown
{current_skill_markdown}
```

Modify this SKILL.md to add these capabilities. Preserve all existing content
that is still relevant and return the complete revised file."""
            else:
                _log("", f"  -> Generating new skill...")
                skill_query = build_skill_query_from_skill_proposer(proposer_trace)
                if induction:
                    skill_query = (
                        "SUCCESS-INDUCTION MODE: the proposer analyzed successful "
                        "trajectories. Generate a skill that captures the reusable "
                        "success mechanism, activation boundary, and preservation "
                        "guards. Do not frame this as repairing a known failure.\n\n"
                        + skill_query.replace(
                            "Root cause analysis from proposer:",
                            "Success mechanism from proposer:",
                        ).replace(
                            "Failure coverage plan:",
                            "Success transfer plan:",
                        )
                    )
            _log("GENERATOR", f"skill generator query built chars={len(skill_query)}")

            skills_before = set(self._get_active_skills())
            # Capture pre-run content hash + text for edit-mode diagnostics and
            # the content-retention guard.
            _pre_skill_hash: str | None = None
            _pre_skill_text: str | None = None
            _changed = False
            if action_type == "edit" and target_skill:
                import hashlib
                _sp = self._project_root / ".claude" / "skills" / target_skill / "SKILL.md"
                if _sp.exists():
                    _pre_skill_bytes = _sp.read_bytes()
                    _pre_skill_hash = hashlib.md5(_pre_skill_bytes).hexdigest()
                    _pre_skill_text = _pre_skill_bytes.decode("utf-8", errors="replace")
            _log("GENERATOR", "skill generator call start")
            skill_trace = await self.agents.skill_generator.run(skill_query)
            _log(
                "GENERATOR",
                (
                    "skill generator call done "
                    f"duration_ms={skill_trace.duration_ms} "
                    f"turns={skill_trace.num_turns} "
                    f"cost=${skill_trace.total_cost_usd:.4f} "
                    f"is_error={skill_trace.is_error} "
                    f"parse_error={skill_trace.parse_error!r}"
                ),
            )
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
                discard_reason = "did not produce a changed SKILL.md"
                # Content-retention guard: reject an edit that silently dropped a
                # large fraction of the file or removed a previously-present
                # section, instead of preserving still-valid content.
                if skill_valid and _pre_skill_text is not None and expected_skill_path.exists():
                    new_text = expected_skill_path.read_text()
                    old_len = len(_pre_skill_text.strip())
                    new_len = len(new_text.strip())
                    ratio = max(0.0, min(1.0, self.config.skill_edit_min_retention_ratio))
                    length_ok = old_len == 0 or new_len >= ratio * old_len
                    dropped_sections = self._section_headers(_pre_skill_text) - self._section_headers(new_text)
                    if not length_ok or dropped_sections:
                        skill_valid = False
                        discard_reason = (
                            f"regressed content (len {old_len}->{new_len}, "
                            f"dropped sections: {sorted(dropped_sections)})"
                        )
                if not skill_valid:
                    _log(
                        "",
                        f"  [WARN] Skill edit {discard_reason}; discarding {child_name}",
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
            _log("PROPOSER", "prompt proposer call start")
            proposer_trace = await self.agents.prompt_proposer.run(proposer_query)
            _log(
                "PROPOSER",
                (
                    "prompt proposer call done "
                    f"duration_ms={proposer_trace.duration_ms} "
                    f"turns={proposer_trace.num_turns} "
                    f"cost=${proposer_trace.total_cost_usd:.4f} "
                    f"is_error={proposer_trace.is_error} "
                    f"parse_error={proposer_trace.parse_error!r}"
                ),
            )
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

        # Phase 1: apply the proposer's playbook delta to the resolved skill,
        # layering managed bullets (with helpful/harmful counters) onto the
        # generated SKILL.md before commit, so the change is versioned together.
        credited_bullets: list[str] = []
        playbook_target = ""
        if evolution_mode == "skill_only" and self.config.use_playbook:
            ops = [op.model_dump() for op in getattr(proposer_output, "bullet_ops", [])]
            if ops:
                playbook_target = (
                    target_skill
                    if action_type == "edit" and target_skill
                    else (created_skill or "")
                )
                if playbook_target:
                    credited_bullets = apply_playbook_delta(
                        self._project_root,
                        playbook_target,
                        ops,
                        dedup_threshold=self.config.playbook_dedup_threshold,
                        max_bullets=self.config.playbook_max_bullets,
                    )
                    _log(
                        "",
                        f"  -> Playbook delta on {playbook_target}: "
                        f"{len(ops)} op(s), {len(credited_bullets)} bullet(s) credited",
                    )

        # Multi-skill proposals: record the primary edit, then layer any extra
        # skills from the proposal onto this SAME child program before commit.
        applied_skills: list[dict] = []
        if evolution_mode == "skill_only":
            primary_skill = (
                target_skill
                if action_type == "edit" and target_skill
                else (created_skill or "")
            )
            if primary_skill:
                applied_skills.append({
                    "skill_name": primary_skill,
                    "target_skill": primary_skill,
                    "action": action_type,
                    "credited_bullets": credited_bullets,
                })
            if self.config.enable_multi_skill:
                for edit in proposer_output.extra_edits():
                    info = await self._generate_one_skill(
                        edit,
                        justification=justification,
                        root_cause_analysis=root_cause_analysis,
                        coverage_plan=coverage_plan,
                    )
                    if info:
                        applied_skills.append(info)
            if len(applied_skills) > 1:
                _log(
                    "",
                    f"  -> Multi-skill proposal applied {len(applied_skills)} skills: "
                    + ", ".join(a["skill_name"] for a in applied_skills),
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
            target_skill=playbook_target,
            credited_bullets=credited_bullets,
            applied_skills=applied_skills,
        )

    async def _mutate_with_fallback(
        self,
        parent: str,
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str]],
        iteration: int | str,
        diversity_hint: str = "",
        questions: list[str] | None = None,
        induction: bool = False,
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
            attempt_iteration = (
                iteration
                if truncation_level == 0
                else f"{iteration}-retry{truncation_level}"
            )

            result = await self._mutate(
                parent,
                failures,
                attempt_iteration,
                truncation_level,
                diversity_hint=diversity_hint,
                questions=questions,
                induction=induction,
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
                f"{iteration}-single",
                truncation_level=max_level,
                diversity_hint=diversity_hint,
                induction=induction,
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
            # The generator emitted an empty/reserved/degenerate name (e.g. a
            # YAML-null surfacing as "null"). Derive a valid slug from the skill's
            # own description rather than dropping an otherwise-usable skill.
            derived = self._slugify_skill_name(
                self._skill_description_from_markdown(skill_markdown)
            ) or self._slugify_skill_name(fallback_description)
            if derived:
                _log(
                    "",
                    f"  [WARN] Generator returned invalid skill name {skill_name!r}; "
                    f"using derived name {derived!r}",
                )
                skill_name = derived
            else:
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

        # Keep the on-disk directory name and the frontmatter name in sync so a
        # derived/edited skill is never invoked under a stale or degenerate name.
        skill_markdown = self._force_frontmatter_name(skill_markdown, skill_name)
        skill_path = self._project_root / ".claude" / "skills" / skill_name / "SKILL.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)

        # Repair a leaked edit-instruction in the description before it reaches the
        # skill list. fallback_description is the proposer's edit prose ("Edit X:
        # (1) Add a playbook bullet ..."), which must never become the trigger.
        prior_desc = ""
        if skill_path.exists():
            prior_desc = self._skill_description_from_markdown(skill_path.read_text())
        clean_desc = self._sanitize_skill_description(
            self._skill_description_from_markdown(skill_markdown) or fallback_description,
            prior=prior_desc,
            body=skill_markdown,
        )
        skill_markdown = self._force_frontmatter_description(skill_markdown, clean_desc)

        if not skill_markdown.endswith("\n"):
            skill_markdown += "\n"
        skill_path.write_text(skill_markdown)

        from src.harness.opencode.skill_utils import ensure_skill_frontmatter

        ensure_skill_frontmatter(
            skill_path,
            description=clean_desc,
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
        tests_payload = getattr(output, "skill_tests", None)
        if tests_payload is not None:
            import json

            tests_path = skill_path.parent / "SKILL_TESTS.json"
            if isinstance(tests_payload, list):
                serializable_tests = [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in tests_payload
                ]
                tests_path.write_text(
                    json.dumps(serializable_tests, ensure_ascii=False, indent=2) + "\n"
                )
                _log(
                    "",
                    f"  -> Materialized SKILL_TESTS.json: {skill_name} ({len(tests_payload)} tests)",
                )
            else:
                _log("", f"  [WARN] Ignoring invalid skill_tests payload for {skill_name}")
        return skill_name

    def _materialize_distilled_skill_bundle(
        self,
        output: ToolGeneratorResponse,
        *,
        fallback_description: str,
    ) -> list[str]:
        """Materialize a final distilled bundle containing one or more skills.

        The normal generator schema is single-skill. For final frontier
        distillation we allow ``generated_skill`` or ``skill_markdown`` to carry
        a JSON object like ``{"skills": [{"skill_markdown": "...",
        "skill_tests": [...]}, ...]}`` so independent mechanisms are preserved as
        separate skills instead of being collapsed into one broad instruction.
        """
        import json

        candidates = [
            str(getattr(output, "generated_skill", "") or "").strip(),
            str(getattr(output, "skill_markdown", "") or "").strip(),
        ]
        for raw in candidates:
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            skills = data.get("skills") if isinstance(data, dict) else None
            if not isinstance(skills, list):
                continue
            materialized: list[str] = []
            for item in skills:
                if not isinstance(item, dict):
                    continue
                single = ToolGeneratorResponse(
                    generated_skill=str(item.get("generated_skill") or item.get("name") or ""),
                    reasoning=str(item.get("reasoning") or fallback_description),
                    skill_path=item.get("skill_path"),
                    skill_markdown=item.get("skill_markdown") or item.get("markdown"),
                    skill_tests=item.get("skill_tests"),
                )
                name = self._materialize_generated_skill(
                    single,
                    action_type="create",
                    target_skill=None,
                    fallback_description=fallback_description,
                )
                if name:
                    materialized.append(name)
            if materialized:
                return materialized

        name = self._materialize_generated_skill(
            output,
            action_type="create",
            target_skill=None,
            fallback_description=fallback_description,
        )
        return [name] if name else []

    # Degenerate names a model emits when it leaves the name blank: a YAML/JSON
    # null or a stray literal. They pass the slug regex but must never become a
    # skill directory (".claude/skills/null/").
    _RESERVED_SKILL_NAMES = frozenset(
        {"null", "none", "nil", "nan", "true", "false", "undefined", "skill", "skills"}
    )

    @staticmethod
    def _is_valid_skill_name(skill_name: str) -> bool:
        import re

        if not skill_name or len(skill_name) < 3:
            return False
        if skill_name.lower() in SelfImprovingLoop._RESERVED_SKILL_NAMES:
            return False
        return bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", skill_name))

    @staticmethod
    def _slugify_skill_name(text: str) -> str | None:
        """Derive a valid hyphenated skill name from free text (e.g. a description)."""
        import re

        filler = {
            "invoke", "when", "the", "a", "an", "for", "to", "of", "and", "or",
            "is", "are", "agent", "skill", "this", "that", "with", "use", "using",
            "should", "must", "be", "in", "on", "it",
        }
        words = [w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in filler]
        slug = "-".join(words[:5])
        return slug if SelfImprovingLoop._is_valid_skill_name(slug) else None

    @staticmethod
    def _skill_description_from_markdown(markdown: str) -> str:
        import re

        m = re.search(r"\A---\n(.*?)\n---", markdown, flags=re.DOTALL)
        if not m:
            return ""
        d = re.search(r"(?m)^description:\s*(.+)$", m.group(1))
        return d.group(1).strip().strip("'\"") if d else ""

    @staticmethod
    def _force_frontmatter_name(markdown: str, name: str) -> str:
        """Rewrite the frontmatter ``name:`` so it matches the resolved dir name."""
        import re

        m = re.match(r"\A---\n(.*?)\n---", markdown, flags=re.DOTALL)
        if not m:
            return markdown
        fm = m.group(1)
        if re.search(r"(?m)^name:\s*.*$", fm):
            new_fm = re.sub(r"(?m)^name:\s*.*$", f"name: {name}", fm, count=1)
        else:
            new_fm = f"name: {name}\n{fm}"
        return markdown[: m.start(1)] + new_fm + markdown[m.end(1) :]

    @staticmethod
    def _force_frontmatter_description(markdown: str, description: str) -> str:
        """Rewrite the frontmatter ``description:`` (the only deployed trigger surface).

        Handles a multi-line YAML description value: the old single-line regex
        replaced only the ``description:`` line and orphaned its indented
        continuation lines, leaving a corrupt frontmatter that callers (and the
        deployed agent) could not read. Here the whole key — its line plus every
        continuation line up to the next top-level key — is removed and replaced
        with one clean single-line description.
        """
        import re

        m = re.match(r"\A---\n(.*?)\n---", markdown, flags=re.DOTALL)
        if not m:
            return markdown
        one_line = " ".join(str(description).split())
        lines = m.group(1).split("\n")
        out: list[str] = []
        i = 0
        replaced = False
        key_re = re.compile(r"^[A-Za-z0-9_][\w.-]*\s*:")
        while i < len(lines):
            if re.match(r"^description\s*:", lines[i]):
                i += 1
                # Consume continuation lines (indented / backslash-folded / blank)
                # until the next top-level YAML key.
                while i < len(lines) and not key_re.match(lines[i]):
                    i += 1
                out.append(f"description: {one_line}")
                replaced = True
            else:
                out.append(lines[i])
                i += 1
        if not replaced:
            out.append(f"description: {one_line}")
        new_fm = "\n".join(out)
        return markdown[: m.start(1)] + new_fm + markdown[m.end(1) :]

    # Strong multi-word phrases that betray an edit-instruction / changelog wrongly
    # placed in the description (the only text the model sees when deciding to invoke
    # a skill). Deliberately specific: a merely long description is NOT corruption —
    # only prose that reads as a change instruction is.
    _BAD_DESCRIPTION_MARKERS = (
        "playbook bullet",
        "no other changes",
        "add a new bullet",
        "add two new",
        "to the skill body",
        "edit the skill",
        "no change to the skill",
        "expand the \"when",
        "expand the when to use",
    )

    @classmethod
    def _sanitize_skill_description(
        cls, candidate: str, *, prior: str = "", body: str = ""
    ) -> str:
        """Return a clean one-line trigger description, repairing instruction leaks.

        The generator (especially on playbook edits, where ``fallback_description``
        was the proposer's edit-instruction prose) sometimes wrote the change
        instructions into ``description`` instead of a trigger. Such a description
        never fires and pollutes the skill list. Detect that specific failure and
        fall back to the skill's prior good description, then to the candidate's
        first sentence. A long-but-valid trigger is left untouched.
        """
        def _bad(text: str) -> bool:
            low = text.lower()
            return any(mark in low for mark in cls._BAD_DESCRIPTION_MARKERS) or (
                low.startswith("edit ") and "invoke when" not in low
            )

        one_line = " ".join(str(candidate or "").split())
        if one_line and not _bad(one_line):
            return one_line
        prior_line = " ".join(str(prior or "").split())
        if prior_line and not _bad(prior_line):
            return prior_line
        # Last resort: keep the candidate's first sentence (often the real trigger
        # before the instruction prose begins), else a generic placeholder.
        first = one_line.split(". ")[0].strip()
        if first and not _bad(first):
            return first[:300]
        return "Invoke when this skill's documented trigger conditions are met."

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

    async def _author_verifier_tests(self, skill_names: list[str]) -> int:
        """CoEvoSkills: have the information-isolated verifier author each skill's
        test cases, overwriting the generator-written SKILL_TESTS.json.

        The verifier sees the skill and the task domain but NOT ground-truth
        answers, and is a different agent from the skill author, so the resulting
        self-test signal cannot be gamed. No new agent rollout is added: the tests
        feed the existing executable-test / judge self-test-pass-rate path. Returns
        the number of tests written.
        """
        if not (self.config.use_skill_verifier and self.agents.skill_verifier is not None):
            return 0
        import hashlib
        import json

        meta = read_meta(self._meta_path)
        max_tests = max(1, int(self.config.skill_verifier_max_tests))
        total = 0
        for skill in dict.fromkeys(s for s in skill_names if s):
            skill_dir = self._project_root / ".claude" / "skills" / skill
            md_path = skill_dir / "SKILL.md"
            if not md_path.exists():
                continue
            skill_md_full = md_path.read_text(errors="replace")
            skill_md = compact_relevant_text(
                skill_md_full,
                (
                    "When To Use Do Not Use Failure Mechanism Procedure Invariants "
                    "High-Risk Operations Regression Risks activation boundary "
                    "forbidden over-trigger"
                ),
                max_chars=4200,
                context_lines=1,
                max_line_chars=700,
            )
            # Reuse tests for identical skill content (skip the verifier call).
            cache_key = hashlib.md5(skill_md_full.encode("utf-8")).hexdigest()
            cached = self._verifier_tests_cache.get(cache_key)
            if cached is not None:
                (skill_dir / "SKILL_TESTS.json").write_text(
                    json.dumps(cached, ensure_ascii=False, indent=2) + "\n"
                )
                total += len(cached)
                continue
            query = (
                f"Candidate skill to verify: {skill}\n\n"
                f"## Task domain (NO ground-truth answers are provided)\n"
                f"{self.task_constraints or '[general agent tasks]'}\n\n"
                f"## SKILL.md\n{skill_md}\n\n"
                + (
                    f"## Known gaming patterns from earlier iterations (probe these)\n{meta}\n\n"
                    if meta.strip()
                    else ""
                )
                + f"Author up to {max_tests} adversarial, behavior-based test cases "
                "(include at least one should_activate=false negative case). Do not "
                "assert specific answer values you cannot derive yourself."
            )
            try:
                trace = await self.agents.skill_verifier.run(query)
            except Exception as e:  # noqa: BLE001 - verifier is best-effort
                _log("", f"  [WARN] [verifier] failed for {skill}: {e}")
                continue
            self._add_iteration_cost(getattr(trace, "total_cost_usd", 0.0) or 0.0, "evolution")
            if trace.output is None or not trace.output.tests:
                continue
            tests = trace.output.tests[:max_tests]
            serializable = [t.model_dump() for t in tests]
            self._verifier_tests_cache[cache_key] = serializable
            (skill_dir / "SKILL_TESTS.json").write_text(
                json.dumps(serializable, ensure_ascii=False, indent=2) + "\n"
            )
            total += len(tests)
            _log(
                "",
                f"  -> [verifier] authored {len(tests)} independent test(s) for {skill} "
                f"(was author-written)",
            )
        if total:
            # Commit so the verifier tests persist on the child branch (an
            # uncommitted SKILL_TESTS.json would be stashed away on the next
            # branch switch, reverting the lineage to the author's tests).
            self.manager.commit(f"verifier tests x{total}")
        return total

    def _get_all_skill_tests_content(
        self,
        skill_names: list[str] | set[str] | None = None,
    ) -> str:
        """Read generated synthetic skill tests for judge-time validation."""
        skills_dir = self._project_root / ".claude" / "skills"
        if not skills_dir.exists():
            return "No skill tests available."
        allowed = {name for name in (skill_names or []) if name}
        parts = []
        for skill_dir in sorted(skills_dir.iterdir()):
            tests_file = skill_dir / "SKILL_TESTS.json"
            if skill_dir.is_dir() and tests_file.exists():
                if allowed and skill_dir.name not in allowed:
                    continue
                text = tests_file.read_text().strip()
                if text:
                    parts.append(f"### Tests for skill: {skill_dir.name}\n{text}")
        return "\n\n".join(parts) if parts else "No skill tests available."

    def _load_skill_test_specs(
        self,
        skill_names: list[str] | set[str] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Load generated synthetic skill tests from active project skills."""
        skills_dir = self._project_root / ".claude" / "skills"
        if not skills_dir.exists():
            return []
        allowed = {name for name in (skill_names or []) if name}
        # Collect tests grouped per skill so the global cap below can be applied
        # per-skill and round-robin, not as a flat truncation. A flat
        # ``specs[:max_cases]`` over a sorted, concatenated list silently tested
        # only the alphabetically-first skill whenever a bundle had more skills
        # than ``max_cases`` tests — every other skill in a distilled bundle went
        # completely untested, and one bad first skill failed the whole bundle.
        per_skill: list[tuple[str, list[dict[str, Any]]]] = []
        for skill_dir in sorted(skills_dir.iterdir()):
            tests_file = skill_dir / "SKILL_TESTS.json"
            if not skill_dir.is_dir() or not tests_file.exists():
                continue
            if allowed and skill_dir.name not in allowed:
                continue
            try:
                payload = json.loads(tests_file.read_text())
            except json.JSONDecodeError as exc:
                _log("WARN", f"  [SELF-TEST] Invalid {tests_file}: {exc}")
                continue
            if not isinstance(payload, list):
                _log("WARN", f"  [SELF-TEST] Ignoring non-list test payload: {tests_file}")
                continue
            items = [item for item in payload if isinstance(item, dict)]
            if items:
                per_skill.append((skill_dir.name, items))

        max_cases = max(0, int(self.config.executable_skill_test_max_cases))
        if not max_cases or not per_skill:
            return []

        # ``max_cases`` is a PER-SKILL budget: every skill in the bundle gets up
        # to ``max_cases`` of its own tests, so each skill is actually exercised.
        # Negative (should-not-activate) tests are kept first within a skill's
        # budget so the over-trigger hard gate always retains signal even if a
        # downstream caller trims the list.
        def _is_negative(item: dict[str, Any]) -> bool:
            return item.get("should_activate") is False

        capped: list[list[tuple[str, dict[str, Any]]]] = []
        for skill_name, items in per_skill:
            ordered = sorted(items, key=lambda it: not _is_negative(it))
            capped.append([(skill_name, it) for it in ordered[:max_cases]])

        # Round-robin interleave across skills so coverage stays balanced.
        specs: list[tuple[str, dict[str, Any]]] = []
        for col in range(max(len(c) for c in capped)):
            for skill_specs in capped:
                if col < len(skill_specs):
                    specs.append(skill_specs[col])
        return specs

    @staticmethod
    def _evidence_words(text: str) -> set[str]:
        import re

        stop = {
            "a", "an", "and", "are", "as", "be", "by", "for", "from", "in",
            "is", "it", "of", "on", "or", "the", "to", "with", "without",
            "must", "should", "would", "that", "this",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9_.%-]+", str(text).lower())
            if len(token) > 2 and token not in stop
        }

    @classmethod
    def _contains_evidence(cls, text: str, phrase: str) -> bool:
        """Loose evidence match for synthetic test rubrics."""
        haystack = str(text or "").lower()
        needle = str(phrase or "").strip().lower()
        if not needle:
            return True
        if needle in haystack:
            return cls._contains_non_negated_marker(haystack, needle)
        words = cls._evidence_words(needle)
        if not words:
            return True
        present = cls._evidence_words(haystack)
        required = 1.0 if len(words) <= 2 else 0.6
        return len(words & present) / max(1, len(words)) >= required

    @staticmethod
    def _contains_non_negated_marker(text: str, marker: str) -> bool:
        """Return whether marker appears outside a nearby negation context."""
        haystack = str(text or "").lower()
        marker = str(marker or "").lower()
        if not marker:
            return False
        start = 0
        negation_re = re.compile(
            r"(do\s+not|don't|does\s+not|doesn't|did\s+not|didn't|"
            r"not|no\s+need\s+to|without|avoid|should\s+not|must\s+not)\s*$"
        )
        while True:
            idx = haystack.find(marker, start)
            if idx < 0:
                return False
            prefix = haystack[max(0, idx - 48):idx]
            if not negation_re.search(prefix):
                return True
            start = idx + len(marker)

    @staticmethod
    def _skill_test_query(skill_name: str, spec: dict[str, Any]) -> str:
        """Turn one synthetic SkillTest spec into a small executable task."""
        must_check = spec.get("must_check") if isinstance(spec.get("must_check"), list) else []
        must_reject = spec.get("must_reject") if isinstance(spec.get("must_reject"), list) else []
        source_data = str(spec.get("source_data") or "").strip()
        task = str(spec.get("task") or "").strip()
        isolation = (
            "Isolation rule: this is a self-contained test. All authoritative "
            "data is in this prompt. Do not use Bash, Read, Glob, Grep, find, rg, "
            "or any filesystem search to inspect /tmp, /home, project data, "
            "historical runs, benchmark artifacts, or external files. If the "
            "prompt lacks data, say what is missing instead of searching."
        )
        if source_data or task:
            parts = [
                "Executable synthetic regression task. Solve it using only the data below; "
                "do not browse the web and do not inspect unrelated project data.",
                isolation,
                f"Test name: {spec.get('name', 'unnamed')}",
                f"Scenario: {spec.get('scenario', '')}",
            ]
            if source_data:
                parts.append(f"Source data:\n{source_data}")
            if task:
                parts.append(f"Task:\n{task}")
            if must_check:
                parts.append(f"Important checks to preserve: {json.dumps(must_check, ensure_ascii=False)}")
            if must_reject:
                parts.append(
                    f"Tempting wrong methods to avoid: {json.dumps(must_reject, ensure_ascii=False)}"
                )
            parts.append(
                "Return the final answer and only the brief work needed to justify it. "
                "Apply any available skill only when its own trigger conditions match."
            )
            return "\n\n".join(parts)
        return (
            "Synthetic skill regression test. Use only the scenario below; do not browse "
            "the web and do not inspect unrelated project data.\n\n"
            f"{isolation}\n\n"
            f"Potential available skill: {skill_name}\n"
            "Use or mention the potential skill only if its written trigger conditions apply.\n"
            f"Test name: {spec.get('name', 'unnamed')}\n"
            f"Scenario: {spec.get('scenario', '')}\n"
            f"Expected behavior: {spec.get('expected_behavior', '')}\n"
            f"Required checks: {json.dumps(must_check, ensure_ascii=False)}\n"
            f"Rejected operations/alternatives: {json.dumps(must_reject, ensure_ascii=False)}\n"
            f"Regression guard: {spec.get('regression_guard', '')}\n\n"
            "Respond with a concise verification note. If the skill should activate, "
            "apply it and explicitly name the checks performed and the rejected "
            "alternatives avoided. If it should not activate, keep the answer simple "
            "and explicitly state that the skill-specific contract is not needed."
        )

    def _make_isolated_self_test_agent(self) -> tuple[Agent[AgentResponse], Path]:
        """Create a base-like agent that can load skills but not project data.

        Executable skill tests are synthetic and self-contained. Running them in
        the real project root lets the agent inspect benchmark files, which
        turns the self-test into another OfficeQA run instead of an activation
        boundary check. The isolated root keeps only `.claude/skills` and exposes
        only the Skill tool.
        """
        sandbox = Path(tempfile.mkdtemp(prefix="evoskill-selftest-"))
        source_skills = self._project_root / ".claude" / "skills"
        dest_skills = sandbox / ".claude" / "skills"
        if source_skills.exists():
            shutil.copytree(source_skills, dest_skills, dirs_exist_ok=True)

        def options_factory() -> Any:
            options = self.agents.base._get_options()
            if isinstance(options, dict):
                updated = dict(options)
                updated["tools"] = {"Skill": {}}
                updated["working_directory"] = str(sandbox)
                updated["cwd"] = str(sandbox)
                updated.pop("data_dirs", None)
                updated.pop("add_dirs", None)
                return updated
            if hasattr(options, "allowed_tools"):
                options.allowed_tools = ["Skill"]
            if hasattr(options, "cwd"):
                options.cwd = str(sandbox)
            if hasattr(options, "working_directory"):
                options.working_directory = str(sandbox)
            if hasattr(options, "add_dirs"):
                options.add_dirs = []
            return options

        return (
            Agent(
                options_factory,
                AgentResponse,
                timeout_seconds=self.config.executable_skill_test_timeout_seconds,
                max_retries=2,
            ),
            sandbox,
        )

    def _score_executable_skill_test(
        self,
        *,
        skill_name: str,
        spec: dict[str, Any],
        trace: AgentTrace,
    ) -> ExecutableSkillTestCaseResult:
        from src.schemas.trajectory import StoredTrajectory

        test_name = str(spec.get("name") or "unnamed")
        should_activate = bool(spec.get("should_activate", False))
        answer = trace.output.final_answer if trace.output and trace.output.final_answer else str(trace.result or "")
        stored = StoredTrajectory.from_trace(
            trace=trace,
            question=self._skill_test_query(skill_name, spec),
            ground_truth=str(spec.get("expected_behavior", "")),
            category="synthetic_skill_test",
            agent_answer=answer,
        )
        transcript = "\n".join([answer, str(trace.result or ""), *stored.trace_messages])
        transcript_lower = transcript.lower()
        isolation_violations = (
            "/tmp/offskillevo",
            "/tmp/examples",
            "find /tmp",
            "rg -n -m",
            "rg --files /home",
            "rg --files /tmp",
            "rg --files /home/admin /tmp",
            "grep -r /tmp",
            "/home/admin/oskill",
            "/home/admin/evoskill/examples",
        )
        isolation_violation = next(
            (pattern for pattern in isolation_violations if pattern in transcript_lower),
            None,
        )
        current_root = str(getattr(self, "_project_root", Path.cwd())).lower()
        if isolation_violation is None and "/tmp/evoskill" in transcript_lower:
            for line in transcript_lower.splitlines():
                if (
                    "/tmp/evoskill" in line
                    and "/tmp/evoskill-selftest-" not in line
                    and current_root not in line
                ):
                    isolation_violation = "/tmp/evoskill outside current project root"
                    break
        # Path-only evidence is too noisy: the harness or agent can inspect
        # SKILL.md without actually applying the skill. Treat explicit reported
        # skill usage or a named contract/protocol marker as activation.
        activation_markers = (
            f"{skill_name} contract",
            f"{skill_name} protocol",
            f"used {skill_name}",
            f"using {skill_name}",
            f"activate {skill_name}",
            f"activated {skill_name}",
            f"`{skill_name}`",
        )
        activated = (
            skill_name in set(stored.skills_used)
            or any(
                self._contains_non_negated_marker(transcript_lower, marker)
                for marker in activation_markers
            )
        )
        must_check = spec.get("must_check") if isinstance(spec.get("must_check"), list) else []
        must_reject = spec.get("must_reject") if isinstance(spec.get("must_reject"), list) else []
        expected_intermediate = (
            spec.get("expected_intermediate")
            if isinstance(spec.get("expected_intermediate"), list)
            else []
        )
        forbidden_outputs = (
            spec.get("forbidden_outputs")
            if isinstance(spec.get("forbidden_outputs"), list)
            else []
        )
        expected_answer = str(spec.get("expected_answer") or "").strip()
        non_answer_phrases = (
            "no exact",
            "no numeric",
            "not asserted",
            "behavior",
            "direct lookup only",
            "derived statistic",
        )
        if any(phrase in expected_answer.lower() for phrase in non_answer_phrases):
            expected_answer = ""
        concrete_task = bool(str(spec.get("task") or "").strip() or str(spec.get("source_data") or "").strip())
        missing_checks = [
            str(item) for item in [*must_check, *expected_intermediate]
            if not self._contains_evidence(transcript, str(item))
        ]
        diagnostic_missing_checks: list[str] = []
        if concrete_task:
            # Concrete tests are behavior tests: final answer, activation
            # boundary, and explicit forbidden final answers are hard signals.
            # Rubric text is useful feedback for refinement, but should not
            # fail a solved mini-task merely because the agent used different
            # wording or omitted a verbose ledger.
            diagnostic_missing_checks = list(missing_checks)
            missing_checks = []

        final_answer_text = str(answer or "").strip().lower()
        forbidden_hits = [
            str(item) for item in forbidden_outputs
            if str(item).strip() and str(item).strip().lower() == final_answer_text
        ]
        missing_rejections: list[str] = []
        if not concrete_task:
            # Backward-compatible abstract tests still expect explicit mention
            # of avoided alternatives because there is no concrete answer to
            # distinguish the wrong method from the right one.
            missing_rejections = [
                str(item) for item in must_reject
                if not self._contains_evidence(transcript, str(item))
            ]

        answer_ok = True
        if expected_answer:
            # Transport timeouts and parse failures can legitimately leave a
            # self-test without an answer. Treat that as failed evidence rather
            # than allowing the reward scorer to abort the entire evolution run.
            answer_ok = bool(str(answer or "").strip()) and (
                score_answer(expected_answer, answer, 0.01) > 0.0
            )
            if not answer_ok:
                missing_checks.append(f"expected_answer={expected_answer}")

        protocol_markers = (
            "source_file:",
            "table_or_section:",
            "included_set",
            "included set",
            "formula_and_sign",
            "operation_chain:",
            "unit_and_denominator:",
            "source-binding manifest",
            "binding manifest",
        )
        if should_activate and not activated:
            marker_hits = sum(1 for marker in protocol_markers if marker in transcript_lower)
            if marker_hits >= 2:
                activated = True

        activation_ok = activated == should_activate
        negative_overtrigger = (not should_activate) and activated
        behavior_ok = (
            answer_ok
            and not isolation_violation
            and not missing_checks
            and not missing_rejections
            and not forbidden_hits
        )
        if concrete_task and expected_answer:
            # Executable self-tests are primarily behavior tests: did the
            # candidate skill actually repair the mini failure case? Positive
            # tests should not fail just because the agent solved the case
            # without explicitly reporting a skill activation. Likewise, a
            # negative case that still produces the correct behavior records
            # over-triggering as diagnostic feedback instead of discarding an
            # otherwise effective child.
            passed = (not trace.is_error) and behavior_ok
        else:
            passed = (not trace.is_error) and activation_ok and behavior_ok
        reasons: list[str] = []
        if trace.is_error:
            reasons.append(f"agent error: {trace.parse_error or 'unknown'}")
        if isolation_violation:
            reasons.append(f"self-test isolation violated: used {isolation_violation}")
        if concrete_task and expected_answer and negative_overtrigger and behavior_ok:
            reasons.append(
                "concrete task solved; activation observed as diagnostic over-trigger"
            )
        elif not activation_ok and not (
            concrete_task and expected_answer and should_activate and not activated
        ):
            reasons.append(
                f"activation mismatch: expected {should_activate}, observed {activated}"
            )
        elif concrete_task and expected_answer and should_activate and not activated:
            reasons.append(
                "skill activation not observed, but concrete task answer is the primary signal"
            )
        if missing_checks:
            reasons.append("missing required checks")
        if missing_rejections:
            reasons.append("missing rejected alternatives")
        if forbidden_hits:
            reasons.append("forbidden output/method appeared")
            missing_rejections.extend(forbidden_hits)
        if expected_answer and not answer_ok:
            reasons.append("expected answer not produced")
        if not reasons:
            reasons.append("activation and rubric evidence matched")
        return ExecutableSkillTestCaseResult(
            skill_name=skill_name,
            test_name=test_name,
            should_activate=should_activate,
            activated=activated,
            passed=passed,
            reason="; ".join(reasons),
            agent_answer=answer,
            missing_checks=missing_checks or diagnostic_missing_checks,
            missing_rejections=missing_rejections,
        )

    async def _run_executable_skill_tests(
        self,
        skill_names: list[str] | set[str] | None = None,
    ) -> ExecutableSkillTestSuiteResult:
        """Run generated SKILL_TESTS.json as real synthetic agent tasks."""
        if not self.config.executable_skill_tests:
            return ExecutableSkillTestSuiteResult(0, 0, 0.0)
        specs = self._load_skill_test_specs(skill_names)
        if not specs:
            return ExecutableSkillTestSuiteResult(0, 0, 0.0)

        cache_key = json.dumps(
            {
                "skills": self._get_all_skills_content(),
                "target_skills": sorted({name for name in (skill_names or []) if name}),
                "tests": specs,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        cached = getattr(self, "_executable_skill_test_cache", {}).get(cache_key)
        if cached is not None:
            return cached

        _log("SELF-TEST", f"Running {len(specs)} executable skill test(s)...")
        test_concurrency = max(1, self.config.executable_skill_test_concurrency)
        if is_claude_sdk():
            # Claude Code backed DeepSeek runs are prone to exit -9 when several
            # synthetic self-tests spawn at once.
            test_concurrency = 1
        semaphore = asyncio.Semaphore(test_concurrency)

        async def run_one(skill_name: str, spec: dict[str, Any]) -> ExecutableSkillTestCaseResult:
            async with semaphore:
                # A timed-out SDK process may clean up or retain its working
                # directory asynchronously. Isolate every case so one failed
                # process cannot invalidate another case's cwd.
                self_test_agent, self_test_sandbox = self._make_isolated_self_test_agent()
                query = self._skill_test_query(skill_name, spec)
                try:
                    try:
                        async with asyncio.timeout(
                            self.config.executable_skill_test_timeout_seconds
                        ):
                            trace = await self_test_agent.run(query)
                        self._add_iteration_cost(trace.total_cost_usd, "self_test")
                    except Exception as exc:  # noqa: BLE001 - harness transports vary
                        trace = AgentTrace(
                            duration_ms=0,
                            total_cost_usd=0.0,
                            num_turns=0,
                            usage={},
                            result="",
                            is_error=True,
                            parse_error=f"{type(exc).__name__}: {exc}",
                            messages=[],
                        )
                    return self._score_executable_skill_test(
                        skill_name=skill_name,
                        spec=spec,
                        trace=trace,
                    )
                finally:
                    shutil.rmtree(self_test_sandbox, ignore_errors=True)

        cases = await asyncio.gather(*(run_one(skill, spec) for skill, spec in specs))
        passed = sum(1 for case in cases if case.passed)
        result = ExecutableSkillTestSuiteResult(
            passed=passed,
            total=len(cases),
            pass_rate=passed / len(cases) if cases else 0.0,
            cases=list(cases),
        )
        self._executable_skill_test_cache[cache_key] = result
        _log(
            "SELF-TEST",
            f"Executable skill tests: {result.passed}/{result.total} pass_rate={result.pass_rate:.3f}",
        )
        for case in result.cases:
            status = "PASS" if case.passed else "FAIL"
            _log(
                "",
                (
                    f"  [{status}] {case.skill_name}/{case.test_name}: "
                    f"expected_activate={case.should_activate} activated={case.activated}; "
                    f"{case.reason}"
                ),
            )
        return result

    @staticmethod
    def _has_failed_negative_self_test(
        result: ExecutableSkillTestSuiteResult | None,
    ) -> bool:
        """Return whether a verifier-authored no-activation case over-triggered.

        A negative test can fail for weak rubric wording or missing expected
        answer. Those are useful refinement feedback, but only observed
        activation on a should-not-activate case is a hard regression signal.
        """
        if result is None:
            return False
        return any(
            not case.should_activate and case.activated and not case.passed
            for case in result.cases
        )

    @staticmethod
    def _skills_failing_negative_tests(
        result: ExecutableSkillTestSuiteResult | None,
    ) -> set[str]:
        """Skills that over-triggered on a should-not-activate case.

        Mirrors ``_has_failed_negative_self_test`` but reports *which* skills are
        responsible, so a distilled multi-skill bundle can prune just the
        offenders instead of being rejected wholesale.
        """
        if result is None:
            return set()
        return {
            case.skill_name
            for case in result.cases
            if not case.should_activate
            and case.activated
            and not case.passed
            and case.skill_name
        }

    def _remove_skill_dirs(self, names: set[str] | list[str]) -> None:
        """Delete the SKILL.md directories for ``names`` from the active project."""
        skills_dir = self._project_root / ".claude" / "skills"
        for name in names:
            target = skills_dir / name
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)

    def _get_program_skills_content(self, program_name: str, restore_to: str | None = None) -> str:
        """Read active skills for a stored program and optionally restore another program."""
        try:
            self.manager.switch_to(program_name)
            return self._get_all_skills_content()
        finally:
            if restore_to is not None:
                self.manager.switch_to(restore_to)

    async def _distill_frontier_skill(
        self,
        *,
        frontier: list[tuple[str, float]],
        judge_provider: str,
        judge_model: str,
        failure_cases: list[tuple[Any, ...]],
        regression_cases: list[tuple[Any, ...]],
    ) -> tuple[str, float] | None:
        """Distill frontier skill variants into one validated final program."""
        if (
            not self.config.distill_frontier_at_end
            or self.config.evolution_mode != "skill_only"
            or not frontier
        ):
            return None

        champion, champion_score = frontier[0]

        distilled_name = "frontier-distilled"
        if distilled_name in self.manager.list_programs():
            suffix = 2
            while f"{distilled_name}-{suffix}" in self.manager.list_programs():
                suffix += 1
            distilled_name = f"{distilled_name}-{suffix}"

        # Retain the champion's FULL skill set as the distilled bundle, then prune
        # only over-triggering skills below. We deliberately do NOT regenerate the
        # bundle through the generator: its single-skill ToolGeneratorResponse
        # schema cannot emit a multi-skill array, so the LLM "merge" collapsed an
        # N-skill champion into one skill and the preceding rmtree threw the rest
        # away. Starting from the champion's own skills guarantees the aggregate
        # survives; per-skill pruning then removes only the harmful ones.
        _log(
            "DISTILL",
            f"Distilling champion {champion} into {distilled_name} (retain + prune)...",
        )
        self.manager.switch_to(champion)
        champion_config = self.manager.get_current()
        self.manager.create_program(
            distilled_name,
            champion_config.mutate(distilled_name),
            parent=champion,
        )
        self.manager.switch_to(distilled_name)
        skill_names = self._get_active_skills()
        if not skill_names:
            _log("WARN", f"  {distilled_name}: champion has no active skills to distill")
            self.manager.discard(distilled_name)
            self.manager.switch_to(champion)
            return None
        _log(
            "DISTILL",
            f"  Retained {len(skill_names)} champion skill(s): {', '.join(skill_names)}",
        )
        self.manager.commit(f"{distilled_name}: retained champion skill bundle")

        await self._author_verifier_tests(skill_names)
        tests = await self._run_executable_skill_tests(skill_names)

        # Per-skill pruning instead of all-or-nothing rejection: an N-skill bundle
        # must not be discarded wholesale because one skill over-triggers. Drop
        # only the skills whose own negative/over-trigger tests failed, keep the
        # rest, and re-validate the survivors once. This is what preserves the
        # aggregate of useful distilled mechanisms.
        bad_skills = self._skills_failing_negative_tests(tests)
        if bad_skills and len(bad_skills) < len(skill_names):
            survivors = [s for s in skill_names if s not in bad_skills]
            _log(
                "DISTILL",
                (
                    f"Pruning {len(bad_skills)} over-triggering skill(s) from "
                    f"{distilled_name}: {', '.join(sorted(bad_skills))}; "
                    f"keeping {len(survivors)} ({', '.join(survivors)})"
                ),
            )
            self._remove_skill_dirs(bad_skills)
            self.manager.commit(
                f"{distilled_name}: pruned {len(bad_skills)} over-triggering skill(s)"
            )
            skill_names = survivors
            tests = await self._run_executable_skill_tests(skill_names)

        if (
            not skill_names
            or self._has_failed_negative_self_test(tests)
            or (tests.total > 0 and tests.pass_rate < self.config.refine_self_test_threshold)
        ):
            _log(
                "DISTILL",
                (
                    f"Rejected {distilled_name}: executable self-tests "
                    f"{tests.passed}/{tests.total}"
                    + ("" if skill_names else " (no skills survived pruning)")
                ),
            )
            self.manager.discard(distilled_name)
            self.manager.switch_to(champion)
            return None

        parent_content = self._get_program_skills_content(champion, restore_to=distilled_name)
        candidate_content = self._get_all_skills_content()
        scoring_cases = [
            *failure_cases[: max(0, int(self.config.frontier_distill_judge_failures))],
            *regression_cases[: max(0, int(self.config.frontier_distill_judge_regressions))],
        ]
        if not scoring_cases:
            _log("DISTILL", f"Rejected {distilled_name}: no scoring cases for independent evaluation")
            self.manager.discard(distilled_name)
            self.manager.switch_to(champion)
            return None
        try:
            result = await self._judge_skill_with_llm(
                scoring_cases,
                judge_provider,
                judge_model,
                child_name=distilled_name,
                parent_name=champion,
                parent_skill_summary=self._summarize_skill_content(parent_content),
                candidate_skill_summary=self._summarize_skill_content(candidate_content),
                skill_diff=self._diff_skill_content(parent_content, candidate_content),
                skill_diff_swapped=self._diff_skill_content(candidate_content, parent_content),
                candidate_skill_tests=self._get_all_skill_tests_content(skill_names),
                executable_skill_test_result=tests,
                proposer_root_cause="Distill compatible, non-regressive mechanisms from frontier.",
            )
        except RuntimeError as exc:
            _log("DISTILL", f"Rejected {distilled_name}: judge failed ({exc})")
            self.manager.discard(distilled_name)
            self.manager.switch_to(champion)
            return None
        if result.average_match_score < 0.5:
            _log(
                "DISTILL",
                f"Rejected {distilled_name}: lost to champion "
                f"(avg_match={result.average_match_score:.3f})",
            )
            self.manager.discard(distilled_name)
            self.manager.switch_to(champion)
            return None

        # Do not inherit the champion score. Map the independently judged
        # champion duel into the same score space: a draw keeps the champion
        # score, a win can move upward, and a loss was rejected above.
        margin = max(-1.0, min(1.0, 2.0 * (result.average_match_score - 0.5)))
        if margin >= 0.0:
            distilled_score = champion_score + margin * (1.0 - champion_score)
        else:
            distilled_score = champion_score + margin * champion_score
        distilled_score = max(0.0, min(1.0, distilled_score))

        self.manager.update_frontier(
            distilled_name,
            distilled_score,
            max_size=max(self.config.frontier_size, len(frontier) + 1),
        )
        self.manager.switch_to(distilled_name)
        _log(
            "DISTILL",
            f"Accepted {distilled_name} as final output "
            f"(judge_score={distilled_score:.4f}, champion={champion}:{champion_score:.4f}, "
            f"avg_match={result.average_match_score:.3f})",
        )
        return distilled_name, distilled_score

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

        # When the evolution root is a swept seed (not bare base), base is only
        # the hidden score origin and must never be a head-to-head opponent.
        # When there was no sweep (root == "base"), base remains a valid anchor
        # for backward compatibility.
        root_name = getattr(self, "_evolution_root", "base")
        existing = set(self.manager.list_programs())
        anchors: list[str] = []
        for name in candidates:
            if name == "base" and root_name != "base":
                continue
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
    def _build_child_diversity_hint(
        child_idx: int,
        sibling_proposals: list[str],
        existing_children: list[ProgramSearchNode] | None = None,
    ) -> str:
        """Build proposer guidance so a parent's children explore different fixes.

        ``existing_children`` makes the constraint persist across iterations. This
        matters when ``children_per_iteration=1``: without it, every expansion is
        treated as the first sibling and repeatedly prefers editing the same skill.
        """
        existing_children = existing_children or []
        prior_actions = [child.proposal_action for child in existing_children if child.proposal_action]
        prior_skills = [child.proposal_skill for child in existing_children if child.proposal_skill]
        effective_idx = len(existing_children) + child_idx
        strategies = [
            (
                "For this child, CREATE a narrow mechanism-specific skill unless an "
                "existing skill has the same earliest divergent step. Do not broaden "
                "a general skill to absorb a different failure mechanism."
            ),
            (
                "For this child, propose an alternative root-cause hypothesis. "
                "Prefer CREATE and do not repeat an earlier child's target skill, "
                "checks, or workflow unless unavoidable."
            ),
            (
                "For this child, target a different intervention surface: source binding, "
                "formula/statistic definition, unit/representation, or output extraction. "
                "CREATE a separate specialized skill when the mechanism differs."
            ),
            (
                "For this child, deliberately choose a different intervention type "
                "from earlier siblings, with different trigger conditions and verification steps."
            ),
        ]
        base_hint = strategies[min(effective_idx, len(strategies) - 1)]
        all_prior = [child.proposal for child in existing_children if child.proposal] + sibling_proposals
        context: list[str] = [base_hint]
        if prior_actions:
            context.append(f"Earlier child actions: {', '.join(prior_actions)}.")
        if prior_skills:
            context.append(f"Earlier child target skills: {', '.join(prior_skills)}.")
        if not all_prior:
            return "\n".join(context)

        prior = "\n".join(f"- {proposal[:240]}" for proposal in all_prior)
        context.append(
            "Already proposed children from this parent:\n"
            f"{prior}\n"
            "Your proposal must be materially different in target, root cause, or verification method."
        )
        return "\n\n".join(context)

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

    def _sample_induction_successes(
        self,
        successes: list[tuple[Any, ...]],
        failures: list[tuple[Any, ...]],
        count: int,
    ) -> list[tuple[Any, ...]]:
        """Sample successful trajectories to distill a skill from (skill induction).

        Unlike _sample_regression_successes (which feeds the judge anti-regression
        cases), these go to the PROPOSER as positive examples. When
        success_induction_prefer_hard is on, successes are drawn from the hardest
        categories first — a success on a mostly-failing category is the most
        valuable to generalize. A rotating offset varies the sample across calls.
        """
        if not successes or count <= 0:
            return []

        succ_cats = {entry[4] for entry in successes}
        if self.config.success_induction_prefer_hard:
            fail_ct: dict[str, int] = {}
            succ_ct: dict[str, int] = {}
            for f in failures:
                fail_ct[f[4]] = fail_ct.get(f[4], 0) + 1
            for s in successes:
                succ_ct[s[4]] = succ_ct.get(s[4], 0) + 1

            def hardness(cat: str) -> float:
                f = fail_ct.get(cat, 0)
                s = succ_ct.get(cat, 0)
                return f / (f + s) if (f + s) else 0.0

            ordered_cats = sorted(succ_cats, key=lambda c: (hardness(c), c), reverse=True)
        else:
            ordered_cats = sorted(succ_cats)

        # Flatten hardest-category-first, then rotate so successive inductions vary.
        ordered: list[tuple[Any, ...]] = []
        for cat in ordered_cats:
            ordered.extend(entry for entry in successes if entry[4] == cat)
        if not ordered:
            return []
        start = self._induction_offset % len(ordered)
        rotated = ordered[start:] + ordered[:start]
        self._induction_offset += count
        return rotated[:count]

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

    def _summarize_trace_for_skill_context(
        self,
        trace: AgentTrace,
        *,
        relevance_query: str,
        max_chars: int = 1800,
    ) -> str:
        """Keep trajectory lines relevant to the current skill/proposal only."""
        raw = trace.summarize(head_chars=1200, tail_chars=800)
        return compact_relevant_text(
            raw,
            relevance_query,
            max_chars=max_chars,
            context_lines=1,
            max_line_chars=700,
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

    async def _call_judge_api(
        self,
        prompt: str,
        provider: str,
        model: str,
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        """Make a direct (non-agent) completion call to the judge model."""
        if provider == "codex":
            from src.harness.codex import executor as codex_executor

            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "root_cause": {"type": "string"},
                    "case_relevance": {"type": "number"},
                    "proposer_root_cause_correct": {"type": "number"},
                    "skill_addresses_root_cause": {"type": "number"},
                    "failure_mechanism_encoding": {"type": "number"},
                    "executable_specificity": {"type": "number"},
                    "high_risk_blacklist": {"type": "number"},
                    "generalization_transfer": {"type": "number"},
                    "self_test_pass_rate": {"type": "number"},
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
                    "case_relevance",
                    "proposer_root_cause_correct",
                    "skill_addresses_root_cause",
                    "failure_mechanism_encoding",
                    "executable_specificity",
                    "high_risk_blacklist",
                    "generalization_transfer",
                    "self_test_pass_rate",
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
        # Safe even when self.config is absent (some unit tests construct the loop
        # via object.__new__): default to the LoopConfig judge budget.
        max_out = int(
            max_output_tokens
            or getattr(getattr(self, "config", None), "judge_max_output_tokens", 1536)
            or 1536
        )
        if provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=api_key)
            messages = [{"role": "user", "content": prompt}]
            system = self._judge_json_only_system_prompt(provider, model)
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": self._judge_max_output_tokens(provider, model, max_out),
                "system": system,
                "messages": messages,
            }
            resp = await client.messages.create(**kwargs)
            self._record_judge_api_usage(provider, model, getattr(resp, "usage", None))
            text = self._extract_anthropic_message_text(resp)
            if text:
                return text

            summary = self._summarize_anthropic_message_blocks(resp)
            if self._anthropic_message_has_only_thinking(resp):
                retry_prompt = self._build_json_only_retry_prompt(prompt)
                _log(
                    "WARN",
                    "  Anthropic judge returned only thinking blocks; retrying once "
                    "with compact JSON-only instruction",
                )
                retry_kwargs = dict(kwargs)
                retry_kwargs["max_tokens"] = self._judge_retry_max_output_tokens(
                    provider,
                    model,
                    max_out,
                )
                retry_kwargs["messages"] = [{"role": "user", "content": retry_prompt}]
                retry_resp = await client.messages.create(**retry_kwargs)
                self._record_judge_api_usage(provider, model, getattr(retry_resp, "usage", None))
                retry_text = self._extract_anthropic_message_text(retry_resp)
                if retry_text:
                    return retry_text
                summary = (
                    f"{summary}; retry={self._summarize_anthropic_message_blocks(retry_resp)}"
                )

            raise RuntimeError(
                "Anthropic judge returned no text content "
                f"({summary})"
            )
        if provider == "openai":
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if model.startswith(("gpt-5", "o1", "o3", "o4")):
                kwargs["max_completion_tokens"] = max_out
            else:
                kwargs["max_tokens"] = max_out
            resp = await client.chat.completions.create(**kwargs)
            self._record_judge_api_usage(provider, model, getattr(resp, "usage", None))
            return resp.choices[0].message.content or ""
        raise ValueError(f"Judge does not support provider: {provider!r}")

    @staticmethod
    def _is_deepseek_anthropic_model(provider: str, model: str) -> bool:
        return provider == "anthropic" and model.lower().startswith("deepseek")

    @staticmethod
    def _judge_json_only_system_prompt(provider: str, model: str) -> str:
        if SelfImprovingLoop._is_deepseek_anthropic_model(provider, model):
            return (
                "You are a strict JSON judge. You may think internally, but the "
                "visible answer must be exactly one valid JSON object matching "
                "the requested fields. Do not output markdown or prose."
            )
        return (
            "You are a strict JSON judge. Return exactly one valid JSON object "
            "matching the requested fields. Do not output markdown or prose."
        )

    @staticmethod
    def _judge_max_output_tokens(provider: str, model: str, configured: int) -> int:
        if SelfImprovingLoop._is_deepseek_anthropic_model(provider, model):
            return max(configured, 8192)
        return configured

    @staticmethod
    def _judge_retry_max_output_tokens(provider: str, model: str, configured: int) -> int:
        if SelfImprovingLoop._is_deepseek_anthropic_model(provider, model):
            return max(configured, 12000)
        return configured

    @staticmethod
    def _build_json_only_retry_prompt(prompt: str) -> str:
        value = str(prompt or "")
        required = (
            'Return JSON only with keys: "root_cause", "case_relevance", '
            '"proposer_root_cause_correct", "skill_addresses_root_cause", '
            '"failure_mechanism_encoding", "executable_specificity", '
            '"high_risk_blacklist", "generalization_transfer", '
            '"self_test_pass_rate", "set_a_success_prob", '
            '"set_b_success_prob", "b_over_a_score", "confidence", '
            '"remaining_blockers", "reasoning". Values must be numbers in '
            '[0,1] where applicable. No markdown. No prose outside JSON.'
        )
        tail = value[-5000:] if len(value) > 5000 else value
        return f"{required}\n\nRelevant original judge request tail:\n{tail}"

    @staticmethod
    def _extract_anthropic_message_text(response: Any) -> str:
        """Extract text from Anthropic-compatible message content blocks.

        DeepSeek's Anthropic-compatible endpoint can return thinking blocks
        before text blocks. Those blocks do not expose ``.text``; judge parsing
        should ignore them instead of failing the whole comparison.
        """
        parts: list[str] = []
        for block in getattr(response, "content", None) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text)
                continue

            payload = None
            for method_name in ("model_dump", "to_dict", "dict"):
                method = getattr(block, method_name, None)
                if callable(method):
                    try:
                        payload = method()
                        break
                    except Exception:
                        pass
            if isinstance(payload, dict):
                text = payload.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)

        return "\n".join(parts).strip()

    @staticmethod
    def _summarize_anthropic_message_blocks(response: Any) -> str:
        """Return safe metadata for Anthropic-compatible content blocks."""
        blocks = getattr(response, "content", None) or []
        block_types: list[str] = []
        for block in blocks:
            block_type = getattr(block, "type", None)
            if not block_type:
                block_type = type(block).__name__
            block_types.append(str(block_type))
        stop_reason = getattr(response, "stop_reason", None)
        usage = getattr(response, "usage", None)
        return (
            f"stop_reason={stop_reason!r}, "
            f"block_types={block_types!r}, "
            f"usage={usage!r}"
        )

    @staticmethod
    def _anthropic_message_has_only_thinking(response: Any) -> bool:
        """Return True when all non-empty content blocks are thinking blocks."""
        blocks = list(getattr(response, "content", None) or [])
        if not blocks:
            return False
        block_types: list[str] = []
        for block in blocks:
            block_type = getattr(block, "type", None)
            if not block_type:
                payload = None
                for method_name in ("model_dump", "to_dict", "dict"):
                    method = getattr(block, method_name, None)
                    if callable(method):
                        try:
                            payload = method()
                            break
                        except Exception:
                            pass
                if isinstance(payload, dict):
                    block_type = payload.get("type")
            block_types.append(str(block_type or type(block).__name__))
        return bool(block_types) and all(t == "thinking" for t in block_types)

    @staticmethod
    def _parse_judge_json_text(text: str) -> dict[str, Any]:
        """Parse judge JSON, tolerating fences and short wrapper text."""
        value = str(text or "").strip()
        if value.startswith("```"):
            value = "\n".join(value.split("\n")[1:])
            value = value.rsplit("```", 1)[0].strip()
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            start = value.find("{")
            end = value.rfind("}")
            if start >= 0 and end > start:
                return json.loads(value[start : end + 1])
            raise

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
                weight = max(0.0, float(getattr(match, "weight", 1.0)))
                if weight <= 0.0:
                    continue
                diff = theta[a] - theta[b]
                # Stable sigmoid.
                if diff >= 0:
                    exp_neg = math.exp(-diff)
                    p = 1.0 / (1.0 + exp_neg)
                else:
                    exp_pos = math.exp(diff)
                    p = exp_pos / (1.0 + exp_pos)
                residual = weight * (y - p)
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
        self_tests = SelfImprovingLoop._clamp01(data.get("self_test_pass_rate"), default=0.0)
        return max(
            0.0,
            min(
                1.0,
                0.30 * mechanism
                + 0.25 * executable
                + 0.18 * blacklist
                + 0.15 * transfer
                + 0.12 * self_tests,
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
    def _judge_case_relevance(data: dict[str, Any]) -> float:
        """Estimate whether a judged failure belongs to the candidate's target mechanism.

        Older judge outputs do not include ``case_relevance``. In that case,
        approximate relevance from the root-cause and mechanism fields so
        unrelated held-out failures become neutral evidence instead of negative
        evidence.
        """
        if "case_relevance" in data:
            return SelfImprovingLoop._clamp01(data.get("case_relevance"), default=0.5)
        return max(
            SelfImprovingLoop._clamp01(data.get("skill_addresses_root_cause"), default=0.0),
            SelfImprovingLoop._clamp01(data.get("proposer_root_cause_correct"), default=0.0),
            0.5
            * SelfImprovingLoop._clamp01(data.get("failure_mechanism_encoding"), default=0.0),
        )

    @staticmethod
    def _apply_unrelated_failure_match_policy(
        match_score: float,
        case_relevance: float,
    ) -> float:
        """Make unrelated failure mechanisms neutral pairwise evidence.

        A child skill is generated from a localized failure mechanism. Held-out
        failures from other mechanisms should surface blockers for later nodes,
        but they should not push the current node below its parent merely because
        it does not solve a mechanism it never targeted.
        """
        score = max(0.0, min(1.0, match_score))
        relevance = max(0.0, min(1.0, case_relevance))
        if relevance < 0.30:
            return 0.5
        if relevance < 0.55:
            weight = (relevance - 0.30) / 0.25
            return 0.5 + (score - 0.5) * weight
        return score

    @staticmethod
    def _apply_regression_match_policy(match_score: float, penalty_weight: float) -> float:
        """Make previously-correct cases preservation-only pairwise evidence."""
        score = max(0.0, min(1.0, match_score))
        # Regression cases are anti-regression checks, not opportunities for
        # positive credit. Also avoid amplifying weak speculative "overhead"
        # concerns: only concrete below-draw evidence should penalize the child.
        if score >= 0.45:
            return 0.5
        weight = max(1.0, float(penalty_weight))
        return max(0.0, 0.5 - (0.5 - score) * weight)

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

    def _proposal_policy_prior(
        self,
        proposer_confidence: float | None = None,
        *,
        proposal: str = "",
        existing_skill_content: str = "",
        sibling_proposals: list[str] | None = None,
    ) -> float:
        """Return a judge-independent policy prior for PUCT expansion.

        Keep this separate from judge value to avoid double-counting judge
        scores in both Q and exploration. Mechanism-overlapping proposals receive
        a lower prior so PUCT spends exploration budget on genuinely different
        interventions instead of repeatedly rewriting the same broad skill.
        """
        if proposer_confidence is None:
            proposer_confidence = self.config.puct_default_prior
        prior = self._clamp01(proposer_confidence, default=self.config.puct_default_prior)
        if not proposal.strip():
            return prior

        candidates = [existing_skill_content, *(sibling_proposals or [])]
        max_similarity = max(
            (self._text_overlap_similarity(proposal, text) for text in candidates if text.strip()),
            default=0.0,
        )
        threshold = max(0.0, min(1.0, self.config.puct_overlap_similarity_threshold))
        if max_similarity <= threshold:
            return prior
        penalty = max(0.0, min(1.0, self.config.puct_overlap_prior_penalty))
        scaled_overlap = (max_similarity - threshold) / max(1e-9, 1.0 - threshold)
        return max(0.01, prior * (1.0 - penalty * scaled_overlap))

    @classmethod
    def _text_overlap_similarity(cls, a: str, b: str) -> float:
        """Deterministic Jaccard overlap for proposal/skill mechanism text."""
        ta = cls._evidence_words(a)
        tb = cls._evidence_words(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _build_program_search_tree(
        self, root_score: float, root_name: str = "base"
    ) -> ProgramSearchNode:
        """Build an in-memory PUCT tree seeded from the current frontier.

        ``root_name`` is the evolution root — ``"base"`` normally, or the seed
        program produced by the pre-PUCT coverage sweep.
        """
        root = ProgramSearchNode(
            name=root_name,
            parent=None,
            prior=1.0,  # policy prior for root: maximum, decoupled from value (score)
            score=root_score,
            visit_count=1,
            total_q=root_score,
            depth=0,
        )

        for name, score in self.manager.get_frontier_with_scores():
            if name == root_name:
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

    async def _sweep_failure_coverage(
        self,
        all_failures_ext: list[tuple[Any, ...]],
        all_successes_ext: list[tuple[Any, ...]] | None = None,
    ) -> str:
        """Generate a skill for every failure cluster before PUCT search.

        Groups all collected failures by failure_type and batches each group by
        ``failure_sample_count``. For each batch it runs the proposer/generator
        (``_mutate_with_fallback``) off the current seed program, authors and
        runs the new skill's executable self-tests, and accepts the skill into
        the growing seed iff it does not over-trigger (no failed negative test).
        Every failure batch is processed, so all failure tasks enter a proposer
        batch — full coverage instead of the fraction the PUCT tree reaches.

        Returns the final seed program name (``"base"`` if nothing was accepted
        or the sweep is disabled), which becomes the PUCT root and BT anchor.
        """
        if not self.config.sweep_failures_before_puct or not all_failures_ext:
            return "base"

        # Group by failure_type (fallback to category) for homogeneous batches.
        groups: dict[str, list[tuple[Any, ...]]] = {}
        for entry in all_failures_ext:
            key = self._trajectory_failure_type(entry) or str(entry[4]) or "unclassified"
            groups.setdefault(key, []).append(entry)

        batch_size = max(1, self.config.failure_sample_count)
        seed = "base"
        accepted = 0
        batch_idx = 0
        # skill_name -> the failure questions that skill was created/edited to fix.
        # Used by post-sweep trigger-fit calibration as each skill's recall probes.
        target_map: dict[str, list[str]] = {}
        total_batches = sum(
            math.ceil(len(items) / batch_size) for items in groups.values()
        )

        # Let the sweep create a dedicated skill per distinct mechanism instead of
        # folding later mechanisms into whatever skills happened to fill the budget
        # first (arrival-order starvation: sign/unit, sorted last, always got folded
        # into a mega-skill whose abstract description then never fired). The budget
        # is re-imposed AFTER the sweep by trigger-fit calibration, which keeps the
        # best-firing max_active_skills and evicts the rest by measured value.
        _saved_budget = self.config.max_active_skills
        if self.config.calibrate_trigger_fit:
            self.config.max_active_skills = 0
        _log(
            "SWEEP",
            (
                f"Coverage sweep over {len(all_failures_ext)} failures in "
                f"{len(groups)} failure_type group(s) → {total_batches} batch(es), "
                f"batch_size={batch_size}"
            ),
        )

        for ftype in sorted(groups):
            items = groups[ftype]
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                batch_idx += 1
                # Mark every failure in the chunk as shown (coverage bookkeeping).
                for entry in chunk:
                    key = str(entry[1])
                    self._failure_shown_count[key] = (
                        self._failure_shown_count.get(key, 0) + 1
                    )

                batch_failures: list[tuple[Any, ...]] = []
                questions: list[str] = []
                for entry in chunk:
                    trace, question, agent_answer, ground_truth, category = entry[:5]
                    feedback = self._trajectory_failure_feedback(entry)
                    if not feedback:
                        feedback = build_answer_comparison_feedback(
                            question,
                            agent_answer,
                            ground_truth,
                            self._trajectory_failure_type(entry) or category,
                            domain_hints=self.config.error_surface_hints,
                        )
                    batch_failures.append(
                        (trace, agent_answer, ground_truth, category, feedback)
                    )
                    questions.append(question)

                _log(
                    "SWEEP",
                    (
                        f"Batch {batch_idx}/{total_batches} [{ftype}] "
                        f"{len(chunk)} failure(s); seed={seed}"
                    ),
                )
                self._iter_cost = 0.0
                mutation_result = await self._mutate_with_fallback(
                    seed,
                    batch_failures,
                    f"sweep-{batch_idx}",
                    questions=questions,
                )
                self._total_cost += self._iter_cost
                if mutation_result is None:
                    _log("SWEEP", "  proposer/generator failed; seed unchanged")
                    continue

                child_name = mutation_result.child_name
                self.manager.switch_to(child_name)

                new_skills = [
                    a.get("skill_name")
                    for a in mutation_result.applied_skills
                    if a.get("skill_name")
                ] or [mutation_result.skill_name or ""]
                new_skills = [s for s in new_skills if s]

                # Optional over-trigger gate. When disabled (default) the sweep is
                # a pure generation phase: every successfully generated skill is
                # accepted into the seed and quality is deferred to PUCT + frontier
                # distillation. The gate is the single most expensive sweep step
                # (verifier authoring runs on the slow evolution model), so it is
                # off by default to keep the sweep fast.
                if self.config.sweep_self_test_gate:
                    self._iter_cost = 0.0
                    if new_skills:
                        await self._author_verifier_tests(new_skills)
                        tests = await self._run_executable_skill_tests(new_skills)
                    else:
                        tests = None
                    self._total_cost += self._iter_cost

                    if self._has_failed_negative_self_test(tests):
                        _log(
                            "SWEEP",
                            f"  [REJECT] {child_name}: over-triggers (negative self-test failed)",
                        )
                        self.manager.discard(child_name)
                        self.manager.switch_to(seed)
                        continue

                # Record which questions each skill is meant to fix; these become its
                # recall probes during calibration. Only attribute a skill to the batch
                # that FIRST produced it (its creation): a later batch editing the same
                # skill must not pollute its target set with that batch's unrelated
                # questions, which previously dragged the tuned description off-topic.
                for sk in new_skills:
                    if sk not in target_map:
                        target_map[sk] = list(questions)

                seed = child_name
                accepted += 1
                n_skills = len(self._get_active_skills())
                _log(
                    "SWEEP",
                    f"  [ACCEPT] seed={seed} (skills={n_skills}, accepted={accepted})",
                )

        # Re-impose the active-skill budget; the sweep ran unlimited so every
        # distinct mechanism could be created without being folded.
        self.config.max_active_skills = _saved_budget

        _log(
            "SWEEP",
            (
                f"Done: {accepted}/{total_batches} batch skills accepted; "
                f"seed program = {seed} with {len(self._get_active_skills())} skill(s)"
            ),
        )

        if self.config.calibrate_trigger_fit and seed != "base":
            seed = await self._calibrate_trigger_fit(
                seed, target_map, all_successes_ext or []
            )

        return seed

    # ------------------------------------------------------------------ #
    # Trigger-fit calibration                                              #
    # ------------------------------------------------------------------ #

    def _read_active_skill_descriptions(self) -> dict[str, str]:
        """Map each active skill name -> its frontmatter description (deployed surface)."""
        out: dict[str, str] = {}
        skills_dir = self._project_root / ".claude" / "skills"
        if not skills_dir.exists():
            return out
        for skill_dir in sorted(skills_dir.iterdir()):
            md = skill_dir / "SKILL.md"
            if skill_dir.is_dir() and md.exists():
                out[skill_dir.name] = self._skill_description_from_markdown(
                    md.read_text()
                )
        return out

    @staticmethod
    def _parse_invoke_list(text: str, valid: set[str]) -> set[str]:
        """Extract the router's ``{"invoke": [...]}`` skill list, intersected with valid names."""
        import json
        import re

        chosen: set[str] = set()
        blob = text or ""
        start, end = blob.find("{"), blob.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(blob[start : end + 1])
                items = data.get("invoke", []) if isinstance(data, dict) else []
                if isinstance(items, list):
                    chosen = {str(x).strip() for x in items}
            except (json.JSONDecodeError, AttributeError):
                chosen = set()
        if not chosen:
            # Fallback: any valid skill name mentioned verbatim in the response.
            for name in valid:
                if re.search(rf"\b{re.escape(name)}\b", blob):
                    chosen.add(name)
        return {c for c in chosen if c in valid}

    async def _probe_skill_invocation(
        self,
        descriptions: dict[str, str],
        question: str,
        provider: str,
        model: str,
    ) -> set[str]:
        """One cheap router call: which skills would a fast agent invoke for this task?

        Shows ONLY name+description (exactly the skill list the deployed agent sees)
        and biases toward literal matching to approximate the small flash model that
        actually decides invocation in production.
        """
        skill_block = "\n".join(
            f"- {name}: {desc}" for name, desc in descriptions.items()
        )
        prompt = (
            "You simulate a fast assistant deciding which optional skills to invoke "
            "for ONE task. You see a SKILL LIST (name + description only — exactly what "
            "the assistant sees before invoking) and the task.\n\n"
            "Invoke a skill ONLY when the task LITERALLY and OBVIOUSLY matches the "
            "trigger conditions stated in its description. Do not infer, do not be "
            "helpful: a fast model skips any skill whose description does not clearly "
            "fit, and most tasks invoke zero skills.\n\n"
            f"SKILL LIST:\n{skill_block}\n\n"
            f"TASK:\n{self._compact_judge_text(question, max_chars=700)}\n\n"
            'Return JSON only: {"invoke": ["<skill-name>", ...]} (empty list if none).'
        )
        try:
            text = await self._call_judge_api(
                prompt, provider, model, max_output_tokens=200
            )
        except Exception as exc:  # noqa: BLE001 - probe must never abort the run
            _log("CALIBRATE", f"  [WARN] probe failed: {exc}")
            return set()
        return self._parse_invoke_list(text, set(descriptions))

    async def _measure_trigger_fit(
        self,
        descriptions: dict[str, str],
        target_map: dict[str, list[str]],
        holdout_qs: list[str],
        provider: str,
        model: str,
        only_skills: set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Probe firing on each skill's target failures (recall) + shared passes (false_trigger)."""
        cap_t = max(1, self.config.trigger_probe_targets)
        # Per-skill target probe questions (dedup, capped).
        skill_targets: dict[str, list[str]] = {}
        for name in descriptions:
            if only_skills is not None and name not in only_skills:
                continue
            seen: list[str] = []
            for q in target_map.get(name, []):
                if q not in seen:
                    seen.append(q)
                if len(seen) >= cap_t:
                    break
            skill_targets[name] = seen

        unique_qs = list(
            {q for qs in skill_targets.values() for q in qs} | set(holdout_qs)
        )
        if not unique_qs:
            return {}

        semaphore = asyncio.Semaphore(max(1, self.config.judge_concurrency))

        async def run_one(q: str) -> tuple[str, set[str]]:
            async with semaphore:
                return q, await self._probe_skill_invocation(
                    descriptions, q, provider, model
                )

        results = await asyncio.gather(*(run_one(q) for q in unique_qs))
        invoked_by_q: dict[str, set[str]] = dict(results)

        fit: dict[str, dict[str, Any]] = {}
        for name, tqs in skill_targets.items():
            hit_t = [q for q in tqs if name in invoked_by_q.get(q, set())]
            false_hits = [q for q in holdout_qs if name in invoked_by_q.get(q, set())]
            recall = (len(hit_t) / len(tqs)) if tqs else 0.0
            ftrig = (len(false_hits) / len(holdout_qs)) if holdout_qs else 0.0
            fit[name] = {
                "recall": recall,
                "false_trigger": ftrig,
                "n_targets": len(tqs),
                "missed_targets": [q for q in tqs if q not in hit_t],
                "false_hits": false_hits,
            }
        return fit

    async def _tune_skill_description(
        self,
        skill: str,
        current_desc: str,
        info: dict[str, Any],
        provider: str,
        model: str,
    ) -> bool:
        """Rewrite a skill's description to fix measured over/under-triggering.

        Returns True if the SKILL.md description line was changed.
        """
        md_path = self._project_root / ".claude" / "skills" / skill / "SKILL.md"
        if not md_path.exists():
            return False
        too_narrow = info["recall"] < self.config.trigger_recall_min
        too_broad = info["false_trigger"] > self.config.trigger_false_trigger_max
        if not (too_narrow or too_broad):
            return False

        md = md_path.read_text()
        # Anchor the rewrite to what the skill ACTUALLY does. Without the body, the
        # tuner broadened low-recall skills to match whatever the should-fire probe
        # questions looked like (mostly gas/thermo here), drifting the description
        # completely off the skill's mechanism. The body is the source of truth.
        purpose = self._skill_purpose_summary(md)

        missed = [self._compact_judge_text(q, max_chars=200) for q in info["missed_targets"][:4]]
        false_hits = [self._compact_judge_text(q, max_chars=200) for q in info["false_hits"][:4]]
        directive = []
        if too_narrow:
            directive.append(
                f"recall={info['recall']:.2f} is LOW: among tasks THIS skill's mechanism "
                "applies to, the description is too narrow/abstract to be recognized. Make it "
                "name the concrete cues of ITS OWN mechanism — do not adopt the topic of the "
                "SHOULD-FIRE tasks if they fall outside this skill's purpose."
            )
        if too_broad:
            directive.append(
                f"false_trigger={info['false_trigger']:.2f} is HIGH: the description fires on "
                "tasks outside this skill's mechanism. Add a sharp scope so it stays dormant on "
                "the SHOULD-NOT-FIRE tasks; avoid generic openers like 'when computing a number'."
            )
        prompt = (
            "Rewrite ONE skill's `description` line (the only text a fast assistant sees "
            "before invoking). You may ONLY adjust how broad/specific it is and add "
            "exclusions. You must NOT change the skill's topic or mechanism — the "
            "description must stay faithful to WHAT THIS SKILL DOES, described below.\n\n"
            f"Skill name: {skill}\n"
            f"What this skill actually does (from its body — the source of truth):\n{purpose}\n\n"
            f"Current description: {current_desc}\n\n"
            f"Problem to fix: {' '.join(directive)}\n\n"
            "SHOULD-FIRE tasks (it currently misses some of these):\n"
            + ("\n".join(f"- {q}" for q in missed) or "- (none)")
            + "\n\nSHOULD-NOT-FIRE tasks (must NOT match these):\n"
            + ("\n".join(f"- {q}" for q in false_hits) or "- (none)")
            + "\n\nIf the SHOULD-FIRE tasks are NOT about this skill's mechanism, do not "
            "broaden toward them — they are mis-attributed; return the current description "
            "unchanged. Return JSON only: {\"description\": \"<one line, <= 60 words, starts "
            "with 'Invoke when ...', faithful to this skill's mechanism>\"}."
        )
        try:
            text = await self._call_judge_api(
                prompt, provider, model, max_output_tokens=300
            )
        except Exception as exc:  # noqa: BLE001
            _log("CALIBRATE", f"  [WARN] tune {skill} failed: {exc}")
            return False

        import json

        new_desc = ""
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                new_desc = str(json.loads(text[start : end + 1]).get("description", "")).strip()
            except (json.JSONDecodeError, AttributeError, TypeError):
                new_desc = ""
        new_desc = " ".join(new_desc.split())
        if len(new_desc) < 15 or new_desc == current_desc.strip():
            return False
        # Drift guard: reject a rewrite that wandered off the skill's actual topic
        # (shares almost no salient vocabulary with the skill name + body).
        if not self._description_on_topic(new_desc, skill, md):
            _log("CALIBRATE", f"  [SKIP tune] {skill}: rewrite drifted off-topic, keeping original")
            return False
        md_path.write_text(self._force_frontmatter_description(md, new_desc))
        _log("CALIBRATE", f"  tuned [{skill}]: {new_desc[:90]}")
        return True

    async def _restore_faithful_descriptions(self, skills: list[str]) -> None:
        """Re-derive each skill's description from its body (repairs corruption).

        Used on sweep-resume: a prior buggy calibration could have rewritten
        descriptions to topics unrelated to the skill's body. This regenerates a
        faithful, concrete one-line trigger from the body, with the same drift guard
        the tuner uses, so a description can never wander off the skill's mechanism.
        """
        provider, model = self._detect_judge_provider_and_model()
        if provider == "codex":
            _log("CALIBRATE", "skipped description repair (codex judge)")
            return
        sem = asyncio.Semaphore(max(1, self.config.judge_concurrency))

        async def fix_one(skill: str) -> None:
            md_path = self._project_root / ".claude" / "skills" / skill / "SKILL.md"
            if not md_path.exists():
                return
            md = md_path.read_text()
            purpose = self._skill_purpose_summary(md)
            prompt = (
                "Write the `description` line for a repo-local skill — the ONLY text a "
                "fast assistant sees before invoking it. Base it STRICTLY on what the "
                "skill does (below); never introduce a topic absent from the body. It "
                "should match every task where this skill's mechanism is the risk and "
                "stay off other tasks.\n\n"
                f"Skill name: {skill}\n"
                f"What this skill does (from its body):\n{purpose}\n\n"
                "Return JSON only: {\"description\": \"<one line, <= 60 words, starts with "
                "'Invoke when ...', concrete cues of THIS skill's mechanism>\"}."
            )
            async with sem:
                try:
                    text = await self._call_judge_api(prompt, provider, model, max_output_tokens=300)
                except Exception as exc:  # noqa: BLE001
                    _log("CALIBRATE", f"  [WARN] desc repair {skill} failed: {exc}")
                    return
            import json

            new_desc = ""
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    new_desc = str(json.loads(text[start : end + 1]).get("description", "")).strip()
                except (json.JSONDecodeError, AttributeError, TypeError):
                    new_desc = ""
            new_desc = " ".join(new_desc.split())
            if len(new_desc) < 15 or not self._description_on_topic(new_desc, skill, md):
                _log("CALIBRATE", f"  [keep] {skill}: repair rejected (drift/empty), leaving as-is")
                return
            md_path.write_text(self._force_frontmatter_description(md, new_desc))
            _log("CALIBRATE", f"  [repaired] {skill}: {new_desc[:90]}")

        await asyncio.gather(*(fix_one(s) for s in skills))

    @staticmethod
    def _skill_purpose_summary(markdown: str, max_chars: int = 700) -> str:
        """Extract a compact 'what this skill does' anchor from its body sections."""
        import re

        parts: list[str] = []
        for header in ("When To Use", "Failure Mechanism", "Procedure"):
            m = re.search(
                rf"(?im)^##\s*{re.escape(header)}\s*$(.+?)(?=^##\s|\Z)",
                markdown,
                re.DOTALL | re.M,
            )
            if m:
                body = " ".join(m.group(1).split())
                if body:
                    parts.append(f"{header}: {body}")
            if sum(len(p) for p in parts) > max_chars:
                break
        summary = " | ".join(parts)
        return summary[:max_chars] if summary else "(skill body unavailable)"

    @staticmethod
    def _description_on_topic(new_desc: str, skill_name: str, markdown: str) -> bool:
        """True if the rewritten description shares salient vocabulary with the skill.

        Guards against the tuner replacing a description with one about a different
        topic. Compares content words of the candidate against the skill name + body;
        a near-zero overlap means the rewrite drifted off the skill's mechanism.
        """
        import re

        _STOP = {
            "invoke", "when", "the", "a", "an", "and", "or", "of", "to", "for", "in",
            "with", "that", "this", "is", "are", "be", "on", "as", "by", "from", "it",
            "its", "problem", "task", "asks", "answer", "value", "numeric", "compute",
            "computing", "calculate", "given", "using", "use", "where", "which", "not",
            "only", "output", "result", "any", "all", "must", "should", "no", "if",
        }
        def toks(text: str) -> set[str]:
            return {
                w for w in re.findall(r"[a-z][a-z0-9-]{3,}", text.lower())
                if w not in _STOP
            }
        anchor = toks(skill_name.replace("-", " ")) | toks(markdown)
        cand = toks(new_desc)
        if not cand:
            return False
        overlap = len(cand & anchor) / len(cand)
        return overlap >= 0.30

    async def _calibrate_trigger_fit(
        self,
        seed: str,
        target_map: dict[str, list[str]],
        all_successes_ext: list[tuple[Any, ...]],
    ) -> str:
        """Measure description-level firing, tune descriptions, keep best-fitting budget.

        Runs once after the sweep. Replaces arrival-order freeze-and-fold: every
        distinct mechanism the sweep created competes on measured trigger-fit, and
        the active-skill budget is filled by value (recall × stay-quiet) rather than
        by which skill happened to be generated first.
        """
        provider, model = self._detect_judge_provider_and_model()
        if provider == "codex":
            _log("CALIBRATE", "skipped (codex judge does not support free-form router probes)")
            return seed

        self.manager.switch_to(seed)
        descriptions = self._read_active_skill_descriptions()
        if not descriptions:
            return seed

        # Shared false-trigger probe pool: a sample of PASSING tasks (the agent must
        # stay dormant on these). Sample across categories for breadth.
        holdout_qs: list[str] = []
        for entry in all_successes_ext:
            q = str(entry[1]) if len(entry) > 1 else ""
            if q:
                holdout_qs.append(q)
        import random as _random
        _random.Random(0).shuffle(holdout_qs)
        holdout_qs = holdout_qs[: max(1, self.config.trigger_probe_holdout)]

        _log(
            "CALIBRATE",
            (
                f"Trigger-fit on {len(descriptions)} skill(s): probing recall over "
                f"targets + false-trigger over {len(holdout_qs)} held-out pass(es)..."
            ),
        )
        fit = await self._measure_trigger_fit(
            descriptions, target_map, holdout_qs, provider, model
        )
        for name, info in sorted(fit.items()):
            _log(
                "CALIBRATE",
                f"  [{name}] recall={info['recall']:.2f} false_trigger={info['false_trigger']:.2f}",
            )

        # --- Description tuning rounds ---
        for round_idx in range(1, max(0, self.config.trigger_tune_rounds) + 1):
            to_tune = {
                name
                for name, info in fit.items()
                if info["recall"] < self.config.trigger_recall_min
                or info["false_trigger"] > self.config.trigger_false_trigger_max
            }
            if not to_tune:
                break
            changed = False
            for name in sorted(to_tune):
                if await self._tune_skill_description(
                    name, descriptions.get(name, ""), fit[name], provider, model
                ):
                    changed = True
            if not changed:
                break
            self.manager.commit(f"{seed}: calibrate skill descriptions (round {round_idx})")
            descriptions = self._read_active_skill_descriptions()
            refit = await self._measure_trigger_fit(
                descriptions, target_map, holdout_qs, provider, model, only_skills=to_tune
            )
            fit.update(refit)
            for name in sorted(to_tune):
                info = fit[name]
                _log(
                    "CALIBRATE",
                    f"  [{name}] re-probe recall={info['recall']:.2f} "
                    f"false_trigger={info['false_trigger']:.2f}",
                )

        # --- Value-based budget: keep the best-fitting max_active_skills ---
        budget = self.config.max_active_skills
        active = list(descriptions)
        if self.config.trigger_fit_evict_to_budget and budget and len(active) > budget:
            def fit_score(name: str) -> float:
                # Net value of the skill's deployed firing: it helps the tasks it
                # targets (recall) but a skill that fires on tasks it cannot help is
                # actively harmful (the eval showed over-triggering LOWERS pass rate),
                # so false-triggering is penalized harder than dormancy. A quiet,
                # never-firing skill (recall 0, ft 0 -> 0) outranks an over-triggering
                # one (recall 1, ft 1 -> -1) and is evicted only after it.
                info = fit.get(name, {"recall": 0.0, "false_trigger": 1.0})
                return info["recall"] - 2.0 * info["false_trigger"]

            ranked = sorted(active, key=fit_score, reverse=True)
            evict = ranked[budget:]
            _log(
                "CALIBRATE",
                (
                    f"Budget {budget}: evicting {len(evict)} lowest trigger-fit skill(s): "
                    + ", ".join(
                        f"{n}(fit={fit_score(n):.2f})" for n in evict
                    )
                ),
            )
            self._remove_skill_dirs(evict)
            self.manager.commit(f"{seed}: trigger-fit budget prune ({len(evict)} evicted)")

        _log(
            "CALIBRATE",
            f"Done: seed {seed} retains {len(self._get_active_skills())} skill(s)",
        )
        return seed

    def _select_puct_node(self, root: ProgramSearchNode) -> ProgramSearchNode:
        """Select the next expandable parent by global PUCT score.

        Each expandable node is treated as an action candidate from its parent.
        We score all expandable nodes directly instead of first descending down
        one greedy path, so a high-value child can keep receiving expansions
        even while the root still has slots, and a full root cannot cause
        empty iterations.
        """
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
                and self._active_puct_child_count(node) < self.config.puct_children_per_node
            ):
                nodes.append(node)
            for child in node.children:
                if not child.discarded:
                    visit(child)

        visit(root)
        return nodes

    @staticmethod
    def _active_puct_child_count(node: ProgramSearchNode) -> int:
        """Count children that still consume this node's expansion capacity."""
        return sum(1 for child in node.children if not child.discarded)

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
            self._last_bandit_keys = []
            return []

        bandit_on = self.config.failure_selection == "bandit"

        # --- failure_type-aware path ---
        typed_failures = [f for f in failures if self._trajectory_failure_type(f)]
        if failure_types and typed_failures:
            types = failure_types
            n_types = len(types)
            if bandit_on:
                # Pick the failure_type by realized learning gain instead of a
                # deterministic sweep; record it so the iteration credits it.
                ft = self._bandit.select(types, 1)[0]
            else:
                ft_idx = self._category_offset % n_types
                ft = types[ft_idx]
            ft_failures = [
                f for f in typed_failures if self._trajectory_failure_type(f) == ft
            ]
            # Advance even if empty so we don't keep retrying the same type
            self._category_offset += 1
            self._last_bandit_keys = [ft]
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
            self._last_bandit_keys = []
            return self._pick_least_shown(failures, self.config.samples_per_category)

        n_cats_this_iter = min(self.config.categories_per_batch, n_cats)
        if bandit_on:
            iter_cats = self._bandit.select(categories, n_cats_this_iter)
        else:
            iter_cats = [
                categories[(self._category_offset + j) % n_cats]
                for j in range(n_cats_this_iter)
            ]
        self._last_bandit_keys = list(iter_cats)
        batch: list[tuple[Any, ...]] = []
        for cat in iter_cats:
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
        iteration: int = 0,
    ) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        """Partition failures (per category) into a proposer-visible pool and a
        disjoint judge-only pool.

        The proposer only ever sees the first pool, while candidates are scored
        on the second. This stops a proposal from being graded on the very cases
        it was written against (memorization / overfitting). Categories with a
        single failure are kept proposer-visible; if no held-out case exists at
        all, both pools fall back to the full set so scoring still has signal.

        ``iteration`` rotates the assignment each round so every failure case
        eventually becomes proposer-visible, preventing any single case from
        being permanently hidden from the proposer.
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
            _random.Random(f"holdout::{cat}::{iteration}").shuffle(items)
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
        batch_failure_types: list[str] | None = None,
    ) -> list[tuple[Any, ...]]:
        """Rotating sample of held-out failures for judging.

        Prefer the same failure_type mechanisms the proposer saw. Category
        alignment is only a fallback when typed held-out cases are unavailable,
        because unrelated mechanisms produce neutral 0.5 matches and erase the
        pairwise signal.

        Uses an offset table separate from proposer sampling so the two never
        advance in lockstep.
        """
        if not judge_pool or count <= 0:
            return []
        selected: list[tuple[Any, ...]] = []
        seen: set[str] = set()

        requested_types = [
            ft for ft in dict.fromkeys(batch_failure_types or []) if ft and ft != "regression_pass"
        ]
        if requested_types:
            by_type: dict[str, list[tuple[Any, ...]]] = {}
            for entry in judge_pool:
                ft = self._trajectory_failure_type(entry)
                if ft:
                    by_type.setdefault(ft, []).append(entry)
            matching_types = [ft for ft in requested_types if ft in by_type]
            i = 0
            guard = 0
            max_guard = count * 4 + sum(len(by_type[ft]) for ft in matching_types)
            while matching_types and len(selected) < count and guard < max_guard:
                ft = matching_types[i % len(matching_types)]
                pool = by_type[ft]
                off = self._per_failure_type_offset.get(ft, 0)
                cand = pool[off % len(pool)]
                self._per_failure_type_offset[ft] = off + 1
                key = str(cand[1])
                if key not in seen:
                    seen.add(key)
                    selected.append(cand)
                i += 1
                guard += 1
            if selected:
                _log(
                    "",
                    (
                        "  Judge held-out aligned by failure_type: "
                        f"{', '.join(matching_types)} -> {len(selected)}/{count}"
                    ),
                )
            if len(selected) >= count:
                return selected

        by_cat: dict[str, list[tuple[Any, ...]]] = {}
        for entry in judge_pool:
            by_cat.setdefault(entry[4], []).append(entry)
        cats = [c for c in dict.fromkeys(batch_categories) if c in by_cat] or sorted(by_cat)

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
        candidate_skill_tests: str = "",
        executable_skill_test_result: ExecutableSkillTestSuiteResult | None = None,
        proposer_root_cause: str = "",
        case_roles: dict[str, str] | None = None,
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
        import random as _random
        deepseek_judge = self._is_deepseek_anthropic_model(provider, model)
        summary_budget = 900 if deepseek_judge else 1600
        diff_budget = 1000 if deepseek_judge else 1800
        tests_budget = 800 if deepseek_judge else 1400
        root_cause_budget = 450 if deepseek_judge else 700
        prompt_budget = 6000 if deepseek_judge else 9000
        line_budget = 500 if deepseek_judge else 800
        question_budget = 650 if deepseek_judge else 900
        feedback_budget = 650 if deepseek_judge else 900
        trace_budget = 900 if deepseek_judge else 1400

        parent_skill_summary = self._compact_judge_text(parent_skill_summary, max_chars=summary_budget)
        candidate_skill_summary = self._compact_judge_text(candidate_skill_summary, max_chars=summary_budget)
        skill_diff = self._compact_judge_text(skill_diff, max_chars=diff_budget)
        if executable_skill_test_result is not None:
            candidate_skill_tests = (
                f"{candidate_skill_tests}\n\n"
                f"{executable_skill_test_result.to_report()}"
            )
        candidate_skill_tests = self._compact_judge_text(candidate_skill_tests, max_chars=tests_budget)
        proposer_root_cause = self._compact_judge_text(proposer_root_cause, max_chars=root_cause_budget)
        executable_self_test_rate = (
            executable_skill_test_result.pass_rate
            if executable_skill_test_result is not None and executable_skill_test_result.total > 0
            else None
        )
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
            candidate_slot: str,
        ) -> dict[str, Any] | None:
            """One judge API call for a single A/B orientation. Returns parsed
            JSON (with A in the parent slot, B in the candidate slot) or None on
            failure."""
            prompt = build_compact_judge_query(
                trace_summary,
                q,
                ans,
                gt,
                parent_skill_summary=a_summary,
                candidate_skill_summary=b_summary,
                skill_diff=diff,
                candidate_skill_tests=candidate_skill_tests,
                case_type=case_type,
                case_feedback=case_feedback,
                proposer_root_cause=proposer_root_cause,
                candidate_slot=candidate_slot,
            )
            prompt = self._compact_judge_text(prompt, max_chars=prompt_budget, max_line_chars=line_budget)
            try:
                text = await asyncio.wait_for(
                    self._call_judge_api(prompt, provider, model),
                    timeout=self.config.judge_call_timeout_seconds,
                )
                text = text.strip()
                data = self._parse_judge_json_text(text)
                data["_raw_response"] = text
                return data
            except Exception as e:
                _log("WARN", f"  Judge call failed ({type(e).__name__}): {e}")
                return None

        case_roles = case_roles or {}

        def _case_role(question: str, case_type: str) -> str:
            if case_type == "regression":
                return "regression"
            if case_type == "success_induction":
                return "proposal_success"
            return case_roles.get(question, "aligned_heldout")

        def _failed_case(index: int, category: str, case_type: str, reason: str, role: str) -> dict[str, Any]:
            category_suffix = (
                f"{category}:regression"
                if case_type == "regression"
                else f"{category}:success_induction"
                if case_type == "success_induction"
                else category
            )
            return {
                "_raw_response": "",
                "_index": index,
                "_category": category_suffix,
                "_case_role": role,
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
                role = _case_role(question, case_type)
                q = self._compact_judge_text(question, max_chars=question_budget)
                ans = self._compact_judge_text(agent_answer, max_chars=500)
                gt = self._compact_judge_text(ground_truth, max_chars=500)
                fb = self._compact_judge_text(case_feedback, max_chars=feedback_budget)
                trace_summary = self._summarize_trace_for_skill_context(
                    trace,
                    relevance_query="\n".join([
                        q,
                        fb,
                        proposer_root_cause,
                        candidate_skill_summary,
                        skill_diff,
                    ]),
                    max_chars=trace_budget,
                )

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
                        a_summary=a_sum,
                        b_summary=b_sum,
                        diff=skill_diff,
                        candidate_slot=candidate_slot,
                    )
                    if raw is None:
                        return None
                    return self._canonicalize_judge_orientation(raw, candidate_slot)

                if self.config.judge_position_swap:
                    # Two complementary orientations, averaged (2x calls).
                    d_b = await _orient("B")
                    d_a = await _orient("A")
                    if d_b is None and d_a is None:
                        return _failed_case(index, category, case_type, "Judge call failed (both orientations)", role)
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
                            "self_test_pass_rate",
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
                        return _failed_case(index, category, case_type, "Judge call failed", role)

                if executable_self_test_rate is not None:
                    # The synthetic tests were actually executed against the
                    # candidate program, so this measured result supersedes the
                    # judge's static estimate.
                    data["self_test_pass_rate"] = executable_self_test_rate
                    data["_executable_self_test_pass_rate"] = executable_self_test_rate
                data["_index"] = index
                data["_category"] = (
                    f"{category}:regression"
                    if case_type == "regression"
                    else f"{category}:success_induction"
                    if case_type == "success_induction"
                    else category
                )
                data["_case_role"] = role
                return data

        raw_results = []
        judge_tasks = []
        for i, entry in enumerate(failures_ext, start=1):
            t, q, ans, gt, cat = entry[:5]
            failure_type = self._trajectory_failure_type(entry)
            feedback = self._trajectory_failure_feedback(entry)
            role = case_roles.get(q, "")
            if role == "proposal_success":
                case_type = "success_induction"
            else:
                case_type = "regression" if failure_type == "regression_pass" else "failure"
            judge_tasks.append(judge_one(i, t, q, ans, gt, cat, case_type, feedback))
        raw_results = await asyncio.gather(*judge_tasks)
        valid_role_counts: dict[str, int] = {}
        for item in raw_results:
            if not bool(item.get("_valid", True)):
                continue
            role = str(item.get("_case_role") or "other")
            valid_role_counts[role] = valid_role_counts.get(role, 0) + 1
        valid_result_count = sum(valid_role_counts.values())

        role_group_weights = {
            "proposal_root": 0.75,
            "proposal_success": 0.75,
            "aligned_heldout": 0.10,
            "regression": 0.15,
            "inherited": 0.10,
            "other": 0.0,
        }

        def _role_case_weight(role: str) -> float:
            group_weight = role_group_weights.get(role, 0.0)
            count = valid_role_counts.get(role, 0)
            if count <= 0:
                return 0.0
            return group_weight / count

        # Anchor ratings: all expected_before values are computed from the same
        # starting point so match ordering has no effect on the final rating.
        _initial_child_rating = child_rating
        _initial_opponent_rating = opponent_rating

        matches: list[JudgeMatchResult] = []
        direct_scores: list[float] = []
        weighted_score_sum = 0.0
        weight_sum = 0.0
        for data in sorted(raw_results, key=lambda item: int(item.get("_index", 0))):
            is_valid = bool(data.get("_valid", True))
            role = str(data.get("_case_role") or "other")
            case_weight = _role_case_weight(role) if is_valid else 0.0
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
                    case_role=role,
                    case_weight=case_weight,
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
            case_relevance = self._judge_case_relevance(data)
            executable_specificity = self._clamp01(data.get("executable_specificity"), default=0.0)
            high_risk_blacklist = self._clamp01(data.get("high_risk_blacklist"), default=0.0)
            generalization_transfer = self._clamp01(data.get("generalization_transfer"), default=0.5)
            self_test_pass_rate = self._clamp01(data.get("self_test_pass_rate"), default=0.0)
            artifact_quality = self._judge_artifact_quality(data)
            gated_match_score = self._apply_artifact_quality_gate(match_score, artifact_quality)
            if gated_match_score != match_score:
                match_score = gated_match_score
                relative_advantage = max(-1.0, min(1.0, 2.0 * (match_score - 0.5)))

            # Asymmetric regression penalty: breaking a previously-correct case is
            # worse than fixing a new failure is good. A regression case is a
            # preservation check, not an opportunity for positive credit: the
            # parent already solved it. Clamp optimistic judge scores to a draw,
            # then amplify below-draw evidence so regressions cannot be averaged
            # away by unrelated fixes.
            is_regression_case = str(data.get("_category", "")).endswith(":regression")
            is_success_induction_case = str(data.get("_category", "")).endswith(":success_induction")
            reg_weight = max(1.0, float(self.config.judge_regression_penalty_weight))
            if is_regression_case:
                match_score = self._apply_regression_match_policy(match_score, reg_weight)
                relative_advantage = max(-1.0, min(1.0, 2.0 * (match_score - 0.5)))
            elif not is_success_induction_case:
                match_score = self._apply_unrelated_failure_match_policy(
                    match_score,
                    case_relevance,
                )
                relative_advantage = max(-1.0, min(1.0, 2.0 * (match_score - 0.5)))

            expected_before = self._elo_expected(_initial_child_rating, _initial_opponent_rating, scale)
            elo_weight = case_weight * max(1, valid_result_count)
            child_rating += k_factor * elo_weight * (match_score - expected_before)
            opponent_expected = 1.0 - expected_before
            opponent_rating += k_factor * elo_weight * ((1.0 - match_score) - opponent_expected)
            direct_scores.append(match_score)
            if case_weight > 0.0:
                weighted_score_sum += case_weight * match_score
                weight_sum += case_weight

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
                case_role=role,
                case_weight=case_weight,
                root_cause=str(data.get("root_cause", "")),
                case_relevance=case_relevance,
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
                self_test_pass_rate=self_test_pass_rate,
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
                        f"role={result.case_role} weight={result.case_weight:.3f} "
                        f"expected={result.expected_before:.3f} "
                        f"rating={result.child_rating_after:.1f}/{result.opponent_rating_after:.1f} "
                        f"would_succeed={result.would_succeed} confidence={result.confidence:.3f} "
                        f"score_pass={result.match_score >= 0.5} "
                        f"p_parent={result.parent_success_prob:.3f} "
                        f"p_candidate={result.candidate_success_prob:.3f} "
                        f"rel_adv={result.relative_advantage:.3f} "
                        f"relevance={result.case_relevance:.3f} "
                        f"root_fit={result.skill_addresses_root_cause:.3f} "
                        f"rc_correct={result.proposer_root_cause_correct:.3f} "
                        f"mechanism={result.failure_mechanism_encoding:.3f} "
                        f"exec={result.executable_specificity:.3f} "
                        f"blacklist={result.high_risk_blacklist:.3f} "
                        f"transfer={result.generalization_transfer:.3f} "
                        f"self_tests={result.self_test_pass_rate:.3f}"
                    ),
                )
                if result.root_cause:
                    _log("", f"  root_cause: {result.root_cause}")
                _log("", f"  action: {result.hypothetical_action}")
                if result.remaining_blockers:
                    _log("", f"  blockers: {', '.join(result.remaining_blockers)}")
                _log("", f"  reasoning: {result.reasoning}")
                _log("", f"  raw: {result.raw_response}")

        average_match_score = (
            weighted_score_sum / weight_sum
            if weight_sum > 0.0
            else (sum(direct_scores) / len(direct_scores) if direct_scores else 0.0)
        )
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
    def _judge_match_is_regression(match: "JudgeMatchResult") -> bool:
        return str(getattr(match, "category", "")).endswith(":regression")

    @staticmethod
    def _judge_match_is_target_relevant(
        match: "JudgeMatchResult",
        *,
        threshold: float = 0.30,
    ) -> bool:
        if SelfImprovingLoop._judge_match_is_regression(match):
            return False
        relevance = max(
            max(0.0, min(1.0, getattr(match, "case_relevance", 0.0))),
            max(0.0, min(1.0, getattr(match, "skill_addresses_root_cause", 0.0))),
            max(0.0, min(1.0, getattr(match, "proposer_root_cause_correct", 0.0))),
        )
        return relevance >= threshold

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
        """Mean root-cause-fit over target-relevant failure matches in [0, 1]."""
        fits = [
            m.skill_addresses_root_cause
            for m in judge_result.matches
            if getattr(m, "valid", True)
            and SelfImprovingLoop._judge_match_is_target_relevant(m)
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
    def _self_test_revision_feedback(
        result: "ExecutableSkillTestSuiteResult | None",
        max_cases: int = 6,
        max_answer_chars: int = 240,
    ) -> str:
        """Compact executable self-test failures for the skill revision prompt."""
        if result is None or result.total <= 0:
            return ""

        ordered = sorted(result.cases, key=lambda c: (c.passed, c.test_name))
        lines = [
            f"actual_pass_rate: {result.pass_rate:.3f} ({result.passed}/{result.total})",
        ]
        for case in ordered[:max_cases]:
            status = "PASS" if case.passed else "FAIL"
            lines.append(
                (
                    f"- {status} {case.skill_name}/{case.test_name}: "
                    f"should_activate={case.should_activate}, activated={case.activated}; "
                    f"{case.reason}"
                )
            )
            if case.missing_checks:
                lines.append(f"  missing_checks: {', '.join(case.missing_checks[:5])}")
            if case.missing_rejections:
                lines.append(f"  missing_rejections: {', '.join(case.missing_rejections[:5])}")
            if case.agent_answer and not case.passed:
                answer = " ".join(case.agent_answer.split())[:max_answer_chars]
                lines.append(f"  agent_answer: {answer}")
        if len(ordered) > max_cases:
            lines.append(f"- ... {len(ordered) - max_cases} more self-test cases omitted")
        return "\n".join(lines)

    @staticmethod
    def _mean_generalization_transfer(judge_result: "JudgeResult") -> float:
        """Mean transferability over target-relevant failure matches."""
        transfers = [
            max(0.0, min(1.0, getattr(m, "generalization_transfer", 0.5)))
            for m in judge_result.matches
            if getattr(m, "valid", True)
            and SelfImprovingLoop._judge_match_is_target_relevant(m)
        ]
        return sum(transfers) / len(transfers) if transfers else 0.0

    @staticmethod
    def _mean_self_test_pass_rate(judge_result: "JudgeResult") -> float:
        """Mean judge-estimated pass rate for candidate-generated skill tests."""
        rates = [
            max(0.0, min(1.0, getattr(m, "self_test_pass_rate", 0.0)))
            for m in judge_result.matches
            if getattr(m, "valid", True)
        ]
        return sum(rates) / len(rates) if rates else 0.0

    @staticmethod
    def _section_headers(markdown: str) -> set[str]:
        """Lowercased set of markdown section header titles in a SKILL.md."""
        import re

        return {
            match.group(1).strip().lower()
            for match in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", markdown or "")
        }

    def _missing_required_sections(self, markdown: str) -> list[str]:
        """Required SKILL.md sections (from config) absent in ``markdown``.

        Completeness guard: a created or edited skill that omits any required
        section is treated as an incomplete modification and routed through one
        judge-feedback refine round to fill the gap.
        """
        required = tuple(getattr(self.config, "required_skill_sections", ()) or ())
        if not required:
            return []
        headers = self._section_headers(markdown)
        return [section for section in required if section.strip().lower() not in headers]

    def _active_skill_markdown(self, skill_name: str) -> str:
        """Read one active skill's SKILL.md, or '' when it does not exist."""
        if not skill_name:
            return ""
        path = self._project_root / ".claude" / "skills" / skill_name / "SKILL.md"
        return path.read_text() if path.exists() else ""

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
        # Self-check / scaffold lines whose number is an intermediate grid or
        # bookkeeping count, not the predicate-satisfying final answer. Kept to a
        # curated keyword set so ordinary computed lines are not flagged.
        scaffold_re = re.compile(
            r"(?im)^\s*(?:"
            r"ledger_count_check|[\w ]*count_check|sanity[_ ]?check|grid[_ ]?size|"
            r"total[_ ]cells|cell[_ ]count|row[_ ]count|expected[\w ]*count|"
            r"expected\b.*\btests?"
            r")\b.*$"
        )
        # Any "<rows> x <cols> = <product>" style scaffold, with x/*/× and
        # optional word labels between the factors.
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
        """Capture skill artifact bytes so a failed revision can be rolled back."""
        skills_dir = self._project_root / ".claude" / "skills"
        snapshot: dict[str, bytes] = {}
        if skills_dir.exists():
            for pattern in ("SKILL.md", "SKILL_TESTS.json"):
                for path in skills_dir.rglob(pattern):
                    snapshot[str(path)] = path.read_bytes()
        return snapshot

    def _restore_skill_files(self, snapshot: dict[str, bytes]) -> None:
        """Restore skill artifacts captured by ``_snapshot_skill_files``."""
        skills_dir = self._project_root / ".claude" / "skills"
        snapshot_paths = {Path(path_str) for path_str in snapshot}
        if skills_dir.exists():
            for pattern in ("SKILL.md", "SKILL_TESTS.json"):
                for path in skills_dir.rglob(pattern):
                    if path not in snapshot_paths:
                        path.unlink()
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
        judge_case_roles: dict[str, str] | None,
        judge_provider: str,
        judge_model: str,
        parent_skill_summary: str,
        parent_skills_content: str,
        proposer_root_cause: str,
        original_proposal: str,
        self_test_gap: bool = False,
        executable_skill_test_result: "ExecutableSkillTestSuiteResult | None" = None,
        induction: bool = False,
    ) -> tuple["JudgeResult", str, str, str, str, bool]:
        """Revise the child's skill once from the judge's verdict, then re-judge.

        Returns (judge_result, candidate_skills_content, candidate_skill_summary,
        skill_diff, skill_diff_swapped, revised_kept). The pre-revision skill
        files and the original judge_result are restored whenever the revision
        does not improve the average match score vs the parent, so a refine can
        never lower a child's score.
        """
        def _artifacts() -> tuple[str, str, str, str, str]:
            content = self._get_all_skills_content()
            summary = self._summarize_skill_content(content)
            diff = self._diff_skill_content(parent_skills_content, content)
            diff_sw = self._diff_skill_content(content, parent_skills_content)
            tests = self._get_all_skill_tests_content([skill_name])
            return content, summary, diff, diff_sw, tests

        skill_path = (
            self._project_root / ".claude" / "skills" / skill_name / "SKILL.md"
            if skill_name
            else None
        )
        if skill_path is None or not skill_path.exists():
            content, summary, diff, diff_sw, _tests = _artifacts()
            return judge_result, content, summary, diff, diff_sw, False

        judge_root_cause, judge_blockers = self._summarize_judge_feedback(judge_result)
        judge_reasoning = self._weakest_case_reasoning(judge_result)
        snapshot = self._snapshot_skill_files()
        current_markdown_full = skill_path.read_text()
        current_markdown = compact_relevant_text(
            current_markdown_full,
            "\n".join([
                proposer_root_cause,
                original_proposal,
                judge_root_cause,
                *judge_blockers,
            ]),
            max_chars=5200,
            context_lines=1,
            max_line_chars=700,
        )
        structural_gaps = (
            self._missing_required_sections(current_markdown_full)
            if self.config.enforce_skill_sections
            else []
        )
        revision_query = build_skill_revision_query(
            target_skill=skill_name,
            current_skill_markdown=current_markdown,
            proposer_root_cause=proposer_root_cause,
            original_proposal=original_proposal,
            judge_root_cause=judge_root_cause,
            judge_blockers=judge_blockers,
            judge_reasoning=judge_reasoning,
            structural_gaps=structural_gaps,
            self_test_gap=self_test_gap,
            executable_self_test_feedback=self._self_test_revision_feedback(
                executable_skill_test_result
            ),
            induction=induction,
        )
        try:
            revise_trace = await self.agents.skill_generator.run(revision_query)
        except Exception as e:  # noqa: BLE001 — generator transport may raise broadly
            _log("WARN", f"  [REFINE] Generator failed ({type(e).__name__}: {e}); keeping original")
            content, summary, diff, diff_sw, _tests = _artifacts()
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
            content, summary, diff, diff_sw, _tests = _artifacts()
            return judge_result, content, summary, diff, diff_sw, False

        rev_content, rev_summary, rev_diff, rev_diff_sw, rev_tests = _artifacts()
        rev_executable_tests = await self._run_executable_skill_tests([skill_name])
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
                candidate_skill_tests=rev_tests,
                executable_skill_test_result=rev_executable_tests,
                proposer_root_cause=proposer_root_cause,
                case_roles=judge_case_roles,
            )
        except RuntimeError as e:
            _log("WARN", f"  [REFINE] Re-judge failed ({e}); keeping original")
            self._restore_skill_files(snapshot)
            content, summary, diff, diff_sw, _tests = _artifacts()
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
        content, summary, diff, diff_sw, _tests = _artifacts()
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
        resume_seed_cfg = getattr(self.config, "sweep_resume_seed", "") or ""
        if not self.config.continue_mode and not resume_seed_cfg:
            if self.config.reset_feedback and self._feedback_path.exists():
                self._feedback_path.unlink()
            if self.config.reset_feedback and self._meta_path.exists():
                self._meta_path.unlink()
            self._meta_records = []
            self._iteration_offset = 0
            self._delete_checkpoint()
            reset_manager = getattr(self.manager, "reset", None)
            if callable(reset_manager):
                reset_manager()
                _log("INIT", "Reset local program state for fresh run")
        else:
            # Resume-from-seed must NOT wipe local_programs (the swept seed lives
            # there); keep iteration numbering past the existing programs.
            self._iteration_offset = self._get_highest_iteration()
            if resume_seed_cfg:
                _log("INIT", f"Sweep-resume mode: preserving programs, reusing seed '{resume_seed_cfg}'")

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
            trace, question, agent_answer, ground_truth, original_category = entry[:5]
            failure_type = self._trajectory_failure_type(entry)
            failure_feedback = self._trajectory_failure_feedback(entry)
            score = self.scorer(question, agent_answer.strip().lower(), ground_truth.strip().lower())
            if score >= self.config.failure_threshold:
                passed += 1
                category = "success"
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
                failure_type = failure_type or "unclassified_failure"
                category = failure_type
                if not failure_feedback:
                    failure_feedback = build_answer_comparison_feedback(
                        question,
                        agent_answer,
                        ground_truth,
                        failure_type or original_category,
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
        dynamic_categories = sorted(
            {
                str(entry[4])
                for entry in [*all_failures_ext, *all_successes_ext]
                if str(entry[4])
            }
        )
        _log(
            "COLLECT",
            (
                "Dynamic trajectory categories: "
                + (", ".join(dynamic_categories) if dynamic_categories else "<none>")
            ),
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
        _log(
            "JUDGE",
            "Judging primarily on proposal failure mechanisms + aligned held-out + regression",
        )
        # Pre-PUCT coverage sweep: generate a skill per failure cluster and chain
        # the accepted ones into a "seed" program. The seed becomes the PUCT root
        # and the Bradley-Terry gauge anchor; base survives only as the hidden
        # base_score (0.70) origin value. When the sweep is disabled or accepts
        # nothing, seed_name == "base" and behavior is unchanged.
        resume_seed = getattr(self.config, "sweep_resume_seed", "") or ""
        if resume_seed and resume_seed in self.manager.list_programs():
            # Reuse an already-swept seed; skip the expensive sweep and just repair
            # the skill descriptions with the fixed (body-anchored) logic, then PUCT.
            _log("SWEEP", f"Resuming from swept seed '{resume_seed}' (skipping sweep)")
            self.manager.switch_to(resume_seed)
            # Purge every OTHER program (old sweep intermediates + stale PUCT children
            # from a prior interrupted run). Otherwise PUCT rebuilds its tree on those
            # leftover nodes (and their old frontier scores), expanding from stale
            # children whose descriptions predate this repair. PUCT must start fresh
            # from the repaired seed only.
            for prog in self.manager.list_programs():
                if prog not in ("base", resume_seed):
                    try:
                        self.manager.discard(prog)
                    except Exception as exc:  # noqa: BLE001
                        _log("SWEEP", f"  [WARN] could not discard stale program {prog}: {exc}")
            self.manager.switch_to(resume_seed)
            self._iteration_offset = 0
            await self._restore_faithful_descriptions(self._get_active_skills())
            self.manager.commit(f"{resume_seed}: re-derived faithful skill descriptions")
            seed_name = resume_seed
        else:
            seed_name = await self._sweep_failure_coverage(
                all_failures_ext, all_successes_ext
            )
        self._evolution_root = seed_name
        if seed_name != "base":
            self.manager.switch_to(seed_name)

        self.manager.update_frontier(seed_name, base_score, max_size=self.config.frontier_size)
        self._emit("baseline", score=base_score, n_skills=len(self._get_active_skills()))
        search_root = self._build_program_search_tree(base_score, root_name=seed_name)
        bt_matches: list[BradleyTerryMatch] = []
        bt_players: set[str] = {seed_name}
        bt_player_judge_results: dict[str, JudgeResult] = {}
        bt_ratings: dict[str, float] = {
            seed_name: self.config.judge_elo_initial_rating,
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
        # In LLM-judge mode categories are derived from outcomes and failure
        # mechanisms, not benchmark difficulty or dataset-provided categories.
        categories = sorted({entry[4] for entry in [*all_failures_ext, *all_successes_ext]})
        failure_types = sorted(
            {
                self._trajectory_failure_type(entry)
                for entry in all_failures_ext
                if self._trajectory_failure_type(entry)
            }
        )
        n_cats = len(categories)
        no_improvement_count = 0
        iteration_count = 0

        for i in range(self.config.max_iterations):
            iteration_count = i + 1
            actual_iteration = iteration_count + self._iteration_offset

            # Rotate held-out / proposer-visible split each iteration so every
            # failure case eventually becomes proposer-visible.
            proposer_failures_ext, judge_failures_ext = self._split_holdout(
                all_failures_ext, iteration=actual_iteration
            )

            parent_node = self._select_puct_node(search_root)
            if (
                parent_node.depth >= self.config.puct_max_depth
                or self._active_puct_child_count(parent_node)
                >= self.config.puct_children_per_node
            ):
                _log(
                    f"ITER {iteration_count}/{self.config.max_iterations}",
                    "No expandable PUCT nodes remain; stopping search early.",
                )
                break
            parent = parent_node.name
            self.manager.switch_to(parent)
            self._iter_cost = 0.0
            _log(
                f"ITER {iteration_count}/{self.config.max_iterations}",
                (
                    f"Parent: {parent} | PUCT path: {self._format_puct_path(parent_node)} | "
                    f"q={parent_node.q_value:.4f}, visits={parent_node.visit_count}, "
                    f"children={self._active_puct_child_count(parent_node)} active/"
                    f"{len(parent_node.children)} total"
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
            # The judge primarily asks whether the proposal fixed the failure
            # mechanisms it was generated for. Same-type held-out cases are
            # secondary transfer checks; regression cases below are preservation
            # checks. This avoids unrelated failures collapsing every match to
            # a neutral draw.
            batch_categories = [f[4] for f in batch_failures_ext]
            batch_failure_types = [self._trajectory_failure_type(f) for f in batch_failures_ext]
            judge_batch_ext = self._sample_judge_failures(
                judge_failures_ext,
                batch_categories,
                count=max(len(batch_failures_ext), self.config.failure_sample_count),
                batch_failure_types=batch_failure_types,
            )
            # By default the judge scores this batch's proposal failures first,
            # then aligned held-out failures (plus a small regression sample
            # below), not the parent's inherited superset — inheriting was the
            # dominant judge-call multiplier.
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
                batch_failures_ext + judge_batch_ext,
                max_size=self.config.samples_per_category * self.config.categories_per_batch * 3,
            )
            regression_cases_ext = self._sample_regression_successes(
                all_successes_ext,
                categories,
                count=max(
                    1,
                    math.ceil(
                        len(batch_failures_ext)
                        * max(0.0, self.config.judge_regression_sample_multiplier)
                    ),
                ),
            )
            regression_questions = {entry[1] for entry in regression_cases_ext}
            scoring_cases_ext = [
                entry
                for entry in scoring_failures_ext
                if entry[1] not in regression_questions
            ] + regression_cases_ext
            judge_case_roles: dict[str, str] = {}
            for entry in batch_failures_ext:
                judge_case_roles[entry[1]] = "proposal_root"
            for entry in judge_batch_ext:
                judge_case_roles.setdefault(entry[1], "aligned_heldout")
            for entry in inherited_failures:
                judge_case_roles.setdefault(entry[1], "inherited")
            for entry in regression_cases_ext:
                judge_case_roles[entry[1]] = "regression"

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
            proposal_questions_set = {f[1] for f in batch_failures_ext}
            heldout_questions_set = {f[1] for f in judge_batch_ext}
            n_proposal_scored = sum(1 for f in scoring_failures_ext if f[1] in proposal_questions_set)
            n_heldout_scored = sum(1 for f in scoring_failures_ext if f[1] in heldout_questions_set)
            n_new = sum(1 for f in scoring_failures_ext if f[1] not in parent_questions)
            n_inherited = len(scoring_failures_ext) - n_new
            _log(
                "",
                (
                    f"  Using {len(batch_failures)} proposal failures; "
                    f"judging on {len(scoring_failures_ext)} failure cases "
                    f"({n_proposal_scored} proposal-root + {n_heldout_scored} aligned held-out, "
                    f"{n_inherited} inherited from parent + {n_new} new) "
                    f"+ {len(regression_cases_ext)} regression cases..."
                ),
            )

            target_children = max(1, self.config.children_per_iteration)
            remaining_child_slots = max(
                0,
                self.config.puct_children_per_node
                - self._active_puct_child_count(parent_node),
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
                diversity_hint = self._build_child_diversity_hint(
                    child_idx,
                    sibling_proposals,
                    existing_children=parent_node.children,
                )

                # Decide this child's source: fix a failure, or distill a skill
                # from a successful trajectory. A fractional accumulator turns the
                # configured ratio into whole induction children over time.
                do_induction = False
                induction_batch: list[tuple[Any, ...]] = []
                induction_questions: list[str] = []
                induction_ext: list[tuple[Any, ...]] = []
                if self.config.induce_skills_from_success and all_successes_ext:
                    self._induction_credit += self.config.success_induction_ratio
                    if self._induction_credit >= 1.0:
                        induction_ext = self._sample_induction_successes(
                            all_successes_ext,
                            all_failures_ext,
                            count=max(1, len(batch_failures)),
                        )
                        if induction_ext:
                            self._induction_credit -= 1.0
                            do_induction = True
                            for s in induction_ext:
                                t, q, ans, gt, cat = s[:5]
                                fb = self._trajectory_failure_feedback(s)
                                induction_batch.append((t, ans, gt, cat, fb))
                                induction_questions.append(q)

                if do_induction:
                    _log(
                        "CHILD",
                        f"  source: SUCCESS-INDUCTION ({len(induction_batch)} "
                        f"winning trajectories)",
                    )
                    mutation_result = await self._mutate_with_fallback(
                        parent,
                        induction_batch,
                        child_iteration_id,
                        diversity_hint=diversity_hint,
                        questions=induction_questions,
                        induction=True,
                    )
                else:
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
                child_scoring_cases_ext = scoring_cases_ext
                child_judge_case_roles = judge_case_roles
                if do_induction and induction_ext:
                    seen_questions: set[str] = set()
                    child_scoring_cases_ext = []
                    child_judge_case_roles = {}
                    for entry in induction_ext:
                        if entry[1] in seen_questions:
                            continue
                        seen_questions.add(entry[1])
                        child_scoring_cases_ext.append(entry)
                        child_judge_case_roles[entry[1]] = "proposal_success"
                    for entry in scoring_failures_ext:
                        if entry[1] in seen_questions:
                            continue
                        seen_questions.add(entry[1])
                        child_scoring_cases_ext.append(entry)
                        child_judge_case_roles[entry[1]] = "aligned_heldout"
                    for entry in regression_cases_ext:
                        if entry[1] in seen_questions:
                            continue
                        seen_questions.add(entry[1])
                        child_scoring_cases_ext.append(entry)
                        child_judge_case_roles[entry[1]] = "regression"
                    _log(
                        "JUDGE",
                        (
                            f"{child_name}: success-induction scoring uses "
                            f"{len(induction_ext)} proposal-success case(s), "
                            f"{len(scoring_failures_ext)} secondary failure case(s), "
                            f"{len(regression_cases_ext)} regression case(s)"
                        ),
                    )
                # CoEvoSkills: replace the author-written tests with independent,
                # information-isolated verifier tests BEFORE reading/judging them.
                await self._author_verifier_tests(
                    [a["skill_name"] for a in mutation_result.applied_skills]
                )
                touched_skill_names = [
                    a["skill_name"]
                    for a in mutation_result.applied_skills
                    if a.get("skill_name")
                ]
                candidate_skills_content = self._get_all_skills_content()
                candidate_skill_tests = self._get_all_skill_tests_content(
                    touched_skill_names
                )
                executable_skill_test_result = await self._run_executable_skill_tests(
                    touched_skill_names
                )
                parent_skill_summary = self._summarize_skill_content(parent_skills_content)
                candidate_skill_summary = self._summarize_skill_content(candidate_skills_content)
                skill_diff = self._diff_skill_content(parent_skills_content, candidate_skills_content)
                skill_diff_swapped = self._diff_skill_content(candidate_skills_content, parent_skills_content)

                # Judge with LLM — no full agent re-run
                _log("", f"  -> Judging {child_name} via LLM ({judge_provider}/{judge_model})...")
                try:
                    judge_result = await self._judge_skill_with_llm(
                        child_scoring_cases_ext,
                        judge_provider,
                        judge_model,
                        child_name=child_name,
                        parent_name=parent,
                        parent_skill_summary=parent_skill_summary,
                        candidate_skill_summary=candidate_skill_summary,
                        skill_diff=skill_diff,
                        skill_diff_swapped=skill_diff_swapped,
                        candidate_skill_tests=candidate_skill_tests,
                        executable_skill_test_result=executable_skill_test_result,
                        proposer_root_cause=mutation_result.root_cause_analysis,
                        case_roles=child_judge_case_roles,
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
                    mean_self_tests = self._mean_self_test_pass_rate(judge_result)
                    misleading_example = (
                        self.config.validate_worked_examples
                        and self._has_misleading_worked_example(candidate_skills_content)
                    )
                    # Completeness: the changed skill must carry every required
                    # SKILL.md section; a missing section is an incomplete edit.
                    missing_sections = (
                        self._missing_required_sections(
                            self._active_skill_markdown(mutation_result.skill_name)
                        )
                        if self.config.enforce_skill_sections and mutation_result.skill_name
                        else []
                    )
                    # The executable self-test suite is the candidate's own re-run
                    # signal; a suite with no negative (should_activate=false) case
                    # never tested over-triggering, so its pass rate is gameable.
                    weak_self_test_coverage = (
                        executable_skill_test_result is not None
                        and executable_skill_test_result.total > 0
                        and not any(
                            not case.should_activate
                            for case in executable_skill_test_result.cases
                        )
                    )
                    needs_refine = (
                        mean_root_fit < self.config.refine_root_cause_threshold
                        or mean_transfer < self.config.refine_generalization_threshold
                        or mean_self_tests < self.config.refine_self_test_threshold
                        or judge_result.average_match_score <= 0.5
                        or misleading_example
                        or bool(missing_sections)
                        or weak_self_test_coverage
                    )
                    if not needs_refine:
                        break
                    if misleading_example:
                        _log(
                            "REFINE",
                            f"{child_name}: worked example emits a scaffold count as the answer "
                            "-> revising",
                        )
                    if missing_sections:
                        _log(
                            "REFINE",
                            f"{child_name}: skill missing required sections "
                            f"{missing_sections} -> revising",
                        )
                    if weak_self_test_coverage:
                        _log(
                            "REFINE",
                            f"{child_name}: self-test suite has no negative "
                            "(should_activate=false) case -> revising",
                        )
                    _log(
                        "REFINE",
                        (
                            f"{child_name}: root_fit={mean_root_fit:.3f} "
                            f"transfer={mean_transfer:.3f} "
                            f"self_tests={mean_self_tests:.3f} "
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
                        scoring_cases_ext=child_scoring_cases_ext,
                        judge_case_roles=child_judge_case_roles,
                        judge_provider=judge_provider,
                        judge_model=judge_model,
                        parent_skill_summary=parent_skill_summary,
                        parent_skills_content=parent_skills_content,
                        proposer_root_cause=mutation_result.root_cause_analysis,
                        original_proposal=proposal,
                        self_test_gap=weak_self_test_coverage,
                        executable_skill_test_result=executable_skill_test_result,
                        induction=do_induction,
                    )
                    candidate_skill_tests = self._get_all_skill_tests_content(
                        touched_skill_names
                    )
                    executable_skill_test_result = await self._run_executable_skill_tests(
                        touched_skill_names
                    )
                    refine_round += 1
                    if not revised_kept:
                        break

                # Estimate total score: Elo-estimated fix rate is the fraction of
                # failed samples the child is expected to recover. Scale it to the
                # observed base-score space.
                failed_negative_self_test = (
                    self.config.discard_on_negative_self_test_failure
                    and self._has_failed_negative_self_test(executable_skill_test_result)
                )
                forced_discard = failed_negative_self_test
                if failed_negative_self_test:
                    _log(
                        "SELF-TEST",
                        (
                            f"{child_name}: failed verifier-authored negative activation "
                            "test; child will be eliminated after judging"
                        ),
                    )
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
                                child_scoring_cases_ext,
                                judge_provider,
                                judge_model,
                                child_name=child_name,
                                parent_name=anchor_name,
                                parent_skill_summary=a_summary,
                                candidate_skill_summary=candidate_skill_summary,
                                skill_diff=a_diff,
                                skill_diff_swapped=a_diff_sw,
                                candidate_skill_tests=candidate_skill_tests,
                                executable_skill_test_result=executable_skill_test_result,
                                proposer_root_cause=mutation_result.root_cause_analysis,
                                case_roles=child_judge_case_roles,
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
                    # Drop non-informative draws (match_score ~ 0.5): cases the
                    # narrow skill does not address return "no effect" and, as
                    # ties, regress every node's rating toward the anchor over
                    # iterations (scores decay to base). Keeping only decisive
                    # comparisons preserves real wins and real regression losses
                    # (over-trigger losses sit far below 0.5 and survive this).
                    draw_eps = max(0.0, self.config.bt_draw_epsilon)
                    new_bt_matches = [
                        BradleyTerryMatch(
                            player=child_name,
                            opponent=anchor,
                            score=match.match_score,
                            category=match.category,
                            index=match.index,
                            weight=match.case_weight
                            * max(1, sum(1 for m in result.matches if m.valid)),
                        )
                        for anchor, result in anchor_results
                        for match in result.matches
                        if match.valid
                        and abs(match.match_score - 0.5) >= draw_eps
                    ]
                    bt_matches.extend(new_bt_matches)
                    # Only fit over matches where both sides are currently
                    # active (frontier + new child + anchors).  Old discarded
                    # nodes stay in bt_matches for history but must not shift
                    # ratings of present candidates.
                    # Gauge anchor is the evolution root (seed), not base. base
                    # only survives as the hidden base_score (0.70) origin value
                    # the seed maps to. This keeps the league connected to a real,
                    # judged node so ratings don't float and collapse onto base.
                    _active_players = (
                        {seed_name, child_name}
                        | {n for n, _ in self.manager.get_frontier_with_scores()}
                        | set(anchors)
                    )
                    _active_bt_matches = self._filter_bt_matches_for_active_players(
                        bt_matches, _active_players
                    )
                    bt_ratings = self._fit_bradley_terry_ratings(
                        _active_bt_matches,
                        sorted(_active_players),
                        anchor=seed_name,
                        initial_rating=self.config.judge_elo_initial_rating,
                        scale=self.config.judge_elo_scale,
                    )
                    anchor_rating = bt_ratings.get(seed_name, self.config.judge_elo_initial_rating)
                    child_rating = bt_ratings.get(child_name, self.config.judge_elo_initial_rating)
                    child_uncertainty = self._bt_player_uncertainty(
                        child_name,
                        _active_bt_matches,
                        bt_player_judge_results,
                    )
                    child_raw_score = self._rating_to_score(
                        child_rating,
                        anchor_rating,
                        base_score,
                        self.config.judge_elo_scale,
                    )
                    child_score = self._rating_to_score_with_uncertainty(
                        child_rating,
                        anchor_rating,
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
                            if rated_name == seed_name
                            else self._rating_to_score_with_uncertainty(
                                rating,
                                anchor_rating,
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
                        anchor_rating,
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
                            f"anchor_rating={anchor_rating:.1f}, expected_vs_anchor={expected_win:.3f}, "
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
                    outcome = (
                        "negative_self_test_failure"
                        if failed_negative_self_test
                        else "duel_loss"
                    )
                    reason = (
                        "failed a verifier-authored negative activation self-test"
                        if failed_negative_self_test
                        else "lost to champion"
                    )
                    _log(
                        "",
                        f"  [DISCARD] {child_name} {reason} (score: {child_score:.4f}); eliminated",
                    )
                else:
                    # Phase 2 (QD): keep one elite per category niche using a
                    # coverage vector from the judge's per-category matches. Only
                    # in non-BT scoring — BT manages the frontier via its league
                    # refit above, which a per-niche reconciliation would fight.
                    use_qd = (
                        self.config.frontier_mode == "qd"
                        and self.config.judge_scoring != "bradley_terry"
                    )
                    if use_qd:
                        coverage = self._judge_coverage_vector(judge_result)
                        added = self.manager.update_frontier_qd(
                            child_name, child_score, coverage
                        )
                    else:
                        added = self.manager.update_frontier(
                            child_name, child_score, max_size=self.config.frontier_size
                        )
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
                    prior=self._proposal_policy_prior(
                        proposer_confidence,
                        proposal=proposal,
                        existing_skill_content=parent_skills_content,
                        sibling_proposals=[
                            child.proposal for child in parent_node.children if child.proposal
                        ],
                    ),
                    discarded=is_discarded,
                )
                child_node.proposal = proposal
                child_node.proposal_action = (
                    mutation_result.applied_skills[0].get("action", "")
                    if mutation_result.applied_skills
                    else ""
                )
                child_node.proposal_skill = mutation_result.skill_name
                # Record which failures this child was scored on so its own
                # children will inherit them and comparisons remain consistent.
                child_node.scoring_failures = child_scoring_cases_ext
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

                # Phase 1: an improving child proves the bullets it added/reinforced
                # were helpful (judge-gated). Bump + commit on the child branch so
                # the credit persists and the lineage inherits it.
                if self.config.use_playbook and added and child_score > parent_score:
                    # Ensure we credit on the child's own branch (judging/duels
                    # may have moved the checkout).
                    self.manager.switch_to(child_name)
                    bumped = self._credit_playbook(mutation_result)
                    if bumped:
                        self.manager.commit(
                            f"{child_name}: playbook +helpful x{bumped}"
                        )

                # Epoch meta: record this child's outcome for later consolidation.
                self._meta_records.append({
                    "outcome": outcome,
                    "delta": child_score - parent_score,
                    "blockers": judge_blockers or [],
                    "root_cause": judge_root_cause or "",
                    "source": "induction" if do_induction else "failure",
                    "skill": mutation_result.skill_name,
                })

            _frontier_after = self.manager.get_frontier_with_scores()
            best_score_after_iter = _frontier_after[0][1] if _frontier_after else base_score
            improved_this_iter = best_score_after_iter > best_score_before_iter
            if improved_this_iter:
                no_improvement_count = 0
            elif any_child_created:
                no_improvement_count += 1
            else:
                no_improvement_count += 1

            # Phase 3: credit the bandit keys (failure_type/category) this batch
            # was drawn from with the iteration's realized learning gain.
            if self.config.failure_selection == "bandit" and self._last_bandit_keys:
                delta = best_score_after_iter - best_score_before_iter
                for key in set(self._last_bandit_keys):
                    self._bandit.update(key, improved_this_iter, delta)

            # Epoch boundary: consolidate accumulated outcomes into the protected
            # meta summary (deterministic; no LLM call, no context collapse).
            if (
                self.config.meta_update_enabled
                and self.config.meta_update_period > 0
                and iteration_count % self.config.meta_update_period == 0
                and self._meta_records
            ):
                meta_text = consolidate_meta(self._meta_records, top_k=self.config.meta_top_k)
                if meta_text.strip():
                    write_meta(self._meta_path, meta_text)
                    _log(
                        "META",
                        f"Consolidated {len(self._meta_records)} outcomes "
                        f"through iter {iteration_count} -> {self._meta_path.name}",
                    )

            if _no_improvement_stop(no_improvement_count, self.config.no_improvement_limit):
                _log("STOP", f"No improvement for {self.config.no_improvement_limit} iterations")
                break

            frontier_str = ", ".join(f"{n}:{s:.2f}" for n, s in self.manager.get_frontier_with_scores())
            _log("", f"  Frontier: [{frontier_str}]")
            self._total_cost += self._iter_cost
            _log("COST", f"Iter {iteration_count} cost: ${self._iter_cost:.4f} | Running total: ${self._total_cost:.4f}")
            self._save_checkpoint(actual_iteration)

        # 5. Distill the strongest frontier variants into one final validated skill.
        frontier = self.manager.get_frontier_with_scores()
        best = self.manager.get_best_from_frontier()
        best_score = frontier[0][1] if frontier else 0.0
        self._iter_cost = 0.0
        distilled = await self._distill_frontier_skill(
            frontier=frontier,
            judge_provider=judge_provider,
            judge_model=judge_model,
            failure_cases=judge_failures_ext or all_failures_ext,
            regression_cases=all_successes_ext,
        )
        if distilled is not None:
            best, best_score = distilled
            frontier = self.manager.get_frontier_with_scores()
        self._total_cost += self._iter_cost

        # 6. Return results
        _log("DONE", f"{iteration_count} iterations, best: {best or 'base'} ({best_score:.4f})")
        _log("COST", self._format_cost_breakdown())
        self._emit("loop_done", best=best or "base", best_score=best_score, iterations=iteration_count)
        return self._build_loop_result(
            frontier,
            best or "base",
            best_score,
            iteration_count,
        )
