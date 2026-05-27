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
    build_proposer_query,
    build_skill_query,
    build_prompt_query,
    build_skill_query_from_skill_proposer,
    build_prompt_query_from_prompt_proposer,
    build_judge_query,
    append_feedback,
    read_feedback_history,
    update_prompt_file,
)


T = TypeVar("T")

TOLERANCE_LEVELS = [0.05, 0.01, 0.1, 0.0, 0.025]


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
    probability_of_success: float = 0.0
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
        preloaded_trajectories: list[tuple[Any, str, str, str, str]] | None = None,
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
        if preloaded_trajectories:
            cats = sorted({entry[4] for entry in preloaded_trajectories})
            self._per_cat_offset: dict[str, int] = {cat: 0 for cat in cats}
        else:
            self._per_cat_offset = {cat: 0 for cat in train_pools.keys()}

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

    def _emit(self, event: str, **data: Any) -> None:
        """Fire an event to the display callback if one is registered."""
        if self.on_event is not None:
            self.on_event(event, data)

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
            self._iter_cost += sum(t.total_cost_usd for t in traces)

            # Collect failures
            failures: list[tuple[AgentTrace, str, str, str]] = []  # (trace, agent_answer, ground_truth, category)
            for trace, (question, answer, category) in zip(traces, test_samples):
                agent_answer = (
                    trace.output.final_answer if trace.output and trace.output.final_answer else "[PARSE FAILED]"
                )
                avg_score = self.scorer(
                    question,
                    agent_answer.strip().lower(),
                    answer.strip().lower(),
                )
                status = "[OK]" if avg_score >= 0.8 else "[FAIL]"
                if self.on_event is None:
                    _log("", f"    {status} [{category}] {question[:40]}...")
                self._emit("sample", question=question, category=category, score=avg_score, passed=avg_score >= 0.8)
                if avg_score < 0.8:
                    failures.append((trace, agent_answer, answer, category))

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
                child_name, proposal, justification, _proposer_confidence = mutation_result

                # Evaluate child
                _log("", f"  -> Evaluating {child_name}...")
                child_score = await self._evaluate(self.val_data)  # accumulates to self._iter_cost

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
        _log("COST", f"Total cost: ${self._total_cost:.4f}")
        self._emit("loop_done", best=best or "base", best_score=best_score, iterations=iteration_count)

        return LoopResult(
            frontier=frontier,
            best_program=best or "base",
            best_score=best_score,
            iterations_completed=iteration_count,
            total_cost_usd=self._total_cost,
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
        _log("COST", f"Base eval cost: ${self._iter_cost:.4f} | Total: ${self._total_cost:.4f}")
        self._emit("baseline", score=base_score, n_skills=len(self._get_active_skills()))

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
                self._iter_cost += result.trace.total_cost_usd
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
    ) -> tuple[str, str, str, float] | None:
        """Run proposer and generator to create a mutation based on multiple failures.

        Args:
            parent: Name of the parent program.
            failures: List of (trace, agent_answer, ground_truth, category) tuples from failed attempts.
            iteration: Current iteration number.
            truncation_level: Context reduction level (0=full, 1=moderate, 2=aggressive).
            diversity_hint: Optional instruction used to diversify sibling children.

        Returns:
            Tuple of (child_name, proposal, justification, proposer_confidence)
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
        )

        if evolution_mode == "skill_only":
            proposer_trace = await self.agents.skill_proposer.run(proposer_query)
            self._iter_cost += proposer_trace.total_cost_usd

            if proposer_trace.output is None:
                _log("", f"  [WARN] Skill proposer failed: {proposer_trace.parse_error}")
                return None

            proposer_output = proposer_trace.output
            proposed = proposer_output.proposed_skill
            justification = proposer_output.justification
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

Read the existing skill at .claude/skills/{target_skill}/SKILL.md
and modify it to add these capabilities. Preserve all existing content that is still relevant."""
            else:
                _log("", f"  -> Generating new skill...")
                skill_query = build_skill_query_from_skill_proposer(proposer_trace)

            skills_before = set(self._get_active_skills())
            skill_trace = await self.agents.skill_generator.run(skill_query)
            self._iter_cost += skill_trace.total_cost_usd
            skills_after = set(self._get_active_skills())
            new_skills = skills_after - skills_before
            created_skill = next(iter(new_skills)) if new_skills else None

            if is_opencode_sdk() or is_openhands_sdk() or is_goose_sdk() or is_codex_sdk():
                from src.harness.opencode.skill_utils import normalize_project_skill_frontmatter
                from src.harness.sdk_config import get_sdk
                skill_descriptions: dict[str, str] = {}
                if target_skill:
                    skill_descriptions[target_skill] = proposed
                if created_skill:
                    skill_descriptions[created_skill] = proposed
                normalize_project_skill_frontmatter(
                    self._project_root,
                    descriptions=skill_descriptions,
                    fallback_description=proposed,
                    compatibility=get_sdk(),
                )

            if skill_trace.output:
                self._emit("skill_written", name=created_skill, action=action_type, target=target_skill)

        else:  # prompt_only
            proposer_trace = await self.agents.prompt_proposer.run(proposer_query)
            self._iter_cost += proposer_trace.total_cost_usd

            if proposer_trace.output is None:
                _log("", f"  [WARN] Prompt proposer failed: {proposer_trace.parse_error}")
                return None

            proposed = proposer_trace.output.proposed_prompt_change
            justification = proposer_trace.output.justification
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
            self._iter_cost += prompt_trace.total_cost_usd
            if prompt_trace.output:
                update_prompt_file(
                    self._prompt_path, prompt_trace.output.optimized_prompt
                )

        # Commit changes
        self.manager.commit(f"{child_name}: {proposed[:50]}")

        # Return mutation info (feedback will be written by caller with outcome)
        return (child_name, proposed, justification, proposer_confidence)

    async def _mutate_with_fallback(
        self,
        parent: str,
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str]],
        iteration: int | str,
        diversity_hint: str = "",
    ) -> tuple[str, str, str, float] | None:
        """Try progressive truncation levels, then single-failure fallback.

        Args:
            parent: Name of the parent program.
            failures: List of (trace, agent_answer, ground_truth, category) tuples.
            iteration: Current iteration number.
            diversity_hint: Optional instruction used to diversify sibling children.

        Returns:
            Tuple of (child_name, proposal, justification, proposer_confidence)
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
            )
            if result is not None:
                return result

        # Final fallback: single failure focus (if enabled and multiple failures)
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
    def load_trajectories_from_dir(
        trajectories_dir: str | Path,
    ) -> list[tuple[Any, str, str, str, str]]:
        """Load pre-collected trajectories from a directory produced by collect_trajectories.py.

        Args:
            trajectories_dir: Directory containing ``trajectories.jsonl``.

        Returns:
            List of (AgentTrace, question, agent_answer, ground_truth, category) tuples
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
    ) -> list[tuple[AgentTrace, str, str, str, str]]:
        """Run all samples concurrently and return (trace, question, agent_answer, ground_truth, category)."""
        traces = await asyncio.gather(*[
            self.agents.base.run(q) for q, _, _ in all_data
        ])
        self._iter_cost += sum(t.total_cost_usd for t in traces)
        result = []
        for trace, (question, ground_truth, category) in zip(traces, all_data):
            agent_answer = (
                trace.output.final_answer
                if trace.output and trace.output.final_answer
                else "[PARSE FAILED]"
            )
            result.append((trace, question, agent_answer, ground_truth, category))
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
        """Choose globally connected Bradley-Terry anchors for a new child."""
        candidates: list[str] = [parent, "base"]
        best = self.manager.get_best_from_frontier()
        if best:
            candidates.append(best)
        candidates.extend(name for name, _score in self.manager.get_frontier_with_scores()[:2])

        existing = set(self.manager.list_programs())
        anchors: list[str] = []
        for name in candidates:
            if name == child_name or name not in existing or name in anchors:
                continue
            anchors.append(name)
        return anchors

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

    def _detect_judge_provider_and_model(self) -> tuple[str, str]:
        """Return (provider, model) for judge calls based on the active SDK."""
        from src.harness.sdk_config import get_sdk
        sdk = get_sdk()
        if sdk == "claude":
            return "anthropic", self.config.judge_model or "claude-haiku-4-5-20251001"
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
            return resp.choices[0].message.content or ""
        raise ValueError(f"Judge does not support provider: {provider!r}")

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
        """Map a global rating advantage over base to the score space."""
        expected_win_rate = SelfImprovingLoop._elo_expected(rating, anchor_rating, scale)
        fix_rate = max(0.0, min(1.0, 2.0 * (expected_win_rate - 0.5)))
        return base_score + fix_rate * (1.0 - base_score)

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
    def _judge_probability_to_match_score(
        probability_of_success: Any,
        skill_addresses_root_cause: Any,
    ) -> float:
        """Convert probabilistic judge estimates to an Elo match score."""
        p_success = SelfImprovingLoop._clamp01(probability_of_success, default=0.5)
        p_root = SelfImprovingLoop._clamp01(skill_addresses_root_cause, default=p_success)
        return max(0.0, min(1.0, 0.75 * p_success + 0.25 * p_root))

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
            prior=max(0.05, base_score),
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
        failures: list[tuple[AgentTrace, str, str, str, str]],
        categories: list[str],
    ) -> list[tuple[AgentTrace, str, str, str, str]]:
        """Sample a rotating category-aware batch for proposal generation."""
        if not failures:
            return []

        n_cats = len(categories)
        if n_cats == 0:
            return failures[: self.config.samples_per_category]

        n_cats_this_iter = min(self.config.categories_per_batch, n_cats)
        batch: list[tuple[AgentTrace, str, str, str, str]] = []
        for j in range(n_cats_this_iter):
            cat_idx = (self._category_offset + j) % n_cats
            cat = categories[cat_idx]
            cat_failures = [f for f in failures if f[4] == cat]
            samples_to_take = min(self.config.samples_per_category, len(cat_failures))
            offset = self._per_cat_offset.get(cat, 0)
            for k in range(samples_to_take):
                idx = (offset + k) % len(cat_failures) if cat_failures else 0
                if idx < len(cat_failures):
                    batch.append(cat_failures[idx])
            if cat_failures:
                self._per_cat_offset[cat] = offset + samples_to_take
        self._category_offset += n_cats_this_iter

        if not batch:
            return failures[: self.config.samples_per_category]
        return batch

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
        failures_ext: list[tuple[AgentTrace, str, str, str, str]],
        provider: str,
        model: str,
        child_name: str = "candidate",
        parent_name: str = "parent",
        proposal: str = "",
        justification: str = "",
        parent_skill_summary: str = "",
        candidate_skill_summary: str = "",
        skill_diff: str = "",
        candidate_skills_content: str = "",
    ) -> JudgeResult:
        """Estimate failure-recovery quality with LLM-judged Elo matches.

        Makes one lightweight API call per failure (no agent re-run).
        Returns Elo ratings plus a calibrated failure-fix estimate in [0, 1].
        """
        import json
        skills_content = candidate_skills_content or self._get_all_skills_content()
        semaphore = asyncio.Semaphore(self.config.judge_concurrency)
        child_rating = self.config.judge_elo_initial_rating
        opponent_rating = self.config.judge_elo_initial_rating
        scale = self.config.judge_elo_scale
        k_factor = self.config.judge_elo_k

        async def judge_one(index: int, trace: AgentTrace, question: str, agent_answer: str, ground_truth: str, category: str) -> dict[str, Any]:
            async with semaphore:
                trace_summary = trace.summarize(head_chars=3000, tail_chars=1500)
                prompt = build_judge_query(
                    trace_summary,
                    question,
                    agent_answer,
                    ground_truth,
                    skills_content,
                    proposal=proposal,
                    justification=justification,
                    parent_skill_summary=parent_skill_summary,
                    candidate_skill_summary=candidate_skill_summary,
                    skill_diff=skill_diff,
                )
                try:
                    text = await self._call_judge_api(prompt, provider, model)
                    text = text.strip()
                    # Strip markdown code fences if present
                    if text.startswith("```"):
                        text = "\n".join(text.split("\n")[1:])
                        text = text.rsplit("```", 1)[0].strip()
                    data = json.loads(text)
                    data["_raw_response"] = text
                    data["_index"] = index
                    data["_category"] = category
                    return data
                except Exception as e:
                    _log("WARN", f"  Judge call failed ({type(e).__name__}): {e}")
                    return {
                        "_raw_response": "",
                        "_index": index,
                        "_category": category,
                        "would_succeed": False,
                        "confidence": 0.0,
                        "hypothetical_action": "",
                        "reasoning": f"Judge call failed: {type(e).__name__}: {e}",
                        "_valid": False,
                    }

        raw_results = await asyncio.gather(*[
            judge_one(i, t, q, ans, gt, cat)
            for i, (t, q, ans, gt, cat) in enumerate(failures_ext, start=1)
        ])

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
            would_succeed = bool(data.get("would_succeed"))
            has_probability_score = (
                "probability_of_success" in data
                or "skill_addresses_root_cause" in data
            )
            if has_probability_score:
                match_score = self._judge_probability_to_match_score(
                    data.get("probability_of_success"),
                    data.get("skill_addresses_root_cause"),
                )
                would_succeed = match_score >= 0.5
            elif self.config.judge_scoring == "average":
                match_score = confidence if would_succeed else 0.0
            else:
                match_score = self._judge_binary_to_match_score(would_succeed, confidence)

            expected_before = self._elo_expected(child_rating, opponent_rating, scale)
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
                hypothetical_action=str(data.get("hypothetical_action", "")),
                reasoning=str(data.get("reasoning", "")),
                raw_response=str(data.get("_raw_response", "")),
                valid=True,
                root_cause=str(data.get("root_cause", "")),
                skill_addresses_root_cause=self._clamp01(
                    data.get("skill_addresses_root_cause"),
                    default=0.0,
                ),
                probability_of_success=self._clamp01(
                    data.get("probability_of_success"),
                    default=match_score,
                ),
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
                        f"p_success={result.probability_of_success:.3f} "
                        f"root_fit={result.skill_addresses_root_cause:.3f}"
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

        return JudgeResult(
            estimated_fix_rate=estimated_fix_rate,
            average_match_score=average_match_score,
            child_rating=child_rating,
            opponent_rating=opponent_rating,
            expected_win_rate=expected_win_rate,
            matches=matches,
        )

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
        else:
            all_data = self._get_all_data()
            _log("COLLECT", f"Running all {len(all_data)} samples upfront (no train/val split)...")
            self._iter_cost = 0.0
            extended_traces = await self._collect_all_trajectories(all_data)
            self._total_cost += self._iter_cost
            _log("COST", f"Trajectory collection: ${self._iter_cost:.4f}")

        # 3. Compute base score and partition failures
        all_failures_ext: list[tuple[AgentTrace, str, str, str, str]] = []
        passed = 0
        for trace, question, agent_answer, ground_truth, category in extended_traces:
            score = self.scorer(question, agent_answer.strip().lower(), ground_truth.strip().lower())
            if score >= 0.8:
                passed += 1
            else:
                all_failures_ext.append((trace, question, agent_answer, ground_truth, category))

        total = len(extended_traces)
        base_score = passed / total if total > 0 else 0.0
        _log(
            "COLLECT",
            f"Base score: {base_score:.4f} ({passed}/{total} passed, {len(all_failures_ext)} failures)",
        )
        _log("JUDGE", "Using each proposal batch for scoring; no fixed judge set")
        self.manager.update_frontier("base", base_score, max_size=self.config.frontier_size)
        self._emit("baseline", score=base_score, n_skills=len(self._get_active_skills()))
        search_root = self._build_program_search_tree(base_score)
        bt_matches: list[BradleyTerryMatch] = []
        bt_players: set[str] = {"base"}
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

        # 4. Evolution loop
        # Categories come from trajectories when preloaded, else from train_pools
        if self._preloaded_trajectories is not None:
            categories = sorted({entry[4] for entry in extended_traces})
        else:
            categories = sorted(self.train_pools.keys())
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

            if not all_failures_ext:
                _log("", "  -> No failures remaining")
                break

            batch_failures_ext = self._sample_proposal_failures(all_failures_ext, categories)
            scoring_failures_ext = batch_failures_ext

            # Convert to legacy format (trace, agent_answer, ground_truth, category) for _mutate
            batch_failures = [(t, ans, gt, cat) for t, _q, ans, gt, cat in batch_failures_ext]
            _log(
                "",
                (
                    f"  Using {len(batch_failures)} proposal failures; "
                    f"judging on {len(scoring_failures_ext)} failures..."
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
            any_child_added = False
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
                )

                if mutation_result is None:
                    _log("", "  [WARN] Child generation failed")
                    continue

                any_child_created = True
                child_name, proposal, justification, proposer_confidence = mutation_result
                sibling_proposals.append(proposal)
                candidate_skills_content = self._get_all_skills_content()
                parent_skill_summary = self._summarize_skill_content(parent_skills_content)
                candidate_skill_summary = self._summarize_skill_content(candidate_skills_content)
                skill_diff = self._diff_skill_content(parent_skills_content, candidate_skills_content)

                # Judge with LLM — no full agent re-run
                _log("", f"  -> Judging {child_name} via LLM ({judge_provider}/{judge_model})...")
                judge_result = await self._judge_skill_with_llm(
                    scoring_failures_ext,
                    judge_provider,
                    judge_model,
                    child_name=child_name,
                    parent_name=parent,
                    proposal=proposal,
                    justification=justification,
                    parent_skill_summary=parent_skill_summary,
                    candidate_skill_summary=candidate_skill_summary,
                    skill_diff=skill_diff,
                    candidate_skills_content=candidate_skills_content,
                )

                # Estimate total score: Elo-estimated fix rate is the fraction of
                # failed samples the child is expected to recover. Scale it to the
                # observed base-score space.
                if self.config.judge_scoring == "bradley_terry":
                    anchors = self._select_bt_anchor_nodes(parent, child_name)
                    anchor_results: list[tuple[str, JudgeResult]] = [(parent, judge_result)]
                    extra_anchors = [anchor for anchor in anchors if anchor != parent]
                    if extra_anchors:
                        _log("JUDGE", f"BT anchors for {child_name}: {', '.join(anchors)}")
                    for anchor in extra_anchors:
                        anchor_skills_content = self._get_program_skills_content(
                            anchor,
                            restore_to=child_name,
                        )
                        anchor_skill_summary = self._summarize_skill_content(anchor_skills_content)
                        anchor_diff = self._diff_skill_content(
                            anchor_skills_content,
                            candidate_skills_content,
                        )
                        _log("", f"  -> Anchor judging {child_name} vs {anchor}...")
                        anchor_result = await self._judge_skill_with_llm(
                            scoring_failures_ext,
                            judge_provider,
                            judge_model,
                            child_name=child_name,
                            parent_name=anchor,
                            proposal=proposal,
                            justification=justification,
                            parent_skill_summary=anchor_skill_summary,
                            candidate_skill_summary=candidate_skill_summary,
                            skill_diff=anchor_diff,
                            candidate_skills_content=candidate_skills_content,
                        )
                        anchor_results.append((anchor, anchor_result))

                    bt_players.update({child_name, *anchors})
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
                    bt_ratings = self._fit_bradley_terry_ratings(
                        bt_matches,
                        sorted(bt_players),
                        anchor="base",
                        initial_rating=self.config.judge_elo_initial_rating,
                        scale=self.config.judge_elo_scale,
                    )
                    base_rating = bt_ratings.get("base", self.config.judge_elo_initial_rating)
                    child_rating = bt_ratings.get(child_name, self.config.judge_elo_initial_rating)
                    child_score = self._rating_to_score(
                        child_rating,
                        base_rating,
                        base_score,
                        self.config.judge_elo_scale,
                    )
                    for rated_name, rating in bt_ratings.items():
                        if rated_name not in bt_players:
                            continue
                        rated_score = (
                            base_score
                            if rated_name == "base"
                            else self._rating_to_score(
                                rating,
                                base_rating,
                                base_score,
                                self.config.judge_elo_scale,
                            )
                        )
                        self.manager.update_frontier(
                            rated_name,
                            rated_score,
                            max_size=self.config.frontier_size,
                        )
                        self._update_puct_node_score(search_root, rated_name, rated_score)

                    expected_win = self._elo_expected(
                        child_rating,
                        base_rating,
                        self.config.judge_elo_scale,
                    )
                    _log(
                        "",
                        (
                            f"  -> Judge BT: global_rating={child_rating:.1f} "
                            f"base_rating={base_rating:.1f}, expected_vs_base={expected_win:.3f}, "
                            f"avg_match={sum(m.score for m in new_bt_matches) / len(new_bt_matches):.3f} "
                            f"matches={len(new_bt_matches)} anchors={len(anchors)} "
                            f"→ global score {child_score:.4f}"
                        ),
                    )
                else:
                    child_score = base_score + judge_result.estimated_fix_rate * (1.0 - base_score)
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

                added = self.manager.update_frontier(child_name, child_score, max_size=self.config.frontier_size)

                if added:
                    outcome = "improved" if child_score > parent_score else "kept"
                    _log("", f"  [OK] Added to frontier (score: {child_score:.4f})")
                    any_child_added = True
                else:
                    outcome = "not_frontier"
                    _log("", f"  [SKIP] Not added to frontier (score: {child_score:.4f}); kept for PUCT exploration")

                child_node = self._add_puct_child(
                    parent_node,
                    child_name,
                    child_score,
                    prior=self._proposal_policy_prior(proposer_confidence),
                    discarded=False,
                )
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
                append_feedback(
                    self._feedback_path,
                    child_name,
                    proposal,
                    justification,
                    outcome=outcome,
                    score=child_score,
                    parent_score=parent_score,
                    active_skills=self._get_active_skills(),
                )

            if any_child_added:
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
        _log("COST", f"Total cost: ${self._total_cost:.4f}")
        self._emit("loop_done", best=best or "base", best_score=best_score, iterations=iteration_count)
        return LoopResult(
            frontier=frontier,
            best_program=best or "base",
            best_score=best_score,
            iterations_completed=iteration_count,
            total_cost_usd=self._total_cost,
        )
