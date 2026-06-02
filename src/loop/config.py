"""Configuration for the self-improving loop."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


EvolutionMode = Literal["prompt_only", "skill_only"]
SelectionStrategy = Literal["best", "random", "round_robin"]
JudgeScoring = Literal["elo", "average", "bradley_terry"]


@dataclass
class LoopConfig:
    """Configuration parameters for SelfImprovingLoop.

    Attributes:
        max_iterations: Maximum number of improvement iterations.
        frontier_size: Number of top-performing programs to keep.
        no_improvement_limit: Stop early after this many iterations without improvement.
        tolerance: Tolerance for answer matching (0.0 = exact match).
        concurrency: Number of concurrent evaluations.
        evolution_mode: Which dimension to evolve ("prompt_only" or "skill_only").
        selection_strategy: Parent selection from frontier — "best" (greedy, default),
            "random" (uniform random), or "round_robin" (cycle through ranked members).
        reset_feedback: Whether to reset feedback_history.md on fresh loop run.
        cache_enabled: Whether to enable run caching.
        cache_dir: Directory for cache storage.
        cache_store_messages: Whether to store full message history in cache.
    """

    max_iterations: int = 5
    frontier_size: int = 3
    no_improvement_limit: int = 5
    tolerance: float = 0.0
    concurrency: int = 4

    # Evolution mode: which dimension to optimize
    evolution_mode: EvolutionMode = "skill_only"

    # Parent selection strategy: how to pick the next parent from the frontier
    selection_strategy: SelectionStrategy = "best"

    # Multi-sample failure analysis: test this many samples before proposing
    # Helps identify patterns rather than overfitting to single failures
    failure_sample_count: int = 2

    # Child scoring (standard, non-judge path): blend "did the candidate fix the
    # in-sample failures it was proposed for" with "does it generalize to unseen
    # held-out samples". child_score = w * in_sample_fix_rate + (1-w) * generalization_rate.
    # w=1.0 reproduces the old pure-in-sample behavior (max overfitting); lower w
    # rewards generalization. The held-out term uses val_data when present, else a
    # slice of the training pools the proposer never saw this iteration.
    in_sample_score_weight: float = 0.5
    # Max held-out samples drawn from the training pools for the generalization
    # term when no dedicated val_data split exists. Bounds extra eval cost.
    generalization_sample_count: int = 4

    # Category-aware sampling: number of categories to sample per batch
    # (capped by actual number of categories and failure_sample_count)
    categories_per_batch: int = 2

    # Feedback configuration
    reset_feedback: bool = True

    # Continue mode: False = start fresh (reset iteration numbering),
    # True = continue from existing frontier/branch
    continue_mode: bool = False

    # Cache configuration
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".cache/runs"))
    cache_store_messages: bool = False

    # Proposer resilience: adaptive truncation on context limit/timeout
    proposer_max_truncation_level: int = 2  # Max truncation level (0=full, 1=moderate, 2=aggressive)
    proposer_single_failure_fallback: bool = True  # Try single shortest failure if all levels fail
    consecutive_proposer_failures_limit: int = 5  # Stop after N consecutive proposer failures

    # Multi-sample per category: collect N samples per category before proposing.
    # This is the dominant driver of proposer prompt size (failures shown per
    # call = categories_per_batch * samples_per_category). Kept at 1 so a single
    # proposer call stays small enough for transports with tight per-line limits
    # (e.g. the codex SDK's aiohttp reader); raise it for richer pattern signal.
    samples_per_category: int = 1  # Helps identify patterns within categories

    # LLM Judge mode: run all trajectories once upfront (no train/val split),
    # then use a lightweight direct API call to judge skill changes instead of
    # re-running the full agent on a validation set.
    use_llm_judge: bool = False
    # Model for judge calls. None = auto-pick a small model for the active SDK
    # (Anthropic → claude-haiku-4-5-20251001, OpenAI → gpt-4o-mini).
    judge_model: str | None = None
    judge_concurrency: int = 8
    judge_call_timeout_seconds: int = 180
    # Judge scoring mode. "elo" treats each judged failure as a pairwise match
    # between the child program and the baseline/parent behavior.
    # "bradley_terry" stores all pairwise match records in one global league
    # and refits ratings after each child so node scores are comparable.
    # "average" keeps the older direct mean of fixed-failure confidences.
    judge_scoring: JudgeScoring = "elo"
    judge_elo_initial_rating: float = 1500.0
    judge_elo_k: float = 128.0
    judge_elo_scale: float = 400.0
    # Position de-bias. The candidate would otherwise always sit in a fixed slot
    # ("B"/candidate) against the incumbent, so any judge slot/label preference
    # biases every child systematically. With judge_randomize_orientation (default)
    # each case randomly assigns the candidate to Skill Set A or B in a SINGLE
    # call, and the canonical candidate-vs-parent score is recovered by un-swapping
    # — zero extra cost, turning a systematic bias into zero-mean noise.
    # judge_position_swap instead judges BOTH orientations and averages (2x calls);
    # it takes precedence when True. Set both False for fixed-slot judging.
    judge_randomize_orientation: bool = True
    judge_position_swap: bool = False
    # Avoid self-enhancement bias: never let the judge run on the same model that
    # authored the skill (the generator). When the resolved judge model matches
    # the generator model, fall back to a distinct model for the provider.
    judge_distinct_from_generator: bool = True
    # Penalize Bradley-Terry scores for sparse/low-confidence evidence. This
    # keeps frontier selection from over-trusting nodes with only a few judged
    # comparisons or uncertain judge outputs.
    judge_bt_uncertainty_penalty: float = 0.05
    judge_log_details: bool = True
    # Hold out a fraction of failures (per category) so the judge scores
    # candidates ONLY on cases the proposer never saw. This prevents the
    # proposer from memorizing the exact items it is graded on. 0.0 disables
    # the split (proposer and judge share all failures, the old behavior).
    judge_holdout_ratio: float = 0.5
    # Optional direct-judge price overrides in USD per 1M tokens. Direct API
    # responses expose token usage but not provider billing; when these are not
    # set, known OpenAI model prices are used for estimation.
    judge_input_cost_per_1m: float | None = None
    judge_output_cost_per_1m: float | None = None
    # PUCT tree search for LLM-judge evolution mode.
    puct_c: float = 0.5
    puct_max_depth: int = 3
    puct_children_per_node: int = 2
    children_per_iteration: int = 2
    puct_default_prior: float = 0.5

    # judge→generator refine loop. After the judge scores a child, if the
    # candidate is judged NOT to address the failure's root cause (mean
    # skill_addresses_root_cause below refine_root_cause_threshold) or it does
    # not beat its parent, feed the judge's independent root-cause verdict and
    # remaining blockers back to the generator for one targeted revision, then
    # re-judge. The better-scoring version (original vs revised) is kept, so a
    # refine can never lower a child's score. Set False to disable the loop.
    refine_with_judge_feedback: bool = True
    refine_max_rounds: int = 1
    refine_root_cause_threshold: float = 0.5
    # Trigger a judge-feedback revision when the change appears to memorize
    # sampled case details instead of transferring the mechanism to new inputs,
    # files, entities, tools, environments, formats, constants, or neighboring
    # operations from the same failure family.
    refine_generalization_threshold: float = 0.55

    # Champion-gated staged dueling (Bradley-Terry mode). A new child first duels
    # the current frontier champion. If it clearly loses (the candidate-vs-champion
    # paired match-score mean is below 0.5 with confidence) it is eliminated after
    # that single duel — no further comparisons. If the duel is close, the child is
    # additionally compared against judge_stage2_anchors random frontier members to
    # disambiguate before the Bradley-Terry refit. A clear win skips the extra PKs.
    # This concentrates judge calls on genuine contenders. Set False to restore the
    # fixed parent+frontier-best anchor set (_select_bt_anchor_nodes).
    judge_champion_gate: bool = True
    judge_stage2_anchors: int = 2
    # Confidence level (delta) for the empirical-Bernstein CI used to decide
    # win/loss/close in a champion duel. Smaller = stricter (needs more separation).
    judge_duel_delta: float = 0.1
    # Minimum valid cases before a duel can be decided win/loss; fewer is "close".
    judge_duel_min_cases: int = 2

    # Judge scoring set composition. When False (default) the judge scores each
    # child only on the current batch's held-out failures plus a small regression
    # sample, instead of also re-judging every failure inherited from the parent
    # node. This is the dominant judge-call multiplier; keeping it off cuts cost
    # sharply. Set True to restore the inherited-superset behavior.
    judge_inherit_parent_failures: bool = False

    # Validate generator-produced worked examples: reject (and trigger one
    # regeneration) when an example's scaffold count (e.g. ledger_count_check,
    # "expected N tests", "rows x periods") is presented as a value that equals
    # the example's stated final answer — the failure mode where the agent emits
    # the grid size instead of the predicate-satisfying subset count.
    validate_worked_examples: bool = True

    # Failure detection threshold used in standard (non-judge) mode.
    # A sample with score below this value is treated as a failure and
    # forwarded to the proposer.  Decoupled from tolerance so both can
    # be tuned independently.
    failure_threshold: float = 0.8

    # Domain-specific keyword→surface mapping for answer-comparison feedback.
    # Keys are surface-description strings; values are lists of question
    # keywords that trigger them.  None → use the built-in defaults defined
    # in helpers._DEFAULT_ERROR_SURFACE_HINTS.
    error_surface_hints: dict | None = None
