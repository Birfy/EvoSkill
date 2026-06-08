"""Configuration for the self-improving loop."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


EvolutionMode = Literal["prompt_only", "skill_only"]
SelectionStrategy = Literal["best", "random", "round_robin", "niche"]
JudgeScoring = Literal["elo", "average", "bradley_terry"]
FailureSelection = Literal["round_robin", "bandit"]
FrontierMode = Literal["scalar", "qd"]


@dataclass
class LoopConfig:
    """Configuration parameters for SelfImprovingLoop.

    Attributes:
        max_iterations: Maximum number of improvement iterations.
        frontier_size: Number of top-performing programs to keep.
        no_improvement_limit: Stop early after this many iterations without
            improvement. Set to 0 or None to disable early stopping and run the
            full max_iterations budget (recommended for budget-based PUCT runs
            where the best score plateaus while the tree is still exploring).
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
    no_improvement_limit: int | None = 5
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
    # Max output tokens for a direct judge completion. The judge prompt asks the
    # model to independently re-derive the earliest divergent step, verify the
    # proposer's root cause, and emit a multi-field JSON verdict; 512 tokens
    # forced it to shortcut that reasoning. Raise for higher-fidelity verdicts.
    judge_max_output_tokens: int = 1536
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
    # Asymmetric regression penalty. Breaking a case that was already correct is
    # worse than fixing a new failure is good, so when the candidate scores below
    # 0.5 on a regression case (likely broke it) the below-0.5 deviation is
    # amplified by this factor before it enters the average / Bradley-Terry match.
    # 1.0 keeps the old symmetric behavior.
    judge_regression_penalty_weight: float = 1.5
    # Number of previously-correct trajectories judged per proposal failure.
    # Regression cases are the only offline signal for whether a broadly
    # applicable skill will disturb behavior that already works.
    judge_regression_sample_multiplier: float = 2.0
    # A verifier-authored negative self-test is specifically designed to catch
    # over-triggering. If one fails, do not allow the child into the frontier.
    discard_on_negative_self_test_failure: bool = False
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
    # Penalize proposal priors when their mechanism text substantially overlaps
    # an existing active skill or an already-generated sibling proposal.
    puct_overlap_prior_penalty: float = 0.6
    puct_overlap_similarity_threshold: float = 0.35

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
    # Trigger a judge-feedback revision when the generator did not provide a
    # meaningful portable synthetic test suite for the candidate skill, or when
    # the judge estimates those tests would not actually pass.
    refine_self_test_threshold: float = 0.55
    # Execute generator-produced SKILL_TESTS.json as small synthetic agent tasks
    # against the candidate program before LLM judging. The resulting measured
    # pass rate overrides the judge's static self_test_pass_rate estimate.
    executable_skill_tests: bool = True
    executable_skill_test_max_cases: int = 4
    executable_skill_test_concurrency: int = 2
    executable_skill_test_timeout_seconds: int = 240

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

    # Skill-modification completeness guards.
    # On an EDIT, reject (restore + discard) a revision whose SKILL.md shrank
    # below this fraction of the pre-edit length, or that dropped a section
    # header the original had — catching silent content loss when the generator
    # rewrites the whole file instead of preserving still-valid content.
    skill_edit_min_retention_ratio: float = 0.5
    # Required SKILL.md section headers (see skill_generator prompt). A created or
    # edited skill that omits any of these is considered incomplete and triggers
    # one judge-feedback refine round to fill the gap. Empty list disables.
    required_skill_sections: tuple[str, ...] = (
        "When To Use",
        "Do Not Use",
        "Failure Mechanism",
        "Procedure",
        "Invariants",
        "High-Risk Operations",
        "Regression Risks",
    )
    enforce_skill_sections: bool = True
    # Active-skill budget. When the number of project skills reaches this, the
    # proposer is told to EDIT an existing skill rather than CREATE a new one, so
    # the skills directory does not bloat (every skill is read by the base agent
    # on each run, inviting over-triggering and context dilution). 0 = unlimited.
    max_active_skills: int = 8

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

    # --- Phase 3: curriculum bandit over failure categories ---------------
    # How the iteration loop picks which categories to sample failures from.
    # "round_robin" (default) cycles deterministically through categories.
    # "bandit" weights categories by recent learning gain
    # (accept_rate * mean_score_delta from feedback outcomes), so effort
    # concentrates on categories where proposals actually move the score and
    # away from ones that never improve. Falls back to round-robin until each
    # category has been tried at least once. Default-on; wired into both the
    # standard loop and the LLM-judge/PUCT loop (over failure_type or category).
    failure_selection: FailureSelection = "bandit"
    # Exploration floor for the bandit's softmax sampling (epsilon). Keeps
    # every category reachable even after it accrues a low value.
    failure_bandit_epsilon: float = 0.15
    # EMA smoothing for per-category learning-gain value updates.
    failure_bandit_ema: float = 0.5

    # --- Phase 1: ACE-style bullet playbook + helpful/harmful counters -----
    # Represent each skill's accumulated rules as an itemized playbook of
    # bullets, each carrying helpful/harmful counters. Proposals are applied as
    # small deterministic deltas (add/reinforce/retire) rather than monolithic
    # rewrites, and bullets that the held-out gate finds harmful are pruned.
    # Default-on; counters are credited in both the standard loop and the
    # LLM-judge/PUCT loop when a child improves over its parent.
    use_playbook: bool = True
    # Max bullets kept per skill playbook before lazy pruning (drops bullets
    # with harmful > helpful first, then least-helpful).
    playbook_max_bullets: int = 12
    # Cosine-similarity threshold above which an added bullet is treated as a
    # reinforcement of an existing one (dedup) instead of a new entry.
    playbook_dedup_threshold: float = 0.85

    # --- Information-isolated skill verifier (CoEvoSkills) -----------------
    # When on (and a skill_verifier agent is provided), a SEPARATE agent authors
    # each candidate skill's test cases — without seeing ground-truth answers and
    # isolated from the skill author — so the self-test signal cannot be gamed by
    # whoever wrote the skill (which is the weakness behind weak_self_test_coverage).
    # The verifier's tests overwrite the generator's SKILL_TESTS.json and feed the
    # existing self_test_pass_rate path (executed if executable_skill_tests is on,
    # otherwise estimated by the judge from the test content — no new agent rollout).
    use_skill_verifier: bool = True
    # Lower cap = fewer verifier output tokens per skill. 4 still leaves room for
    # an activation, an over-trigger negative, and a procedure-fidelity case.
    skill_verifier_max_tests: int = 4

    # --- Multi-skill proposals (token/latency control) --------------------
    # When False (default), a proposal applies only its primary skill edit even if
    # the proposer returned extra `skill_edits` — this avoids the extra generator
    # (and verifier) calls per additional skill. Turn on to let one proposal create
    # or edit several skills at once.
    enable_multi_skill: bool = False

    # --- Final frontier distillation --------------------------------------
    # After search, synthesize one compact skill from the strongest frontier
    # programs. The distilled child is returned only if executable self-tests
    # pass and the judge finds it non-regressive versus the frontier champion.
    distill_frontier_at_end: bool = True
    frontier_distill_top_k: int = 4
    frontier_distill_judge_failures: int = 4
    frontier_distill_judge_regressions: int = 4

    # --- Epoch-wise slow/meta update (SkillOpt) ---------------------------
    # Every meta_update_period iterations the LLM-judge loop consolidates the
    # accumulated per-child outcomes into a compact, PROTECTED meta summary
    # (improvements / regressions / persistent unresolved blockers / what worked),
    # written to .evoskill/meta.md and prepended to the proposer's context. The
    # consolidation is deterministic (no LLM call) so it cannot suffer the
    # "context collapse" of repeated monolithic rewrites; step-level feedback can
    # never overwrite it. Gives the proposer longitudinal memory: stop re-trying
    # fixes that never resolved a blocker, and build on skills that worked.
    meta_update_enabled: bool = True
    meta_update_period: int = 5
    # Cap how many persistent blockers / winning skills the meta lists.
    meta_top_k: int = 5

    # --- Skill induction from successful trajectories ---------------------
    # Besides fixing failures, mine successful trajectories to distill the
    # reusable procedure/decision that made them work into a new or edited skill.
    # Induced skills still pass through the same judge gate, so only ones that
    # actually help survive. Wired into the LLM-judge/PUCT loop. Default-on.
    induce_skills_from_success: bool = True
    # Long-run fraction of generated children drawn from success induction (the
    # rest from failure analysis). 0.25 ≈ 3 failure : 1 success. A fractional
    # accumulator schedules induction children so the ratio holds even when only
    # 1-2 children are generated per iteration.
    success_induction_ratio: float = 0.25
    # Prefer successes in hard categories (high failure share) — a success on a
    # mostly-failing category is the most valuable to generalize into a skill.
    success_induction_prefer_hard: bool = True

    # --- Phase 2: quality-diversity / Pareto frontier ---------------------
    # "scalar" keeps the top-k-by-average frontier. "qd" (default) keeps one
    # elite per failure-category niche (MAP-Elites-lite) plus a global best, so
    # programs that specialize on a single category survive instead of being
    # crowded out by the average-best program — preserving diversity for the
    # PUCT/parent search and complementary-skill merges. In the LLM-judge loop
    # QD applies to non-Bradley-Terry scoring (BT manages the frontier via its
    # league refit); the niche vector comes from the judge's per-category matches.
    frontier_mode: FrontierMode = "qd"
