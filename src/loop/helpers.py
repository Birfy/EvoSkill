"""Helper functions for the self-improving loop."""

from pathlib import Path
from typing import TYPE_CHECKING, Any
import re

from src.harness.opencode.skill_utils import (
    ensure_skill_frontmatter,
    normalize_project_skill_frontmatter,
)

if TYPE_CHECKING:
    from src.harness import AgentTrace
    from src.schemas import ProposerResponse, SkillProposerResponse, PromptProposerResponse


def build_proposer_query(
    traces_with_answers: list[tuple[Any, ...]],
    feedback_history: str,
    evolution_mode: str = "skill_only",
    truncation_level: int = 0,
    task_constraints: str = "",
    project_root: str | Path | None = None,
    diversity_hint: str = "",
    questions: list[str] | None = None,
    domain_hints: dict[str, list[str]] | None = None,
) -> str:
    """Build the query for the proposer agent from multiple failure traces.

    Args:
        traces_with_answers: List of (trace, agent_answer, ground_truth, category) tuples.
        feedback_history: Previous feedback history.
        evolution_mode: "skill_only" or "prompt_only" - affects trace truncation.
        truncation_level: Context reduction level (0=full, 1=moderate, 2=aggressive).
        task_constraints: Optional task-specific constraints to include in the query.
        diversity_hint: Optional instruction to make sibling child proposals differ.

    Returns:
        Formatted query string for the proposer.
    """
    # Truncation level settings: (head_chars, tail_chars, feedback_lines, max_failures)
    TRUNCATION_SETTINGS = [
        (60_000, 60_000, None, None),    # Level 0: full
        (20_000, 10_000, 20, 3),         # Level 1: moderate
        (5_000, 2_000, 5, 2),            # Level 2: aggressive
    ]
    head_chars, tail_chars, feedback_lines, max_failures = TRUNCATION_SETTINGS[
        min(truncation_level, len(TRUNCATION_SETTINGS) - 1)
    ]

    # Apply max_failures limit
    if max_failures is not None and len(traces_with_answers) > max_failures:
        traces_with_answers = traces_with_answers[:max_failures]

    # Apply feedback truncation
    if feedback_lines is not None:
        feedback_lines_list = feedback_history.split("\n")
        if len(feedback_lines_list) > feedback_lines:
            feedback_history = "\n".join(feedback_lines_list[-feedback_lines:])

    # Get existing skills for context
    skills_dir = Path(project_root) / ".claude" / "skills" if project_root else Path(".claude/skills")
    existing_skills = []
    if skills_dir.exists():
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                existing_skills.append(skill_dir.name)
    skills_list = "\n".join([f"- {s}" for s in existing_skills]) or "None"

    # Collect categories for summary
    categories = [failure[3] for failure in traces_with_answers if len(failure) > 3]
    category_summary = ", ".join(sorted(set(categories)))

    # Build failure summaries with truncation-level-aware settings
    failure_sections = []
    for i, failure in enumerate(traces_with_answers, 1):
        trace, agent_answer, ground_truth, category = failure[:4]
        feedback = str(failure[4] or "").strip() if len(failure) > 4 else ""
        # For prompt mode, use more aggressive truncation to focus on patterns
        # For skill mode, keep full trace to see tool usage (but respect truncation level)
        if evolution_mode == "prompt_only":
            # Prompt mode uses tighter truncation even at level 0
            effective_head = min(head_chars, 20_000)
            effective_tail = min(tail_chars, 10_000)
        else:
            effective_head = head_chars
            effective_tail = tail_chars

        trace_summary = trace.summarize(head_chars=effective_head, tail_chars=effective_tail)

        question_line = f"\nQuestion: {questions[i - 1]}" if questions and i - 1 < len(questions) else ""
        if not feedback:
            feedback = build_answer_comparison_feedback(
                questions[i - 1] if questions and i - 1 < len(questions) else "",
                str(agent_answer),
                str(ground_truth),
                domain_hints=domain_hints,
            )

        feedback_section = f"\nStructured Failure Feedback:\n{feedback}\n" if feedback else ""

        # NOTE: the raw agent answer and the ground-truth answer are deliberately
        # NOT printed here. Leaking them lets the skill author memorize per-task
        # answers (verbatim "Answer: X / Wrong: Y" worked examples) instead of a
        # general procedure. Only the structured, non-oracle failure feedback below
        # (error shape, likely surfaces) is exposed.
        failure_sections.append(f"""### Failure {i} [Category: {category}]{question_line}
{trace_summary}
{feedback_section}
""")

    failures_text = "\n".join(failure_sections)

    constraints_section = f"\n## Task Constraints\n{task_constraints}\n" if task_constraints else ""
    diversity_section = (
        f"\n## Diversity Requirement\n{diversity_hint.strip()}\n"
        if diversity_hint.strip()
        else ""
    )
    existing_skill_rule = (
        "There are no existing project skills. You MUST propose action=\"create\" "
        "and leave target_skill null/empty. Do not edit or reference non-project "
        "skills such as brainstorming or framework/system skills."
        if not existing_skills
        else (
            "You may propose action=\"edit\" only for one of the exact skill names "
            "listed above. If no listed skill directly applies, propose action=\"create\"."
        )
    )

    return f"""## Existing Skills (check before proposing new ones)
{skills_list}

Existing skill rule: {existing_skill_rule}
{constraints_section}
{diversity_section}
## Previous Attempts Feedback
{feedback_history}

## Current Failures ({len(traces_with_answers)} samples across categories: {category_summary})

Analyze the patterns across these failures to identify a GENERAL improvement, not a fix for any single case.

{failures_text}

## Your Task
1. For each failure, name the earliest divergent source/table/row/column/formula/unit/rounding step (not just the final answer).
2. Propose ONE general, reusable fix for the shared root cause; keep distinct failure modes separate and state any uncovered failure in coverage_plan.
3. Prefer EDITING a listed project skill when one should have caught these; otherwise CREATE. Note how your proposal differs from any DISCARDED iteration.
4. Fill all required fields with concrete details: root_cause_analysis, coverage_plan, should_apply_when, should_not_apply_when, invariants_to_preserve, regression_risks."""


_DEFAULT_ERROR_SURFACE_HINTS: dict[str, list[str]] = {
    "statistic definition, population/sample basis, complete time-point subset, rounding": [
        "standard deviation", "variance", "mean", "average",
    ],
    "explicit category set, inequality strictness, per-period count then aggregation": [
        "how many", "categories", "threshold", "more than", "less than", "at least",
    ],
    "contributor inclusion/exclusion, rollup overlap, missing/extra rows, unit scale": [
        "sum", "total", "aggregate",
    ],
    "semantic row-label grounding and correct group-by dimension": [
        "labeled", "classified by", "first callable", "grouped by",
    ],
    "model feature/time index, predicted-vs-actual binding, absolute-difference formula": [
        "ols", "predicted", "actual", "absolute difference", "model",
    ],
    "formula definition, period length, percent/fraction scaling, output vector order": [
        "cagr", "elasticity", "decay", "growth rate", "logarithmic",
    ],
}


def build_regression_success_feedback(
    question: str,
    agent_answer: str,
    ground_truth: str,
    domain_hints: dict[str, list[str]] | None = None,
) -> str:
    """Create specific feedback for a regression case where the agent succeeded.

    Tells the judge which computation surfaces were handled correctly and must
    be preserved by the candidate skill set.
    """
    pred_nums = _extract_numbers(agent_answer)
    exp_nums = _extract_numbers(ground_truth)
    # Answer-blind: state that this case was already correct and which surfaces to
    # preserve, without printing the (correct) answer value itself.
    lines = [
        "- regression_case: trajectory already produced the correct answer",
    ]
    if pred_nums and exp_nums and pred_nums == exp_nums:
        lines.append("- match_type: exact numeric match")
    elif agent_answer.strip().lower() == ground_truth.strip().lower():
        lines.append("- match_type: exact string match")
    else:
        lines.append("- match_type: tolerance/fuzzy match passed")

    preserved = _likely_error_surfaces(
        question.lower(), agent_answer, ground_truth, pred_nums, exp_nums,
        domain_hints=domain_hints,
    )
    lines.append("- preserved_surfaces: " + "; ".join(preserved))
    lines.append(
        "- preservation_requirement: candidate skills must NOT alter the "
        "source selection, operator, units, rounding, or aggregation method "
        "used to reach this answer."
    )
    return "\n".join(lines)


def build_answer_comparison_feedback(
    question: str,
    predicted: str,
    expected: str,
    failure_type: str = "",
    domain_hints: dict[str, list[str]] | None = None,
) -> str:
    """Create structured, non-oracle comparison feedback for proposer prompts.

    Deliberately answer-blind: it never emits the predicted answer, the expected
    (ground-truth) answer, or any signed delta/ratio that would let the skill
    author reconstruct the answer (predicted ± delta = expected). It exposes only
    the *shape* of the error (arity, scale-factor, close/large bucket) and the
    likely verification surfaces, so the proposer learns a general procedure
    instead of memorizing per-task answers.
    """
    pred_nums = _extract_numbers(predicted)
    exp_nums = _extract_numbers(expected)
    q = question.lower()
    lines = []
    if failure_type:
        lines.append(f"- failure_type: {failure_type}")

    if pred_nums and exp_nums:
        pairs = list(zip(pred_nums, exp_nums))
        abs_diffs = [abs(p - e) for p, e in pairs]
        ratios = [
            (p / e)
            for p, e in pairs
            if e not in (0.0, -0.0)
        ]
        lines.append(
            "- mismatch_shape: "
            + _numeric_mismatch_shape(abs_diffs, ratios, len(pred_nums), len(exp_nums))
        )
    else:
        lines.append("- mismatch_shape: non_numeric_or_unparsed_answer_mismatch")

    likely = _likely_error_surfaces(q, predicted, expected, pred_nums, exp_nums, domain_hints=domain_hints)
    lines.append("- likely_error_surfaces: " + "; ".join(likely))
    lines.append(
        "- first_divergence_to_inspect: compare the trajectory's extracted rows/columns, "
        "unit conversions, formula choice, aggregation set, and final rounding against the "
        "structured mismatch labels above; do not assume final-answer formatting is the root cause."
    )
    lines.append(
        "- required_proposer_fix: propose a reusable verification gate that would force the agent "
        "to expose and check the earliest divergent intermediate. Do NOT hardcode any specific "
        "answer or wrong value into the skill; worked examples must use general/synthetic placeholders."
    )
    return "\n".join(lines)


def _extract_numbers(text: str) -> list[float]:
    nums: list[float] = []
    normalized = text.replace(",", "").replace("−", "-")
    for match in re.finditer(r"-?\d+(?:\.\d+)?", normalized):
        try:
            nums.append(float(match.group(0)))
        except ValueError:
            pass
    return nums


def _numeric_mismatch_shape(
    abs_diffs: list[float],
    ratios: list[float],
    n_pred: int,
    n_exp: int,
) -> str:
    # Emit only categorical labels. Raw deltas/ratios are withheld: combined with
    # the (also-withheld) predicted answer they would reconstruct the ground truth.
    labels: list[str] = []
    if n_pred != n_exp:
        labels.append(f"numeric_arity_mismatch(pred={n_pred}, expected={n_exp})")
    if ratios:
        max_ratio_gap = max(abs(r - 1.0) for r in ratios)
        if any(abs(abs(r) - 10.0) < 0.5 or abs(abs(r) - 100.0) < 5.0 for r in ratios):
            labels.append("possible_scale_factor_error")
        elif max_ratio_gap < 0.02:
            labels.append("close_numeric_mismatch")
        elif max_ratio_gap > 0.5:
            labels.append("large_numeric_mismatch")
        else:
            labels.append("moderate_numeric_mismatch")
    elif abs_diffs:
        labels.append("numeric_mismatch")
    return ", ".join(labels) if labels else "numeric_mismatch"


def _likely_error_surfaces(
    question_lc: str,
    predicted: str,
    expected: str,
    pred_nums: list[float],
    exp_nums: list[float],
    domain_hints: dict[str, list[str]] | None = None,
) -> list[str]:
    surfaces: list[str] = []
    hints = domain_hints if domain_hints is not None else _DEFAULT_ERROR_SURFACE_HINTS
    for surface, keywords in hints.items():
        if any(kw in question_lc for kw in keywords):
            surfaces.append(surface)
    if pred_nums and exp_nums and len(pred_nums) == len(exp_nums):
        deltas = [abs(p - e) for p, e in zip(pred_nums, exp_nums)]
        if deltas and max(deltas) <= 1.0:
            surfaces.append("small numeric drift: rounding stage, interpolation method, boundary value")
    if not surfaces:
        surfaces.append("answer parsing/formatting or unclassified semantic mismatch")
    return surfaces


def build_skill_query(proposer_trace: "AgentTrace[ProposerResponse]") -> str:
    """Build the query for the skill generator agent.

    Args:
        proposer_trace: The trace from the proposer agent.

    Returns:
        Formatted query string for the skill generator.
    """
    return f"""Proposed tool or skill (high level description): {proposer_trace.output.proposed_skill_or_prompt}

Justification: {proposer_trace.output.justification}"""


def build_prompt_query(
    proposer_trace: "AgentTrace[ProposerResponse]", original_prompt: str
) -> str:
    """Build the query for the prompt generator agent.

    Args:
        proposer_trace: The trace from the proposer agent.
        original_prompt: The original system prompt to optimize.

    Returns:
        Formatted query string for the prompt generator.
    """
    return f"""## Original Prompt
{original_prompt}

## Proposed Change
{proposer_trace.output.proposed_skill_or_prompt}

## Justification
{proposer_trace.output.justification}"""


def append_feedback(
    path: Path,
    iteration: str,
    proposal: str,
    justification: str,
    outcome: str | None = None,
    score: float | None = None,
    parent_score: float | None = None,
    active_skills: list[str] | None = None,
    failure_category: str | None = None,
    root_cause: str | None = None,
    remaining_blockers: list[str] | None = None,
) -> None:
    """Append feedback entry to history file with outcome tracking.

    Args:
        path: Path to the feedback history file.
        iteration: Iteration identifier (e.g., "iter-1").
        proposal: The skill or prompt that was proposed.
        justification: Why this change was proposed.
        outcome: "improved", "no_improvement", or "discarded".
        score: The score achieved after applying this proposal.
        parent_score: The parent's score before this proposal.
        active_skills: List of skills that were active during evaluation.
        failure_category: Category of failure (e.g., "methodology", "formatting").
        root_cause: Brief description of root cause (from the judge when available).
        remaining_blockers: Judge-identified blockers the proposal did NOT resolve,
            so the next proposer knows what still needs fixing instead of only a score.
    """
    # Build outcome section if available
    outcome_section = ""
    if outcome is not None:
        delta = (score - parent_score) if (score is not None and parent_score is not None) else None
        delta_str = f" ({delta:+.4f})" if delta is not None else ""
        score_str = f" (score: {score:.4f}{delta_str})" if score is not None else ""
        outcome_section = f"\n**Outcome**: {outcome.upper()}{score_str}"

    # Build diagnostic section
    diagnostic_section = ""
    if active_skills:
        diagnostic_section += f"\n**Active Skills**: {', '.join(active_skills)}"
    if failure_category:
        diagnostic_section += f"\n**Failure Category**: {failure_category}"
    if root_cause:
        diagnostic_section += f"\n**Root Cause**: {root_cause}"
    if remaining_blockers:
        blockers = "; ".join(b.strip() for b in remaining_blockers if str(b).strip())
        if blockers:
            diagnostic_section += f"\n**Remaining Blockers (judge)**: {blockers}"

    entry = f"""
## {iteration}
**Proposal**: {proposal}
**Justification**: {justification}{outcome_section}{diagnostic_section}

"""
    with open(path, "a") as f:
        f.write(entry)


def read_feedback_history(path: Path) -> str:
    """Read feedback history or return default message.

    Args:
        path: Path to the feedback history file.

    Returns:
        Contents of feedback file or default message.
    """
    if path.exists():
        return path.read_text()
    return "No previous attempts."


def update_prompt_file(file_path: Path, new_prompt: str) -> None:
    """Write the new prompt to prompt.txt.

    The Agent reads this file at runtime on each run().

    Args:
        file_path: Path to the prompt file.
        new_prompt: The new prompt content.
    """
    file_path.write_text(new_prompt.strip())


def build_skill_query_from_skill_proposer(
    proposer_trace: "AgentTrace[SkillProposerResponse]",
) -> str:
    """Build the query for the skill generator from a skill proposer trace.

    Args:
        proposer_trace: The trace from the skill proposer agent.

    Returns:
        Formatted query string for the skill generator.
    """
    root_cause_analysis = getattr(proposer_trace.output, "root_cause_analysis", "") or "[not provided]"
    coverage_plan = getattr(proposer_trace.output, "coverage_plan", "") or "[not provided]"
    should_apply_when = getattr(proposer_trace.output, "should_apply_when", "") or "[not provided]"
    should_not_apply_when = getattr(proposer_trace.output, "should_not_apply_when", "") or "[not provided]"
    invariants_to_preserve = getattr(proposer_trace.output, "invariants_to_preserve", "") or "[not provided]"
    regression_risks = getattr(proposer_trace.output, "regression_risks", "") or "[not provided]"

    return f"""Proposed tool or skill (high level description): {proposer_trace.output.proposed_skill}

Justification: {proposer_trace.output.justification}

Root cause analysis from proposer:
{root_cause_analysis}

Failure coverage plan:
{coverage_plan}

Use this skill only when:
{should_apply_when}

Do not use this skill when:
{should_not_apply_when}

Invariants to preserve:
{invariants_to_preserve}

Regression risks / anti-regression guards:
{regression_risks}"""


def build_judge_query(
    trace_summary: str,
    question: str,
    agent_answer: str,
    ground_truth: str,
    parent_skill_summary: str = "",
    candidate_skill_summary: str = "",
    skill_diff: str = "",
    case_type: str = "failure",
    case_feedback: str = "",
    proposer_root_cause: str = "",
) -> str:
    """Build the prompt for the LLM judge.

    The judge sees the concrete artifact (skill diff + summaries), the case, and
    the proposer's claimed root cause. It must do two distinct things: (1) decide
    whether the proposer's root-cause diagnosis is actually correct for this case
    (so a misdiagnosis is caught rather than rubber-stamped), and (2) decide
    whether the candidate diff concretely enforces a fix for that root cause. The
    judge is NOT given the proposer's justification or coverage plan — only the
    one-line root cause — so it cannot simply echo the author's reasoning.

    Returns:
        A prompt string for the judge LLM.
    """
    is_regression = case_type == "regression"
    answer_label = "Agent's Prior Correct Answer" if is_regression else "Agent's Wrong Answer"
    case_intro = (
        "On this case the original skills already produced the correct answer."
        if is_regression
        else "On this case the original skills previously produced a wrong answer."
    )
    direction = (
        "One set includes the Change Being Evaluated and the other does not — work out "
        "which from the summaries. If that change risks breaking this currently-correct "
        "case (picks a different source/row, changes units/rounding, adds an override), "
        "the including set should score clearly LOWER."
        if is_regression
        else "One set includes the Change Being Evaluated and the other does not — work out "
        "which from the summaries. If that change concretely fixes the true cause of this "
        "failure, the including set should score clearly HIGHER; if it is cosmetic, "
        "off-target, or risky, it should NOT score higher."
    )
    feedback_section = f"\n## Case Feedback\n{case_feedback}\n" if case_feedback else ""
    proposer_rc_section = (
        f"\n## Proposer's Claimed Root Cause (verify — do not assume correct)\n{proposer_root_cause}\n"
        if proposer_root_cause and proposer_root_cause.strip()
        else ""
    )

    return f"""You are a careful, decisive reviewer comparing two skill sets on ONE case.
Two skill sets are shown below as "Skill Set A" and "Skill Set B". The A/B
assignment is randomized: neither label implies which set is older, newer, or
proposed — judge purely on content. They differ only by the single "Change Being
Evaluated" shown afterward, which one set includes and the other does not.

{case_intro}
Judge ONLY from the concrete skill content and the change shown; you are NOT given
any author's rationale. Estimate each set's probability of reaching the expected
answer on THIS case, then give a pairwise score for Set B over Set A.

Be decisive: commit to whichever set is more likely to reach the expected answer.
Use 0.5 ONLY when the two sets are genuinely indistinguishable on this case — never
as a safe default. A small but real difference must move the score off 0.5. {direction}

Do not reward polished but generic skill text. A useful skill is defined by three
case-grounded dimensions:
1. failure_mechanism_encoding: does the change name the domain-specific failure
   mechanism that caused THIS case, not just a broad category like "verification"?
2. executable_specificity: does it force concrete operations/checks that would
   change the agent's behavior on THIS case, rather than advice like "be careful"?
3. high_risk_blacklist: does it explicitly forbid tempting operations that would
   cause the same failure or regressions on similar cases?

The pairwise score must reflect those dimensions. If the Change Being Evaluated
lacks the true failure mechanism or an executable remedy, do NOT give it a high
B-over-A score merely because it is well written. If it introduces a risky
operation without blacklisting it, lower the including set on regression cases
and avoid credit on failure cases.

Also judge transferability. The skill should abstract from the observed case to
the reusable failure mechanism. Penalize changes that look strong only because
they name this case's surface details (exact files, dates/times, labels, entity
names, paths, API objects, UI elements, answer values, or a single operation) in
activation rules or invariants. Worked examples may contain concrete values, but
the rule itself must help a new task with different inputs, files, entities,
tools, environments, formats, constants, and neighboring operations from the same
failure family. Score generalization_transfer high only when the change is
specific about the mechanism while remaining portable across such variants.

First derive, independently from the trajectory and the expected answer, the
earliest divergent step that actually caused this case to fail. Then judge the
proposer's claimed root cause against your own finding: set
proposer_root_cause_correct high only if the proposer located the same earliest
divergence; set it low when the proposer misdiagnosed (e.g. blamed a formula
when the real error was reading the wrong row/table, or the wrong percent
denominator convention). Then judge skill_addresses_root_cause: how concretely
the Change Being Evaluated enforces a fix for the TRUE root cause — not merely
whether it adds plausible-looking structure. A change that builds elaborate
contracts but does not constrain the actual divergent step scores low here.

## Question
{question}

## Expected Answer
{ground_truth}

## {answer_label}
{agent_answer}

## Prior Trajectory (summary)
{trace_summary}
{feedback_section}{proposer_rc_section}
## Skill Set A
{parent_skill_summary or "[not provided]"}

## Skill Set B
{candidate_skill_summary or "[not provided]"}

## Change Being Evaluated (present in exactly one of the two sets above)
{skill_diff or "[not provided]"}

Respond ONLY with a JSON object (no markdown fences):
{{
  "root_cause": "<the earliest divergent step you independently identified>",
  "proposer_root_cause_correct": <0.0-1.0 how well the proposer's claimed root cause matches yours>,
  "skill_addresses_root_cause": <0.0-1.0 how concretely the Change Being Evaluated fixes the TRUE root cause>,
  "failure_mechanism_encoding": <0.0-1.0 whether the change encodes the concrete domain-specific failure mechanism>,
  "executable_specificity": <0.0-1.0 whether the change gives concrete operations/checks that would alter behavior>,
  "high_risk_blacklist": <0.0-1.0 whether the change forbids likely harmful operations/regressions>,
  "generalization_transfer": <0.0-1.0 whether the change abstracts the mechanism beyond this exact case without becoming vague>,
  "set_a_success_prob": <0.0-1.0 probability Skill Set A reaches the expected answer>,
  "set_b_success_prob": <0.0-1.0 probability Skill Set B reaches the expected answer>,
  "b_over_a_score": <0.0-1.0 pairwise score for Set B over Set A; 0.5 ONLY if truly indistinguishable>,
  "confidence": <0.0-1.0 confidence in your estimate>,
  "remaining_blockers": ["<short blocker the change still does not fix>"],
  "reasoning": "<one sentence>"
}}"""


def build_skill_revision_query(
    target_skill: str,
    current_skill_markdown: str,
    proposer_root_cause: str,
    original_proposal: str,
    judge_root_cause: str,
    judge_blockers: list[str],
    judge_reasoning: str = "",
) -> str:
    """Build the generator query for one judge-driven revision of a child skill.

    The generator is shown two clearly separated signals: the proposer's ORIGINAL
    intent (what the skill tried to fix) and the JUDGE's independent verdict on
    why that fix falls short (the correction direction). The generator must keep
    the skill's valid structure and make a targeted change toward the judge's
    blockers — not rewrite from scratch.
    """
    blockers_text = "\n".join(f"- {b}" for b in judge_blockers) or "- (none stated)"
    return f"""REVISE existing skill: {target_skill}

A skeptical judge reviewed this skill against held-out failures and concluded it
does not yet solve the failure's true root cause. Revise the skill with ONE
targeted change. Keep all still-valid content (triggers, invariants, correct
worked examples); do not rewrite from scratch.

## Original intent (proposer) — what this skill tried to fix
{proposer_root_cause or original_proposal or "[not provided]"}

## Judge's independent verdict — the correction direction (treat as authoritative)
Judge's identified true root cause:
{judge_root_cause or "[not provided]"}

Why the current skill still falls short (unresolved blockers):
{blockers_text}

Judge note: {judge_reasoning or "[none]"}

## Current SKILL.md (revise this)
{current_skill_markdown}

Produce the complete revised SKILL.md. Focus the change on enforcing a fix for
the judge's true root cause (e.g. add/repair the protocol-block field or gate
that pins the actual divergent step), and ensure worked examples show the real
final answer rather than any intermediate scaffold value. If the judge's
blockers indicate low transferability or case-specific overfitting, revise by
lifting exact files, dates/times, labels, entity names, paths, answer values,
one-off operations, and environment-specific constants out of the main rules into
a reusable failure-family contract."""


def build_prompt_query_from_prompt_proposer(
    proposer_trace: "AgentTrace[PromptProposerResponse]",
    original_prompt: str,
) -> str:
    """Build the query for the prompt generator from a prompt proposer trace.

    Args:
        proposer_trace: The trace from the prompt proposer agent.
        original_prompt: The original system prompt to optimize.

    Returns:
        Formatted query string for the prompt generator.
    """
    return f"""## Original Prompt
{original_prompt}

## Proposed Change
{proposer_trace.output.proposed_prompt_change}

## Justification
{proposer_trace.output.justification}"""
