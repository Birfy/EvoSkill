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

        failure_sections.append(f"""### Failure {i} [Category: {category}]{question_line}
{trace_summary}

Agent Answer: {agent_answer}
Ground Truth: {ground_truth}
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
1. Build a per-failure root-cause matrix before choosing a proposal.
   - For each failure, identify the earliest divergent source/table/row/column/formula/unit/rounding step.
   - Use the provided predicted-vs-expected comparison to locate the first wrong intermediate, not just the final answer.
2. Identify the shared reusable capability gap, but keep distinct failure modes separate.
   - Do not hide uncovered cases behind a generic "verify more carefully" proposal.
   - If one sampled failure is not covered by your proposal, say that explicitly in coverage_plan.
3. Check if any EXISTING skill should have handled these failures.
4. If yes → propose EDITING that listed project skill only (action="edit", target_skill="skill-name").
5. If no → propose a NEW skill (action="create").
6. Reference any related DISCARDED iterations and explain how your proposal differs.
7. Include regression-risk analysis: how this proposal could degrade previously correct tasks, and what guard prevents that.
8. Fill root_cause_analysis, coverage_plan, and regression_risks with concrete details."""


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
    lines = [
        "- regression_case: trajectory already produced the correct answer",
        f"- correct_answer: {agent_answer}",
        f"- expected_answer: {ground_truth}",
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

    This does not invent hidden intermediate values.  It makes the known answer
    delta explicit and lists the most likely verification surfaces the proposer
    should inspect in the trajectory.
    """
    pred_nums = _extract_numbers(predicted)
    exp_nums = _extract_numbers(expected)
    q = question.lower()
    lines = []
    if failure_type:
        lines.append(f"- failure_type: {failure_type}")
    lines.append(f"- predicted_answer: {predicted}")
    lines.append(f"- expected_answer: {expected}")

    if pred_nums and exp_nums:
        pairs = list(zip(pred_nums, exp_nums))
        diffs = [p - e for p, e in pairs]
        abs_diffs = [abs(d) for d in diffs]
        ratios = [
            (p / e)
            for p, e in pairs
            if e not in (0.0, -0.0)
        ]
        lines.append(
            "- numeric_delta: "
            + ", ".join(
                f"pred[{idx}] - expected[{idx}] = {diff:.12g}"
                for idx, diff in enumerate(diffs[:6])
            )
        )
        if ratios:
            lines.append(
                "- numeric_ratio: "
                + ", ".join(
                    f"pred[{idx}] / expected[{idx}] = {ratio:.12g}"
                    for idx, ratio in enumerate(ratios[:6])
                )
            )
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
        "unit conversions, formula choice, aggregation set, and final rounding against the expected answer; "
        "do not assume final-answer formatting is the root cause unless numeric values already match."
    )
    lines.append(
        "- required_proposer_fix: propose a reusable verification gate that would force the agent "
        "to expose and check the earliest divergent intermediate, not merely restate the final expected answer."
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
    labels: list[str] = []
    if n_pred != n_exp:
        labels.append(f"numeric_arity_mismatch(pred={n_pred}, expected={n_exp})")
    if abs_diffs:
        max_abs = max(abs_diffs)
        labels.append(f"max_abs_delta={max_abs:.12g}")
    if ratios:
        max_ratio_gap = max(abs(r - 1.0) for r in ratios)
        labels.append(f"max_relative_gap={max_ratio_gap:.6g}")
        if any(abs(abs(r) - 10.0) < 0.5 or abs(abs(r) - 100.0) < 5.0 for r in ratios):
            labels.append("possible_scale_factor_error")
        elif max_ratio_gap < 0.02:
            labels.append("close_numeric_mismatch")
        elif max_ratio_gap > 0.5:
            labels.append("large_numeric_mismatch")
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
        root_cause: Brief description of root cause.
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
    regression_risks = getattr(proposer_trace.output, "regression_risks", "") or "[not provided]"

    return f"""Proposed tool or skill (high level description): {proposer_trace.output.proposed_skill}

Justification: {proposer_trace.output.justification}

Root cause analysis from proposer:
{root_cause_analysis}

Failure coverage plan:
{coverage_plan}

Regression risks / anti-regression guards:
{regression_risks}"""


def build_judge_query(
    trace_summary: str,
    question: str,
    agent_answer: str,
    ground_truth: str,
    skills_content: str,
    proposal: str = "",
    justification: str = "",
    proposer_root_cause_analysis: str = "",
    proposer_coverage_plan: str = "",
    proposer_regression_risks: str = "",
    parent_skill_summary: str = "",
    candidate_skill_summary: str = "",
    skill_diff: str = "",
    case_type: str = "failure",
    case_feedback: str = "",
) -> str:
    """Build the prompt for the LLM judge.

    The judge sees the failed trajectory, the expected answer, and all current
    skills, then predicts what the agent would do differently and whether the
    new action would succeed.

    Returns:
        A prompt string for the judge LLM.
    """
    parent_section = parent_skill_summary or "[not provided]"
    candidate_section = candidate_skill_summary or "[not provided]"
    diff_section = skill_diff or "[not provided]"

    is_regression = case_type == "regression"
    case_intro = (
        "You are comparing two skill sets on a case where the parent/original behavior was already correct."
        if is_regression
        else "You are comparing two skill sets on a case where the parent/original behavior previously failed."
    )
    baseline_label = "the parent/anchor program skill set"
    answer_label = "Agent's Prior Correct Answer" if is_regression else "Agent's Wrong Answer"
    task_steps = (
        """1. Identify why the original trajectory reached the expected answer.
2. Estimate parent_success_prob: probability the parent/anchor skill set preserves that correct behavior.
3. Estimate candidate_success_prob: probability the candidate skill set preserves that correct behavior.
4. Compare the concrete skill diff and proposer regression risks.
5. Set match_score from candidate-vs-parent relative strength, below 0.5 when the candidate is more likely to regress."""
        if is_regression
        else """1. Identify the key step where the original trajectory went wrong.
2. Estimate parent_success_prob: probability the parent/anchor skill set would solve this same case.
3. Estimate candidate_success_prob: probability the candidate skill set would solve this same case.
4. Compare the concrete skill diff and proposer coverage plan.
5. Set match_score from candidate-vs-parent relative strength, above 0.5 only when the candidate is meaningfully more likely to solve the case."""
    )

    regression_note = (
        "\nFor this regression case, the parent/original behavior already reached the expected answer. "
        "Score the challenger below 0.5 if the candidate skill is likely to change a correct method, "
        "choose a different source slice, alter units/rounding, or otherwise introduce a regression.\n"
        if is_regression
        else ""
    )
    feedback_section = f"\n## Case Feedback\n{case_feedback}\n" if case_feedback else ""
    proposer_analysis_section = ""
    if proposer_root_cause_analysis or proposer_coverage_plan or proposer_regression_risks:
        proposer_analysis_section = f"""
## Proposer Deep Analysis
Root cause analysis:
{proposer_root_cause_analysis or "[not provided]"}

Coverage plan:
{proposer_coverage_plan or "[not provided]"}

Regression risks / guards:
{proposer_regression_risks or "[not provided]"}

Use this as the proposer's hypothesis, not as ground truth. Verify whether the
candidate skill actually implements the stated coverage and anti-regression
guards for this specific case.
"""

    return f"""{case_intro}

Treat this as a pairwise match:
- Player A / parent: {baseline_label}.
- Player B / candidate: the same agent with the candidate skills listed below.

This is an offline skill-evolution estimate, not a strict acceptance test.
Do NOT score the candidate in isolation. Estimate both parent_success_prob and
candidate_success_prob on the same case, then convert the relative advantage
into match_score:
- 0.50 means no meaningful relative advantage.
- >0.50 means candidate is more likely to succeed than parent.
- <0.50 means candidate is worse or more likely to regress than parent.
- Keep match_score near 0.50 when both programs are likely to fail or evidence is weak.
{regression_note}

## Original Question
{question}

## Expected Answer
{ground_truth}

## Agent's Prior Attempt (trajectory summary)
{trace_summary}

## {answer_label}
{agent_answer}
{feedback_section}

## Parent Skill Summary
{parent_section}

## Candidate Skill Summary
{candidate_section}

## Proposal / Intended Change
{proposal or "[not provided]"}

## Proposal Justification
{justification or "[not provided]"}
{proposer_analysis_section}

## Skill Diff / Concrete Change
{diff_section}

## Candidate Skills Now Available to the Agent
{skills_content}

## Your Task
{task_steps}

Respond ONLY with a JSON object (no markdown fences):
{{
  "root_cause": "<brief description of the original failure cause>",
  "parent_hypothetical_action": "<brief description of what the parent/anchor skill set would make the agent do>",
  "candidate_hypothetical_action": "<brief description of what the candidate skill set would make the agent do differently>",
  "hypothetical_action": "<short candidate action summary for backward compatibility>",
  "parent_success_prob": <0.0 to 1.0, probability parent/anchor reaches the expected answer on this case>,
  "candidate_success_prob": <0.0 to 1.0, probability candidate reaches the expected answer on this case>,
  "relative_advantage": <-1.0 to 1.0, candidate_success_prob - parent_success_prob after considering regression risk>,
  "match_score": <0.0 to 1.0, pairwise score for candidate vs parent; 0.5 is draw>,
  "skill_addresses_root_cause": <0.0 to 1.0, how directly the skill addresses the failure cause>,
  "probability_of_success": <0.0 to 1.0, same as candidate_success_prob for backward compatibility>,
  "would_succeed": <true or false>,
  "confidence": <0.0 to 1.0, confidence in your probability estimate>,
  "remaining_blockers": ["<short blocker>", "..."],
  "reasoning": "<one or two sentences>"
}}"""


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
