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
    from src.schemas import SkillProposerResponse, PromptProposerResponse


_COMPACT_STOPWORDS = {
    "about", "above", "after", "again", "agent", "answer", "before", "being",
    "below", "case", "cases", "change", "check", "could", "every", "final",
    "from", "have", "into", "judge", "must", "only", "output", "result",
    "should", "skill", "task", "that", "their", "there", "these", "this",
    "trace", "true", "when", "where", "with", "would", "wrong",
}


def _compact_terms(text: str, max_terms: int = 48) -> list[str]:
    """Return stable, low-noise relevance terms from a prompt/query seed."""
    seen: set[str] = set()
    terms: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}|\d+(?:\.\d+)?", str(text or "").lower()):
        token = raw.strip("_-")
        if not token or token in _COMPACT_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def _wrap_long_line(line: str, max_line_chars: int) -> list[str]:
    if len(line) <= max_line_chars:
        return [line]
    chunks: list[str] = []
    start = 0
    while start < len(line):
        remaining = len(line) - start
        suffix = " [line continued]" if remaining > max_line_chars else ""
        chunk_chars = max_line_chars - len(suffix) if suffix else max_line_chars
        chunks.append(line[start : start + chunk_chars] + suffix)
        start += chunk_chars
    return chunks


def compact_relevant_text(
    text: str,
    relevance_query: str = "",
    *,
    max_chars: int = 3500,
    context_lines: int = 1,
    max_line_chars: int = 700,
) -> str:
    """Compress text by keeping lines relevant to the current proposal/skill.

    This is stricter than head/tail truncation: long trajectories often contain
    many unrelated tool blocks, so we keep matching lines plus small local
    context, then add a short head/tail fallback only if nothing matched.
    """
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return "\n".join(
            chunk
            for line in value.splitlines()
            for chunk in _wrap_long_line(line, max_line_chars)
        )

    terms = _compact_terms(relevance_query)
    signal_re = re.compile(
        r"(?i)\b(error|exception|warning|fail|parse|tool|call|read|grep|rg|"
        r"python|bash|file|path|selected|computed|calculated|final[_ ]?answer|"
        r"answer|output|result|verify|validate|wrong|expected)\b"
    )
    lines = value.splitlines()
    keep: set[int] = set()
    for idx, line in enumerate(lines):
        low = line.lower()
        term_hits = sum(1 for term in terms if term in low)
        signal = bool(signal_re.search(line))
        if term_hits >= 1 or (signal and term_hits >= 0 and len(keep) < 24):
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)
            keep.update(range(start, end))

    if not keep:
        head = value[: max_chars * 2 // 3].rstrip()
        tail = value[-(max_chars - len(head)) :].lstrip()
        return f"{head}\n[... compacted: no relevance hits ...]\n{tail}"

    selected: list[str] = []
    last = -2
    for idx in sorted(keep):
        if idx != last + 1:
            selected.append("[...]")
        selected.append(lines[idx])
        last = idx

    compacted = "\n".join(selected)
    if len(compacted) > max_chars:
        head_chars = max_chars * 3 // 4
        tail_chars = max_chars - head_chars
        compacted = (
            f"{compacted[:head_chars].rstrip()}\n"
            f"[... {len(compacted) - max_chars:,} relevant chars omitted ...]\n"
            f"{compacted[-tail_chars:].lstrip()}"
        )
    return "\n".join(
        chunk
        for line in compacted.splitlines()
        for chunk in _wrap_long_line(line, max_line_chars)
    )


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
    max_active_skills: int = 0,
    induction: bool = False,
    use_playbook: bool = True,
) -> str:
    """Build the query for the proposer agent from multiple failure traces.

    Args:
        traces_with_answers: List of (trace, agent_answer, ground_truth, category) tuples.
        feedback_history: Previous feedback history.
        evolution_mode: "skill_only" or "prompt_only" - affects trace truncation.
        truncation_level: Context reduction level (0=full, 1=moderate, 2=aggressive).
        task_constraints: Optional task-specific constraints to include in the query.
        diversity_hint: Optional instruction to make sibling child proposals differ.
        induction: when True, frame the trajectories as SUCCESSES to distill a
            reusable skill from, instead of failures to fix.

    Returns:
        Formatted query string for the proposer.
    """
    item_label = "Success" if induction else "Failure"
    # Truncation level settings:
    # (head_chars, tail_chars, feedback_lines, max_failures, trace_budget)
    TRUNCATION_SETTINGS = [
        (8_000, 4_000, 40, None, 5_000),  # Level 0: relevant lines
        (4_000, 2_000, 20, 3, 3_200),     # Level 1: moderate
        (2_000, 1_000, 5, 2, 1_800),      # Level 2: aggressive
    ]
    head_chars, tail_chars, feedback_lines, max_failures, trace_budget = TRUNCATION_SETTINGS[
        min(truncation_level, len(TRUNCATION_SETTINGS) - 1)
    ]
    skill_budgets = (3_500, 1_800, 900)
    skill_budget = skill_budgets[min(truncation_level, len(skill_budgets) - 1)]

    # Apply max_failures limit
    if max_failures is not None and len(traces_with_answers) > max_failures:
        traces_with_answers = traces_with_answers[:max_failures]

    # Apply feedback truncation
    if feedback_lines is not None:
        feedback_lines_list = feedback_history.split("\n")
        if len(feedback_lines_list) > feedback_lines:
            feedback_history = "\n".join(feedback_lines_list[-feedback_lines:])

    # Get existing skills for context. Include compact SKILL.md content so the
    # proposer does not need to inspect the filesystem and accidentally read
    # stale skills from historical /tmp runs.
    skills_dir = Path(project_root) / ".claude" / "skills" if project_root else Path(".claude/skills")
    existing_skills = []
    existing_skill_blocks = []
    if skills_dir.exists():
        for skill_dir in sorted(skills_dir.iterdir()):
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                existing_skills.append(skill_dir.name)
                skill_md = (skill_dir / "SKILL.md").read_text(errors="replace")
                if len(skill_md) > skill_budget:
                    head = max(1, int(skill_budget * 0.72))
                    tail = max(1, skill_budget - head)
                    skill_md = (
                        skill_md[:head]
                        + "\n\n...[truncated for proposer retry]...\n\n"
                        + skill_md[-tail:]
                    )
                existing_skill_blocks.append(
                    f"### {skill_dir.name}\n"
                    f"Path: .claude/skills/{skill_dir.name}/SKILL.md\n"
                    f"```markdown\n{skill_md}\n```"
                )
    skills_list = "\n".join([f"- {s}" for s in existing_skills]) or "None"
    skills_content = "\n\n".join(existing_skill_blocks) or "None"

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

        question_line = f"\nQuestion: {questions[i - 1]}" if questions and i - 1 < len(questions) else ""
        q_text = questions[i - 1] if questions and i - 1 < len(questions) else ""
        if not feedback:
            if induction:
                feedback = build_regression_success_feedback(
                    q_text, str(agent_answer), str(ground_truth), domain_hints=domain_hints
                )
            else:
                feedback = build_answer_comparison_feedback(
                    q_text, str(agent_answer), str(ground_truth), domain_hints=domain_hints,
                )

        raw_trace_summary = trace.summarize(head_chars=effective_head, tail_chars=effective_tail)
        relevance_seed = "\n".join([
            q_text,
            str(category),
            feedback,
            skills_list,
            skills_content[:1800],
            diversity_hint,
        ])
        trace_summary = compact_relevant_text(
            raw_trace_summary,
            relevance_seed,
            max_chars=trace_budget,
            context_lines=1,
            max_line_chars=700,
        )

        feedback_label = "Preserved-Surface Feedback" if induction else "Structured Failure Feedback"
        feedback_section = f"\n{feedback_label}:\n{feedback}\n" if feedback else ""

        # NOTE: the raw agent answer and the ground-truth answer are deliberately
        # NOT printed here. Leaking them lets the skill author memorize per-task
        # answers (verbatim "Answer: X / Wrong: Y" worked examples) instead of a
        # general procedure. Only the structured, non-oracle failure feedback below
        # (error shape, likely surfaces) is exposed.
        failure_sections.append(f"""### {item_label} {i} [Category: {category}]{question_line}
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
    playbook_section = (
        "\n## Playbook Mode\n"
        "Playbook mode is disabled. Return bullet_ops=[] in both the top-level "
        "proposal and every skill_edits entry.\n"
        if not use_playbook
        else ""
    )
    budget_reached = bool(max_active_skills) and len(existing_skills) >= max_active_skills
    if not existing_skills:
        existing_skill_rule = (
            "There are no existing project skills. You MUST propose action=\"create\" "
            "and leave target_skill null/empty. Do not edit or reference non-project "
            "skills such as brainstorming or framework/system skills."
        )
    elif budget_reached:
        existing_skill_rule = (
            f"The active-skill budget is reached ({len(existing_skills)}/"
            f"{max_active_skills}). You MUST propose action=\"edit\" on the most "
            "relevant listed skill and fold the fix into it; do NOT create a new "
            "skill (more always-on skills dilute activation and over-trigger). "
            "Only if no listed skill is even remotely related may you create one."
        )
    else:
        existing_skill_rule = (
            "You may propose action=\"edit\" only for one of the exact skill names "
            "listed above. EDIT only when that skill owns the same earliest divergent "
            "mechanism. A broad skill that merely mentions the same task domain does "
            "not count as a match; if the mechanism differs, propose action=\"create\"."
        )

    if induction:
        body_header = (
            f"## Successful Trajectories ({len(traces_with_answers)} samples across "
            f"categories: {category_summary})\n\n"
            "These trajectories produced the CORRECT answer. Identify the reusable "
            "decision, procedure, or guard that made them succeed and abstract it into a "
            "general skill — do NOT memorize the specific answers or inputs."
        )
        task_block = (
            "## Your Task\n"
            "1. Identify the key reusable step(s) that made these succeed (task "
            "interpretation, artifact or tool selection, input binding, operation "
            "semantics, state ordering, validation, or output handling), stated so they "
            "transfer to different inputs, artifacts, tools, environments, and constants.\n"
            "2. Propose ONE general skill (CREATE) or fold the procedure into an existing "
            "skill (EDIT) that would lead an agent to repeat this success on similar tasks.\n"
            "3. Define precise activation boundaries so it does NOT over-trigger when its "
            "preconditions are absent or on unrelated cases; put concrete should_apply_when / "
            "should_not_apply_when.\n"
            "4. Fill all required fields. Use root_cause_analysis to name the success "
            "mechanism being captured; invariants_to_preserve / regression_risks must keep "
            "currently-correct behavior intact."
        )
    else:
        body_header = (
            f"## Current Failures ({len(traces_with_answers)} samples across "
            f"categories: {category_summary})\n\n"
            "Analyze the patterns across these failures to identify a GENERAL improvement, "
            "not a fix for any single case."
        )
        task_block = (
            "## Your Task\n"
            "1. For each failure, name the earliest divergent task-interpretation, "
            "artifact-selection, tool-choice, input-binding, parsing, operation, state, "
            "validation, or output-contract step (not just the final answer).\n"
            "2. Propose ONE general, reusable fix for the shared root cause; keep distinct "
            "failure modes separate and state any uncovered failure in coverage_plan.\n"
            "3. Prefer EDITING a listed project skill when one should have caught these; "
            "otherwise CREATE. Note how your proposal differs from any DISCARDED iteration.\n"
            "4. Fill all required fields with concrete details: root_cause_analysis, "
            "coverage_plan, should_apply_when, should_not_apply_when, invariants_to_preserve, "
            "regression_risks."
        )

    return f"""## Existing Skills (check before proposing new ones)
{skills_list}

Existing skill rule: {existing_skill_rule}

## Existing SKILL.md Context
Use only the listed project-local skills below as the current skill state. Do
not search `/home`, `/tmp`, `.codex`, historical runs, or unrelated repos for
skills; those may be stale and must not influence this proposal.

{skills_content}
{constraints_section}
{diversity_section}
{playbook_section}
## Previous Attempts Feedback
{feedback_history}

{body_header}

{failures_text}

{task_block}"""


_DEFAULT_ERROR_SURFACE_HINTS: dict[str, list[str]] = {
    "artifact or resource selection, scope, version, and authority": [
        "file", "document", "source", "resource", "version", "scope",
    ],
    "tool or API choice, argument binding, and preconditions": [
        "tool", "api", "command", "function", "argument", "parameter",
    ],
    "input parsing, representation, qualifiers, and set membership": [
        "format", "parse", "field", "label", "category", "include", "exclude",
    ],
    "operation semantics, ordering, and intermediate state": [
        "calculate", "compute", "convert", "update", "edit", "before", "after",
    ],
    "validation, error handling, and preservation of existing behavior": [
        "verify", "validate", "error", "test", "preserve", "unchanged",
    ],
    "output contract, structure, precision, and final-answer extraction": [
        "output", "return", "answer", "format", "json", "list", "precision",
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
        "resource selection, operation semantics, state transitions, validation, "
        "or output contract used to reach this answer."
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
        "- first_divergence_to_inspect: compare the trajectory's task interpretation, "
        "artifact and tool selection, input binding, parsing, operation semantics, state "
        "transitions, validation, and output handling against the structured mismatch labels "
        "above; do not assume final-answer formatting is the root cause."
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


def consolidate_meta(records: list[dict[str, Any]], top_k: int = 5) -> str:
    """Deterministically consolidate per-child outcomes into a compact meta block.

    SkillOpt's epoch-wise slow update, done ACE-style (no LLM call, so no context
    collapse). Buckets outcomes into improvements / kept / regressions, ranks the
    most persistent unresolved blockers and the skills that most often improved
    the score, and reports the failure/induction source mix. The result is a
    bounded, protected summary the proposer can use as longitudinal memory.

    Args:
        records: list of dicts with keys outcome, delta, blockers (list),
            root_cause, source, skill.
        top_k: cap on listed blockers / winning skills.

    Returns:
        Markdown meta block, or "" when there is nothing to consolidate.
    """
    from collections import Counter

    if not records:
        return ""

    improvements = [r for r in records if r.get("outcome") == "improved"]
    kept = [r for r in records if r.get("outcome") == "kept"]
    regressions = [r for r in records if (r.get("delta") or 0.0) < 0.0]

    blocker_ct: "Counter[str]" = Counter()
    for r in records:
        for b in r.get("blockers") or []:
            b = str(b).strip()
            if b:
                blocker_ct[b] += 1

    skill_ct: "Counter[str]" = Counter(
        r.get("skill", "") for r in improvements if r.get("skill")
    )
    src_ct: "Counter[str]" = Counter(r.get("source", "failure") for r in records)
    induction = [r for r in records if r.get("source") == "induction"]
    induction_wins = sum(1 for r in induction if r.get("outcome") == "improved")

    lines = [
        f"### Consolidated meta (protected; through {len(records)} attempts)",
        f"- improvements: {len(improvements)} | kept: {len(kept)} | "
        f"regressions: {len(regressions)}",
    ]
    if skill_ct:
        lines.append(
            "- what worked, build on these: "
            + ", ".join(f"{s} (x{c})" for s, c in skill_ct.most_common(top_k))
        )
    if blocker_ct:
        lines.append(
            "- persistent unresolved blockers (do NOT re-propose the same fix): "
            + ", ".join(f"{b} (x{c})" for b, c in blocker_ct.most_common(top_k))
        )
    src_line = (
        f"- source mix: failure {src_ct.get('failure', 0)} / "
        f"induction {src_ct.get('induction', 0)}"
    )
    if induction:
        src_line += f" (induction wins: {induction_wins}/{len(induction)})"
    lines.append(src_line)
    return "\n".join(lines)


def read_meta(path: Path) -> str:
    """Read the protected meta summary, or '' when absent."""
    return path.read_text() if path.exists() else ""


def write_meta(path: Path, text: str) -> None:
    """Write the protected meta summary (overwrites; epoch-consolidator only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


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

Filesystem rule: use only the information in this prompt. Do not search `/home`,
`/tmp`, `.codex`, historical runs, or unrelated repos for existing skills or
examples. For a new skill, produce `.claude/skills/<skill-name>/SKILL.md` from
the proposal and the provided evidence only.

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
    candidate_skill_tests: str = "",
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
    is_success_induction = case_type == "success_induction"
    answer_label = (
        "Agent's Prior Correct Answer"
        if is_regression or is_success_induction
        else "Agent's Wrong Answer"
    )
    if is_success_induction:
        case_intro = (
            "On this case the original trajectory succeeded and was used as positive "
            "evidence for skill induction."
        )
    else:
        case_intro = (
            "On this case the original skills already produced the correct answer."
            if is_regression
            else "On this case the original skills previously produced a wrong answer."
        )
    if is_success_induction:
        direction = (
            "One set includes the Change Being Evaluated and the other does not — work out "
            "which from the summaries. This is a positive-induction check: reward the "
            "including set only if it captures the reusable success mechanism from this "
            "trajectory and would make that behavior more reliable on similar cases. "
            "Do not reward answer leakage, restating the solved example, or broad "
            "over-triggering prose."
        )
    else:
        direction = (
        "One set includes the Change Being Evaluated and the other does not — work out "
        "which from the summaries. This is a preservation check: the original "
        "set already succeeded. The changed set should usually TIE. Score it "
        "lower only when the change has a concrete activation path on this case "
        "that would pick a different resource/tool, change an operation, alter "
        "the output contract, or add an unsafe override. Do not penalize vague "
        "overhead concerns or extra checks that would not change the successful "
        "path. Do not score the changed set higher merely because it adds "
        "relevant-looking checks or prose."
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

Important scope rule: a narrow skill is not required to fix every unrelated
failure. On a failure case whose root cause is outside the Change Being
Evaluated's stated mechanism, set case_relevance low and keep the pairwise score
near 0.5 unless the change would truly alter this case's outcome. Use unrelated
failure cases to report remaining blockers, not to punish the skill for not
solving mechanisms it did not target. Regression cases are preservation-only:
penalize only concrete breakage or a specific false-activation path, not generic
concern about more text, extra validation, or runtime overhead.
Success-induction cases are positive evidence: judge whether the change captures
the successful mechanism in a portable way and preserves the original successful
path. Do not require it to solve unrelated failures.

Do not reward polished but generic skill text. A useful skill is narrow: it fixes
the true failure mechanism and stays out of unrelated cases. Judge four
case-grounded dimensions:
1. failure_mechanism_encoding: does the change name the domain-specific failure
   mechanism that caused THIS case, not just a broad category like "verification"?
2. executable_specificity: does it force concrete operations/checks that would
   change the agent's behavior on THIS case, rather than advice like "be careful"?
3. high_risk_blacklist: does it explicitly forbid tempting operations that would
   cause the same failure or regressions on similar cases?
4. activation_boundary: would the skill avoid over-triggering on tasks outside
   its stated preconditions and already-unambiguous cases?

The pairwise score must reflect those dimensions. If the Change Being Evaluated
lacks the true failure mechanism or an executable remedy, do NOT give it a high
B-over-A score merely because it is well written. If it introduces a risky
operation without blacklisting it, avoid credit on failure cases. On regression
cases, lower the including set only for a concrete disturbance of the successful
path; otherwise leave it at a draw.

Also judge transferability. The skill should abstract from the observed case to
the reusable failure mechanism. Penalize changes that look strong only because
they name this case's surface details (exact files, dates/times, labels, entity
names, paths, API objects, UI elements, answer values, or a single operation) in
activation rules or invariants. Worked examples may contain concrete values, but
the rule itself must help a new task with different inputs, files, entities,
tools, environments, formats, constants, and neighboring operations from the same
failure family. Score generalization_transfer high only when the change is
specific about the mechanism while remaining portable across such variants.

Also evaluate the candidate-generated synthetic tests, if provided. Good suites
include positive activation, negative no-activation, over-trigger regression, and
high-risk blacklist/adversarial cases. They should also include a nontrivial
already-correct boundary case with competing artifacts, tools, representations,
or state transitions; this catches skills that preserve simple cases but disturb
realistic workflows. Penalize tautological tests, copied
sampled tasks, answer leakage, missing negative cases, and tests disconnected
from the diff. self_test_pass_rate is the fraction of meaningful tests that
would pass.

If the Candidate-Generated Skill Tests section includes
EXECUTABLE_SKILL_TEST_RESULTS, those tests were actually run against the
candidate program. Treat actual_pass_rate as authoritative for
self_test_pass_rate; use the listed failures to judge whether the skill
over-triggers, fails to activate, omits required checks, or does not reject
dangerous operations.

First derive, independently from the trajectory and the expected answer, the
earliest divergent step that actually caused this case to fail, or for a
success-induction case the key decision/procedure/guard that made it succeed.
Then judge the proposer's claimed root cause/success mechanism against your own
finding: set proposer_root_cause_correct high only if the proposer located the same earliest
divergence or success mechanism; set it low when the proposer misdiagnosed (e.g. blamed an operation
when the real error was choosing the wrong artifact, tool, input binding, or
precondition). Then judge skill_addresses_root_cause: how concretely
the Change Being Evaluated enforces a fix for the TRUE root cause or codifies the
TRUE success mechanism — not merely whether it adds plausible-looking structure.
A change that builds elaborate contracts but does not constrain the actual
divergent/success step scores low here.

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

## Candidate-Generated Skill Tests
{candidate_skill_tests or "[not provided]"}

Respond ONLY with a JSON object (no markdown fences):
{{
  "root_cause": "<the earliest divergent step, or success mechanism for success-induction>",
  "case_relevance": <0.0-1.0 whether THIS case belongs to the mechanism targeted by the Change Being Evaluated>,
  "proposer_root_cause_correct": <0.0-1.0 how well the proposer's claimed root cause/success mechanism matches yours>,
  "skill_addresses_root_cause": <0.0-1.0 how concretely the Change Being Evaluated fixes the TRUE root cause or codifies the success mechanism>,
  "failure_mechanism_encoding": <0.0-1.0 whether the change encodes the concrete domain-specific failure mechanism>,
  "executable_specificity": <0.0-1.0 whether the change gives concrete operations/checks that would alter behavior>,
  "high_risk_blacklist": <0.0-1.0 whether the change forbids likely harmful operations/regressions>,
  "generalization_transfer": <0.0-1.0 whether the change abstracts the mechanism beyond this exact case without becoming vague>,
  "self_test_pass_rate": <0.0-1.0 fraction of candidate-generated tests that are meaningful and would pass>,
  "set_a_success_prob": <0.0-1.0 probability Skill Set A reaches the expected answer>,
  "set_b_success_prob": <0.0-1.0 probability Skill Set B reaches the expected answer>,
  "b_over_a_score": <0.0-1.0 pairwise score for Set B over Set A; 0.5 ONLY if truly indistinguishable>,
  "confidence": <0.0-1.0 confidence in your estimate>,
  "remaining_blockers": ["<short blocker the change still does not fix>"],
  "reasoning": "<one sentence>"
}}"""


def build_compact_judge_query(
    trace_summary: str,
    question: str,
    agent_answer: str,
    ground_truth: str,
    parent_skill_summary: str = "",
    candidate_skill_summary: str = "",
    skill_diff: str = "",
    candidate_skill_tests: str = "",
    case_type: str = "failure",
    case_feedback: str = "",
    proposer_root_cause: str = "",
    candidate_slot: str = "B",
) -> str:
    """Build a short judge prompt for direct model calls.

    This keeps the same output contract as ``build_judge_query`` but removes the
    long rubric text. It is intended for models whose reasoning budget can eat
    the response budget when prompts are verbose.
    """
    if case_type == "success_induction":
        case_intro = (
            "SUCCESS-INDUCTION: this successful trajectory was shown to the proposer. "
            "Reward the changed set only if it captures the reusable success mechanism "
            "and keeps it portable; tie or penalize answer leakage, over-triggering, "
            "or prose that would not change future behavior."
        )
    else:
        case_intro = (
            "REGRESSION: original skills already answered this case correctly. This "
            "is preservation-only: default to a draw. Penalize the changed set only "
            "for concrete false activation or a specific change to the successful "
            "resource/tool/operation/output path, not vague overhead."
            if case_type == "regression"
            else "FAILURE: original skills answered this case incorrectly. Reward the "
            "changed set only if it concretely fixes the true earliest divergence."
        )
    feedback_section = f"\n## Case Feedback\n{case_feedback}\n" if case_feedback else ""
    proposer_rc_section = (
        f"\n## Proposer Claimed Root Cause\n{proposer_root_cause}\n"
        if proposer_root_cause and proposer_root_cause.strip()
        else ""
    )
    normalized_slot = "A" if str(candidate_slot).upper() == "A" else "B"
    parent_slot = "B" if normalized_slot == "A" else "A"
    return f"""Compare Skill Set A vs Skill Set B on ONE case. A/B order is random.
The CANDIDATE containing the Change Being Evaluated is Skill Set {normalized_slot}.
The unchanged PARENT is Skill Set {parent_slot}. Estimate success probabilities
for the labeled sets and B-over-A score from concrete behavior, not writing quality.

{case_intro}

Scope rule: a narrow skill is not required to fix unrelated failure mechanisms.
For a failure outside the Change Being Evaluated's mechanism, set
case_relevance low and keep the score near 0.5 unless the change truly changes
this case's outcome. Use unrelated cases for blockers, not punishment.
Regression cases penalize concrete breakage of previously-correct paths, not
generic concerns about extra checks or more skill text.
Success-induction cases should score whether the change captures and preserves
the positive mechanism from the successful trajectory; do not demand that it fix
unseen unrelated failures.

Judge these dimensions:
- case relevance to the change's targeted failure mechanism
- true earliest divergent step/root cause, or success mechanism for induction
- whether proposer identified that root cause
- whether the change enforces an executable fix
- failure-mechanism specificity
- high-risk operations it blacklists
- activation boundary / over-trigger risk
- transfer to nearby tasks without memorizing exact files, labels, entities, or values
- meaningful self-test pass rate; if executable test results exist, treat actual_pass_rate as authoritative

Use 0.5 only for genuinely indistinguishable sets. Be decisive for small real
advantages. Do not use hidden reasoning; output the JSON immediately. Return
ONLY minified JSON. Keep root_cause/reasoning <=25 words each; remaining_blockers
<=2 short strings.

## Question
{question}

## Expected Answer
{ground_truth}

## Agent Prior Answer
{agent_answer}

## Prior Trajectory
{trace_summary}
{feedback_section}{proposer_rc_section}
## Skill Set A
{parent_skill_summary or "[not provided]"}

## Skill Set B
{candidate_skill_summary or "[not provided]"}

## Change Being Evaluated
Present in Skill Set {normalized_slot}; absent from Skill Set {parent_slot}.
{skill_diff or "[not provided]"}

## Candidate Skill Tests
{candidate_skill_tests or "[not provided]"}

JSON schema:
{{"root_cause":"...","case_relevance":0.0,"proposer_root_cause_correct":0.0,"skill_addresses_root_cause":0.0,"failure_mechanism_encoding":0.0,"executable_specificity":0.0,"high_risk_blacklist":0.0,"generalization_transfer":0.0,"self_test_pass_rate":0.0,"set_a_success_prob":0.0,"set_b_success_prob":0.0,"b_over_a_score":0.5,"confidence":0.0,"remaining_blockers":["..."],"reasoning":"..."}}"""


def build_skill_revision_query(
    target_skill: str,
    current_skill_markdown: str,
    proposer_root_cause: str,
    original_proposal: str,
    judge_root_cause: str,
    judge_blockers: list[str],
    judge_reasoning: str = "",
    structural_gaps: list[str] | None = None,
    self_test_gap: bool = False,
    executable_self_test_feedback: str = "",
    induction: bool = False,
) -> str:
    """Build the generator query for one judge-driven revision of a child skill.

    The generator is shown two clearly separated signals: the proposer's ORIGINAL
    intent (what the skill tried to fix) and the JUDGE's independent verdict on
    why that fix falls short (the correction direction). The generator must keep
    the skill's valid structure and make a targeted change toward the judge's
    blockers — not rewrite from scratch.

    ``structural_gaps`` (required SKILL.md sections that are missing) and
    ``self_test_gap`` (the synthetic suite lacks a negative no-activation case)
    are completeness corrections appended as explicit directives so a revision
    closes them rather than only chasing the judge's root-cause feedback.
    """
    blockers_text = "\n".join(f"- {b}" for b in judge_blockers) or "- (none stated)"
    completeness_directives: list[str] = []
    if structural_gaps:
        completeness_directives.append(
            "Add these required sections that are currently missing (keep them "
            f"concrete and case-grounded): {', '.join(structural_gaps)}."
        )
    if self_test_gap:
        completeness_directives.append(
            "The synthetic test suite has no negative (should_activate=false) "
            "case. Add at least one negative/over-trigger test so the suite "
            "actually exercises when the skill must stay out of the way."
        )
    completeness_section = (
        "\n## Completeness Corrections (must close all of these)\n"
        + "\n".join(f"- {d}" for d in completeness_directives)
        + "\n"
        if completeness_directives
        else ""
    )
    executable_self_test_section = (
        "\n## Executable Self-Test Feedback (real agent runs; fix these failures)\n"
        f"{executable_self_test_feedback.strip()}\n"
        if executable_self_test_feedback.strip()
        else ""
    )
    review_context = (
        "successful induction examples"
        if induction
        else "held-out failures"
    )
    shortfall = (
        "does not yet capture the successful trajectory's reusable mechanism"
        if induction
        else "does not yet solve the failure's true root cause"
    )
    intent_label = (
        "what this skill tried to preserve/transfer"
        if induction
        else "what this skill tried to fix"
    )
    judge_label = (
        "Judge's identified true success mechanism"
        if induction
        else "Judge's identified true root cause"
    )
    final_instruction = (
        "Focus the change on codifying the judge's success mechanism as a "
        "portable activation rule and executable procedure, while preserving the "
        "original successful path and avoiding answer leakage."
        if induction
        else "Focus the change on enforcing a fix for the judge's true root cause "
        "(for example, add or repair the protocol field or gate that pins the "
        "actual divergent step), and ensure worked examples show the real final "
        "result rather than any intermediate scaffold value."
    )
    return f"""REVISE existing skill: {target_skill}

A skeptical judge reviewed this skill against {review_context} and concluded it
{shortfall}. Revise the skill with ONE
targeted change. Keep all still-valid content (triggers, invariants, correct
worked examples); do not rewrite from scratch.

Filesystem rule: the authoritative current skill is printed below. Do not search
`/home`, `/tmp`, `.codex`, historical runs, or unrelated repos for other versions
of this skill.

## Original intent (proposer) — {intent_label}
{proposer_root_cause or original_proposal or "[not provided]"}

## Judge's independent verdict — the correction direction (treat as authoritative)
{judge_label}:
{judge_root_cause or "[not provided]"}

Why the current skill still falls short (unresolved blockers):
{blockers_text}

Judge note: {judge_reasoning or "[none]"}
{completeness_section}{executable_self_test_section}
## Current SKILL.md (revise this)
{current_skill_markdown}

Produce the complete revised SKILL.md. {final_instruction} If the judge's
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
