"""Tests for src/loop/ — LoopConfig, helpers (build_proposer_query, feedback management)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock


# ===========================================================================
# LoopConfig — dataclass defaults and field types
# ===========================================================================

class TestNoImprovementStop:
    """Early-stop guard: a falsy limit disables it entirely."""

    def test_fires_at_limit(self):
        from src.loop.runner import _no_improvement_stop

        assert _no_improvement_stop(5, 5) is True
        assert _no_improvement_stop(6, 5) is True

    def test_does_not_fire_below_limit(self):
        from src.loop.runner import _no_improvement_stop

        assert _no_improvement_stop(4, 5) is False

    def test_disabled_when_limit_zero(self):
        from src.loop.runner import _no_improvement_stop

        # Must NOT fire even when the count is huge — runs the full budget.
        assert _no_improvement_stop(0, 0) is False
        assert _no_improvement_stop(999, 0) is False

    def test_disabled_when_limit_none(self):
        from src.loop.runner import _no_improvement_stop

        assert _no_improvement_stop(999, None) is False


class TestLoopConfig:
    def test_default_max_iterations(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.max_iterations == 5

    def test_no_improvement_limit_accepts_zero_and_none(self):
        from src.loop.config import LoopConfig

        assert LoopConfig(no_improvement_limit=0).no_improvement_limit == 0
        assert LoopConfig(no_improvement_limit=None).no_improvement_limit is None

    def test_default_frontier_size(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.frontier_size == 3

    def test_default_evolution_mode(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.evolution_mode == "skill_only"

    def test_default_selection_strategy(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.selection_strategy == "best"

    def test_default_tolerance_is_zero(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.tolerance == 0.0

    def test_default_cache_enabled(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.cache_enabled is True

    def test_default_cache_dir(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.cache_dir == Path(".cache/runs")

    def test_default_reset_feedback(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.reset_feedback is True

    def test_custom_values_accepted(self):
        from src.loop.config import LoopConfig

        config = LoopConfig(
            max_iterations=20,
            frontier_size=5,
            evolution_mode="prompt_only",
            selection_strategy="random",
            tolerance=0.05,
        )
        assert config.max_iterations == 20
        assert config.frontier_size == 5
        assert config.evolution_mode == "prompt_only"
        assert config.selection_strategy == "random"
        assert config.tolerance == pytest.approx(0.05)

    def test_cache_dir_is_path_instance(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert isinstance(config.cache_dir, Path)

    def test_default_concurrency(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().concurrency == 4

    def test_default_no_improvement_limit(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().no_improvement_limit == 5

    def test_default_failure_sample_count(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().failure_sample_count == 2

    def test_default_samples_per_category(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().samples_per_category == 1

    def test_proposer_max_truncation_level_default(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().proposer_max_truncation_level == 2

    def test_consecutive_proposer_failures_limit_default(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().consecutive_proposer_failures_limit == 5

    def test_executable_skill_tests_defaults(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.executable_skill_tests is True
        assert config.executable_skill_test_max_cases == 4
        assert config.executable_skill_test_concurrency == 2
        assert config.executable_skill_test_timeout_seconds == 240


def test_multi_tolerance_scorer_empty_prediction_scores_zero() -> None:
    from src.loop.runner import _score_multi_tolerance

    assert _score_multi_tolerance("question", "", "42") == 0.0
    assert _score_multi_tolerance("question", None, "42") == 0.0


# ===========================================================================
# build_proposer_query
# ===========================================================================

def _make_trace(result_text="Agent result", parse_error=None):
    """Create a lightweight AgentTrace mock."""
    from src.harness.agent import AgentTrace

    trace = AgentTrace(
        duration_ms=1000,
        total_cost_usd=0.01,
        num_turns=2,
        usage={},
        result=result_text,
        is_error=False,
        messages=[],
        model="claude-opus-4-5",
        parse_error=parse_error,
    )
    return trace


class TestBuildProposerQuery:
    def test_returns_string(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "agent_answer", "ground_truth", "math")]
        result = build_proposer_query(
            traces, "No previous attempts.", project_root=tmp_path
        )
        assert isinstance(result, str)

    def test_includes_failure_section(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace("Agent said X"), "X", "Y", "category_a")]
        result = build_proposer_query(
            traces, "No previous attempts.", project_root=tmp_path
        )
        assert "Failure" in result

    def test_omits_agent_answer_and_ground_truth(self, tmp_path):
        # Answer-blind contract: the skill author must never see the raw agent
        # answer or the ground truth, or it memorizes per-task answers instead of
        # learning a general procedure.
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "predicted_42", "true_100", "finance")]
        result = build_proposer_query(
            traces, "No previous attempts.", project_root=tmp_path
        )
        assert "predicted_42" not in result
        assert "true_100" not in result
        assert "Ground Truth" not in result

    def test_includes_structured_failure_feedback(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "42", "100", "finance", "custom failure feedback")]
        result = build_proposer_query(
            traces,
            "No previous attempts.",
            project_root=tmp_path,
            questions=["What is the total?"],
        )
        assert "Structured Failure Feedback" in result
        assert "custom failure feedback" in result
        assert "Question: What is the total?" in result

    def test_auto_generates_answer_comparison_feedback(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "42", "100", "finance")]
        result = build_proposer_query(
            traces,
            "",
            project_root=tmp_path,
            questions=["What is the total?"],
        )
        # Structured, answer-blind feedback is present...
        assert "mismatch_shape" in result
        assert "required_proposer_fix" in result
        # ...but the raw agent answer, ground truth, and any signed delta/ratio
        # that would reconstruct the answer must NOT leak to the skill author.
        assert "Ground Truth" not in result
        assert "expected_answer" not in result
        assert "predicted_answer" not in result
        assert "numeric_delta" not in result

    def test_includes_categories_summary(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [
            (_make_trace(), "a1", "gt1", "math"),
            (_make_trace(), "a2", "gt2", "finance"),
        ]
        result = build_proposer_query(traces, "", project_root=tmp_path)
        assert "math" in result
        assert "finance" in result

    def test_includes_feedback_history(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(
            traces,
            "iter-1: tried numeric extraction",
            project_root=tmp_path,
        )
        assert "iter-1" in result

    def test_skill_only_mode_default(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(
            traces, "", evolution_mode="skill_only", project_root=tmp_path
        )
        assert isinstance(result, str)

    def test_prompt_only_mode(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(
            traces, "", evolution_mode="prompt_only", project_root=tmp_path
        )
        assert isinstance(result, str)

    def test_truncation_level_1_limits_failures(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        # Truncation level 1 limits to max_failures=3
        traces = [(_make_trace(f"result {i}"), f"a{i}", f"gt{i}", "cat") for i in range(6)]
        result = build_proposer_query(
            traces, "", truncation_level=1, project_root=tmp_path
        )
        # Should mention at most 3 failures
        assert "Failure 4" not in result

    def test_truncation_level_2_aggressive(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace("x" * 10000), "a", "gt", "cat")]
        # Should not raise and should produce a shorter query than level 0
        result_full = build_proposer_query(
            traces, "", truncation_level=0, project_root=tmp_path
        )
        result_agg = build_proposer_query(
            traces, "", truncation_level=2, project_root=tmp_path
        )
        assert len(result_agg) <= len(result_full)

    def test_truncation_level_reduces_existing_skill_context(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        skill_dir = tmp_path / ".claude" / "skills" / "large-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Skill\n" + ("rule text\n" * 800))
        traces = [(_make_trace("short"), "a", "gt", "cat")]

        result_full = build_proposer_query(
            traces, "", truncation_level=0, project_root=tmp_path
        )
        result_agg = build_proposer_query(
            traces, "", truncation_level=2, project_root=tmp_path
        )

        assert len(result_agg) < len(result_full)
        assert "truncated for proposer retry" in result_agg

    def test_playbook_disabled_requires_empty_bullet_ops(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        result = build_proposer_query(
            [(_make_trace(), "a", "gt", "cat")],
            "",
            project_root=tmp_path,
            use_playbook=False,
        )

        assert "Playbook mode is disabled" in result
        assert "bullet_ops=[]" in result

    def test_task_constraints_included(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(
            traces,
            "",
            task_constraints="Only use Python tools.",
            project_root=tmp_path,
        )
        assert "Only use Python tools." in result

    def test_existing_skills_listed(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        # Create a fake skill directory
        skills_dir = tmp_path / ".claude" / "skills" / "my-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("# My Skill")

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(traces, "", project_root=tmp_path)
        assert "my-skill" in result

    def test_no_skills_shows_none(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(traces, "", project_root=tmp_path)
        assert "None" in result


# ===========================================================================
# append_feedback
# ===========================================================================

class TestAppendFeedback:
    def test_creates_file_if_not_exists(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(path, "iter-1", "skill proposal", "justification")
        assert path.exists()

    def test_appends_entry_to_existing_file(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        path.write_text("# History\n")
        append_feedback(path, "iter-1", "proposal A", "reason A")
        append_feedback(path, "iter-2", "proposal B", "reason B")
        content = path.read_text()
        assert "iter-1" in content
        assert "iter-2" in content

    def test_entry_contains_proposal_and_justification(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(path, "iter-1", "my proposal", "my reason")
        content = path.read_text()
        assert "my proposal" in content
        assert "my reason" in content

    def test_outcome_section_when_provided(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path,
            "iter-1",
            "proposal",
            "reason",
            outcome="improved",
            score=0.85,
            parent_score=0.72,
        )
        content = path.read_text()
        assert "IMPROVED" in content
        assert "0.8500" in content

    def test_outcome_no_improvement(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path, "iter-1", "proposal", "reason", outcome="no_improvement"
        )
        assert "NO_IMPROVEMENT" in path.read_text()

    def test_active_skills_included(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path,
            "iter-1",
            "proposal",
            "reason",
            active_skills=["skill-a", "skill-b"],
        )
        content = path.read_text()
        assert "skill-a" in content
        assert "skill-b" in content

    def test_failure_category_included(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path, "iter-1", "proposal", "reason", failure_category="formatting"
        )
        assert "formatting" in path.read_text()

    def test_root_cause_included(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path, "iter-1", "proposal", "reason", root_cause="Agent skips steps"
        )
        assert "Agent skips steps" in path.read_text()

    def test_remaining_blockers_included(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path,
            "iter-1",
            "proposal",
            "reason",
            remaining_blockers=["wrong source column", "missing unit conversion"],
        )
        content = path.read_text()
        assert "Remaining Blockers (judge)" in content
        assert "wrong source column" in content
        assert "missing unit conversion" in content

    def test_remaining_blockers_blank_when_empty(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path, "iter-1", "proposal", "reason", remaining_blockers=["", "  "]
        )
        assert "Remaining Blockers" not in path.read_text()

    def test_delta_displayed_with_sign(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path,
            "iter-1",
            "proposal",
            "reason",
            outcome="improved",
            score=0.90,
            parent_score=0.80,
        )
        content = path.read_text()
        assert "+0.1000" in content


# ===========================================================================
# _summarize_judge_feedback
# ===========================================================================

class TestSummarizeJudgeFeedback:
    @staticmethod
    def _match(match_score, root_cause="", blockers=None, valid=True):
        from types import SimpleNamespace

        return SimpleNamespace(
            match_score=match_score,
            root_cause=root_cause,
            remaining_blockers=blockers or [],
            valid=valid,
        )

    def _summarize(self, matches):
        from types import SimpleNamespace
        from src.loop.runner import SelfImprovingLoop

        return SelfImprovingLoop._summarize_judge_feedback(
            SimpleNamespace(matches=matches)
        )

    def test_none_and_empty(self):
        from src.loop.runner import SelfImprovingLoop

        assert SelfImprovingLoop._summarize_judge_feedback(None) == ("", [])
        assert self._summarize([]) == ("", [])

    def test_root_cause_from_weakest_case(self):
        matches = [
            self._match(0.9, root_cause="strong case cause"),
            self._match(0.1, root_cause="weak case cause"),
        ]
        root_cause, _ = self._summarize(matches)
        assert root_cause == "weak case cause"

    def test_blockers_deduped_and_capped(self):
        matches = [
            self._match(0.2, blockers=["b1", "B1", "b2"]),
            self._match(0.3, blockers=["b3", "b4", "b5", "b6"]),
        ]
        _, blockers = self._summarize(matches)
        assert blockers == ["b1", "b2", "b3", "b4", "b5"]  # deduped (B1==b1), capped at 5

    def test_invalid_matches_skipped_when_valid_exist(self):
        matches = [
            self._match(0.1, root_cause="invalid cause", blockers=["x"], valid=False),
            self._match(0.5, root_cause="valid cause", blockers=["y"], valid=True),
        ]
        root_cause, blockers = self._summarize(matches)
        assert root_cause == "valid cause"
        assert blockers == ["y"]


# ===========================================================================
# read_feedback_history
# ===========================================================================

class TestReadFeedbackHistory:
    def test_returns_file_contents_when_exists(self, tmp_path):
        from src.loop.helpers import read_feedback_history

        path = tmp_path / "feedback.md"
        path.write_text("## iter-1\nsome content")
        result = read_feedback_history(path)
        assert "iter-1" in result

    def test_returns_default_message_when_file_missing(self, tmp_path):
        from src.loop.helpers import read_feedback_history

        path = tmp_path / "nonexistent.md"
        result = read_feedback_history(path)
        assert "No previous attempts" in result


# ===========================================================================
# build_skill_query_from_skill_proposer
# ===========================================================================

class TestBuildSkillQuery:
    def test_includes_proposed_skill(self):
        from src.loop.helpers import build_skill_query_from_skill_proposer
        from src.schemas import SkillProposerResponse
        from src.harness.agent import AgentTrace

        proposer_output = SkillProposerResponse(
            proposed_skill="Build a numeric extraction tool",
            justification="Agent lacks unit handling",
            root_cause_analysis="Failure 1 diverged at unit conversion.",
            coverage_plan="Covers unit mismatches by forcing a conversion ledger.",
            should_apply_when="Use when table values require unit conversion before answering.",
            should_not_apply_when="Do not use when source and requested units already match.",
            invariants_to_preserve="Preserve explicit row, column, source unit, target unit, and rounding order.",
            regression_risks="Could over-convert already-normalized values.",
        )
        trace = AgentTrace(
            duration_ms=500,
            total_cost_usd=0.005,
            num_turns=1,
            usage={},
            result="",
            is_error=False,
            messages=[],
            output=proposer_output,
        )
        query = build_skill_query_from_skill_proposer(trace)
        assert "numeric extraction tool" in query
        assert "unit handling" in query
        assert "unit conversion" in query
        assert "conversion ledger" in query
        assert "Use this skill only when" in query
        assert "source and requested units already match" in query
        assert "rounding order" in query
        assert "over-convert" in query


# ===========================================================================
# build_judge_query
# ===========================================================================

class TestBuildJudgeQuery:
    def test_judges_diff_only_without_proposer_rationale(self):
        from src.loop.helpers import build_judge_query

        query = build_judge_query(
            trace_summary="trace summary text",
            question="what is the total?",
            agent_answer="42",
            ground_truth="100",
            parent_skill_summary="parent skill",
            candidate_skill_summary="candidate skill",
            skill_diff="+ add a conversion ledger",
        )

        # The judge must see the concrete artifact and the case...
        assert "conversion ledger" in query
        assert "100" in query  # expected answer
        assert "Change Being Evaluated" in query
        assert "decisive" in query
        # ...but NEVER the proposer's own rationale for why the change works.
        assert "Proposer Deep Analysis" not in query
        assert "justification" not in query.lower()

    def test_regression_case_is_described_as_preservation_only(self):
        from src.loop.helpers import build_judge_query

        query = build_judge_query(
            trace_summary="trace",
            question="question",
            agent_answer="correct",
            ground_truth="correct",
            case_type="regression",
        )

        assert "preservation check" in query
        assert "do NOT score the including set higher" in query


# ===========================================================================
# build_prompt_query_from_prompt_proposer
# ===========================================================================

class TestBuildPromptQuery:
    def test_includes_proposed_change_and_original(self):
        from src.loop.helpers import build_prompt_query_from_prompt_proposer
        from src.schemas import PromptProposerResponse
        from src.harness.agent import AgentTrace

        proposer_output = PromptProposerResponse(
            proposed_prompt_change="Add step-by-step instructions",
            justification="Agent skips reasoning",
        )
        trace = AgentTrace(
            duration_ms=500,
            total_cost_usd=0.005,
            num_turns=1,
            usage={},
            result="",
            is_error=False,
            messages=[],
            output=proposer_output,
        )
        query = build_prompt_query_from_prompt_proposer(trace, "Original prompt text")
        assert "Original prompt text" in query
        assert "step-by-step" in query
        assert "Agent skips reasoning" in query


# ===========================================================================
# ensure_skill_frontmatter
# ===========================================================================

class TestEnsureSkillFrontmatter:
    def test_adds_frontmatter_when_missing(self, tmp_path):
        from src.harness.opencode.skill_utils import ensure_skill_frontmatter

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# Skill body content\n")

        result = ensure_skill_frontmatter(
            skill_file, description="A useful skill"
        )
        assert result is True
        content = skill_file.read_text()
        assert "---" in content
        assert "my-skill" in content

    def test_returns_false_when_file_missing(self, tmp_path):
        from src.harness.opencode.skill_utils import ensure_skill_frontmatter

        result = ensure_skill_frontmatter(
            tmp_path / "nonexistent" / "SKILL.md",
            description="desc",
        )
        assert result is False

    def test_preserves_existing_description(self, tmp_path):
        from src.harness.opencode.skill_utils import ensure_skill_frontmatter

        skill_dir = tmp_path / "skill-a"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("---\nname: skill-a\ndescription: Old description\n---\n\nBody")

        result = ensure_skill_frontmatter(skill_file, description="New description")
        # Description already exists → no rewrite
        assert result is False
        content = skill_file.read_text()
        assert "Old description" in content

    def test_description_truncated_if_too_long(self, tmp_path):
        from src.harness.opencode.skill_utils import ensure_skill_frontmatter

        skill_dir = tmp_path / "long-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# Body")

        long_desc = "x " * 600  # > 1024 chars
        ensure_skill_frontmatter(skill_file, description=long_desc)
        content = skill_file.read_text()
        # Description in frontmatter should be truncated with ellipsis
        assert "..." in content


# ===========================================================================
# Judge→generator refine loop, config, and supporting helpers
# ===========================================================================

class TestRefineAndJudgeConfig:
    def test_refine_and_streamlining_defaults(self):
        from src.loop.config import LoopConfig

        c = LoopConfig()
        assert c.refine_with_judge_feedback is True
        assert c.refine_max_rounds == 1
        assert c.refine_root_cause_threshold == 0.5
        assert c.refine_generalization_threshold == 0.55
        assert c.refine_self_test_threshold == 0.55
        # Streamlining defaults: swap off, no inherited-failure superset.
        assert c.judge_position_swap is False
        assert c.judge_inherit_parent_failures is False
        assert c.validate_worked_examples is True


class TestMisleadingWorkedExample:
    def test_flags_scaffold_count_as_answer(self):
        from src.loop.runner import SelfImprovingLoop

        md = (
            "## Worked Example\n"
            "ledger_count_check: expected 7 categories * 2 periods = 14 threshold tests\n"
            "Answer: 14 categories\n"
        )
        assert SelfImprovingLoop._has_misleading_worked_example(md) is True

    def test_allows_distinct_answer(self):
        from src.loop.runner import SelfImprovingLoop

        md = (
            "## Worked Example\n"
            "ledger_count_check: expected 7 categories * 2 periods = 14 threshold tests\n"
            "Answer: 3 categories\n"
        )
        assert SelfImprovingLoop._has_misleading_worked_example(md) is False

    def test_empty_is_clean(self):
        from src.loop.runner import SelfImprovingLoop

        assert SelfImprovingLoop._has_misleading_worked_example("") is False


class TestMaterializeGeneratedSkill:
    def test_writes_skill_tests_json_from_pydantic_items(self, tmp_path):
        import json
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop
        from src.schemas import ToolGeneratorResponse

        loop = object.__new__(SelfImprovingLoop)
        loop._project_root = tmp_path
        loop.config = LoopConfig(validate_worked_examples=False)

        output = ToolGeneratorResponse(
            generated_skill="derived-numeric-contract",
            reasoning="Need a portable numeric contract.",
            skill_markdown=(
                "---\n"
                "name: derived-numeric-contract\n"
                "description: Validate derived numeric calculations.\n"
                "---\n\n"
                "# derived-numeric-contract\n"
            ),
            skill_tests=[
                {
                    "name": "negative direct lookup",
                    "scenario": "A source prints the final answer directly.",
                    "should_activate": False,
                    "expected_behavior": "Do not force a derivation contract.",
                    "must_check": ["final quantity is already printed"],
                    "must_reject": ["inventing arithmetic"],
                    "regression_guard": "Direct lookup answers remain concise.",
                }
            ],
        )

        skill_name = loop._materialize_generated_skill(
            output,
            action_type="create",
            target_skill=None,
            fallback_description="Validate derived numeric calculations.",
        )

        tests_path = tmp_path / ".claude" / "skills" / "derived-numeric-contract" / "SKILL_TESTS.json"
        assert skill_name == "derived-numeric-contract"
        payload = json.loads(tests_path.read_text())
        assert payload[0]["name"] == "negative direct lookup"
        assert payload[0]["should_activate"] is False


class TestExecutableSkillTests:
    def test_concrete_skill_test_query_uses_inline_task_without_expected_activation(self):
        from src.loop.runner import SelfImprovingLoop

        q = SelfImprovingLoop._skill_test_query(
            "source-lock",
            {
                "name": "mini table",
                "scenario": "Choose detailed table over summary.",
                "source_data": "| row | detailed | summary |\n| A | 12 | 9 |",
                "task": "Return the detailed value for A.",
                "should_activate": True,
                "expected_behavior": "Use detailed value.",
                "expected_answer": "12",
                "must_check": ["detailed"],
                "must_reject": ["summary"],
            },
        )

        assert "Source data:" in q
        assert "Task:" in q
        assert "Return the detailed value for A." in q
        assert "Expected activation" not in q
        assert "Potential available skill" not in q
        assert "source-lock" not in q

    def test_scores_activation_and_rubric_evidence(self):
        from src.harness.agent import AgentTrace
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop
        from src.schemas import AgentResponse

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        loop._project_root = __import__("pathlib").Path("/tmp/current-run")
        spec = {
            "name": "bind source",
            "scenario": "Two tables share labels; one has the wrong scale.",
            "should_activate": True,
            "expected_behavior": "Lock table title and unit scale.",
            "must_check": ["source_table_title", "source_unit_scale"],
            "must_reject": ["copying the summary table"],
        }
        trace = AgentTrace(
            duration_ms=1,
            total_cost_usd=0.0,
            num_turns=1,
            usage={},
            result=(
                "Used source-lock contract. Checked source_table_title and "
                "source_unit_scale; rejected copying the summary table."
            ),
            is_error=False,
            output=AgentResponse(
                final_answer=(
                    "Checked source_table_title and source_unit_scale; rejected "
                    "copying the summary table. Used source-lock contract."
                ),
                reasoning="",
            ),
            messages=[
                {
                    "type": "command_execution",
                    "command": "sed -n '1,80p' .claude/skills/source-lock/SKILL.md",
                    "aggregated_output": "---\nname: source-lock\n---\n",
                }
            ],
        )

        result = loop._score_executable_skill_test(
            skill_name="source-lock",
            spec=spec,
            trace=trace,
        )

        assert result.passed is True
        assert result.activated is True
        assert result.missing_checks == []
        assert result.missing_rejections == []

    def test_empty_answer_is_a_failed_self_test_not_an_exception(self):
        from pathlib import Path

        from src.harness.agent import AgentTrace
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        loop._project_root = Path("/tmp/current-run")
        trace = AgentTrace(
            duration_ms=0,
            total_cost_usd=0.0,
            num_turns=0,
            usage={},
            result="",
            is_error=True,
            parse_error="TimeoutError",
            messages=[],
        )

        result = loop._score_executable_skill_test(
            skill_name="numeric-check",
            spec={
                "name": "empty transport result",
                "task": "Compute 2 + 2.",
                "source_data": "2 + 2",
                "should_activate": True,
                "expected_behavior": "Return the sum.",
                "expected_answer": "4",
            },
            trace=trace,
        )

        assert result.passed is False
        assert "expected answer not produced" in result.reason

    def test_current_project_skill_read_is_not_self_test_isolation_violation(self):
        from pathlib import Path

        from src.harness.agent import AgentTrace
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop
        from src.schemas import AgentResponse

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        loop._project_root = Path("/tmp/current-run")
        spec = {
            "name": "mini diff",
            "source_data": "| row | detailed | summary |\n| A | 12 | 9 |",
            "task": "Return the detailed value for A.",
            "should_activate": True,
            "expected_behavior": "Use detailed value.",
            "expected_answer": "12",
        }
        trace = AgentTrace(
            duration_ms=1,
            total_cost_usd=0.0,
            num_turns=1,
            usage={},
            result="Using source-lock contract. 12",
            is_error=False,
            output=AgentResponse(final_answer="Using source-lock contract. 12", reasoning=""),
            messages=[
                {
                    "type": "command_execution",
                    "command": (
                        "sed -n '1,80p' "
                        "/tmp/current-run/.claude/skills/source-lock/SKILL.md"
                    ),
                    "aggregated_output": "---\nname: source-lock\n---\n",
                }
            ],
        )

        result = loop._score_executable_skill_test(
            skill_name="source-lock",
            spec=spec,
            trace=trace,
        )

        assert result.passed is True
        assert "isolation" not in result.reason

    def test_historical_tmp_skill_search_fails_self_test_isolation(self):
        from pathlib import Path

        from src.harness.agent import AgentTrace
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop
        from src.schemas import AgentResponse

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        loop._project_root = Path("/tmp/current-run")
        spec = {
            "name": "mini diff",
            "source_data": "| row | detailed | summary |\n| A | 12 | 9 |",
            "task": "Return the detailed value for A.",
            "should_activate": True,
            "expected_behavior": "Use detailed value.",
            "expected_answer": "12",
        }
        trace = AgentTrace(
            duration_ms=1,
            total_cost_usd=0.0,
            num_turns=1,
            usage={},
            result="Using source-lock contract. 12",
            is_error=False,
            output=AgentResponse(final_answer="Using source-lock contract. 12", reasoning=""),
            messages=[
                {
                    "type": "command_execution",
                    "command": (
                        "sed -n '1,80p' "
                        "/tmp/evoskill-officeqa-old/.claude/skills/source-lock/SKILL.md"
                    ),
                    "aggregated_output": "---\nname: source-lock\n---\n",
                }
            ],
        )

        result = loop._score_executable_skill_test(
            skill_name="source-lock",
            spec=spec,
            trace=trace,
        )

        assert result.passed is False
        assert "isolation" in result.reason

    def test_fails_when_skill_over_triggers_on_negative_case(self):
        from src.harness.agent import AgentTrace
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop
        from src.schemas import AgentResponse

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        spec = {
            "name": "plain lookup",
            "scenario": "A direct lookup with no ambiguity.",
            "should_activate": False,
            "expected_behavior": "Do not use the skill.",
            "must_check": [],
            "must_reject": [],
        }
        trace = AgentTrace(
            duration_ms=1,
            total_cost_usd=0.0,
            num_turns=1,
            usage={},
            result="Used source-lock contract anyway.",
            is_error=False,
            output=AgentResponse(final_answer="Used source-lock contract anyway.", reasoning=""),
            messages=[
                {
                    "type": "command_execution",
                    "command": "sed -n '1,80p' .claude/skills/source-lock/SKILL.md",
                    "aggregated_output": "---\nname: source-lock\n---\n",
                }
            ],
        )

        result = loop._score_executable_skill_test(
            skill_name="source-lock",
            spec=spec,
            trace=trace,
        )

        assert result.passed is False
        assert result.activated is True
        assert "activation mismatch" in result.reason

    def test_scores_concrete_task_by_expected_answer(self):
        from src.harness.agent import AgentTrace
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop
        from src.schemas import AgentResponse

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        spec = {
            "name": "mini diff",
            "scenario": "Detailed table beats summary.",
            "source_data": "| row | detailed | summary |\n| A | 12 | 9 |",
            "task": "Return the detailed value for A.",
            "should_activate": True,
            "expected_behavior": "Use detailed value.",
            "expected_answer": "12",
            "forbidden_outputs": ["9"],
        }
        trace = AgentTrace(
            duration_ms=1,
            total_cost_usd=0.0,
            num_turns=1,
            usage={},
            result="Used source-lock contract. 12",
            is_error=False,
            output=AgentResponse(final_answer="Used source-lock contract. 12", reasoning=""),
            messages=[
                {
                    "type": "command_execution",
                    "command": "sed -n '1,80p' .claude/skills/source-lock/SKILL.md",
                    "aggregated_output": "---\nname: source-lock\n---\n",
                }
            ],
        )

        result = loop._score_executable_skill_test(
            skill_name="source-lock",
            spec=spec,
            trace=trace,
        )

        assert result.passed is True

    def test_concrete_positive_passes_when_problem_solved_without_reported_activation(self):
        from src.harness.agent import AgentTrace
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop
        from src.schemas import AgentResponse

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        spec = {
            "name": "mini diff",
            "scenario": "Detailed table beats summary.",
            "source_data": "| row | detailed | summary |\n| A | 12 | 9 |",
            "task": "Return the detailed value for A.",
            "should_activate": True,
            "expected_behavior": "Use detailed value.",
            "expected_answer": "12",
            "must_check": ["detailed source"],
        }
        trace = AgentTrace(
            duration_ms=1,
            total_cost_usd=0.0,
            num_turns=1,
            usage={},
            result="The detailed value for A is 12.",
            is_error=False,
            output=AgentResponse(final_answer="12", reasoning=""),
            messages=[],
        )

        result = loop._score_executable_skill_test(
            skill_name="source-lock",
            spec=spec,
            trace=trace,
        )

        assert result.passed is True
        assert result.activated is False
        assert "primary signal" in result.reason

    def test_concrete_negative_does_not_require_rubric_recitation(self):
        from src.harness.agent import AgentTrace
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop
        from src.schemas import AgentResponse

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        spec = {
            "name": "plain sum",
            "scenario": "Explicit same-unit arithmetic.",
            "source_data": "",
            "task": "Compute 47 + 19.",
            "should_activate": False,
            "expected_behavior": "Answer directly.",
            "expected_answer": "66",
            "must_check": ["no ledger"],
            "must_reject": ["source binding"],
        }
        trace = AgentTrace(
            duration_ms=1,
            total_cost_usd=0.0,
            num_turns=1,
            usage={},
            result="66",
            is_error=False,
            output=AgentResponse(final_answer="66", reasoning=""),
            messages=[],
        )

        result = loop._score_executable_skill_test(
            skill_name="source-lock",
            spec=spec,
            trace=trace,
        )

        assert result.passed is True
        assert result.activated is False
        assert result.missing_checks == ["no ledger"]

    def test_concrete_negative_overtrigger_is_diagnostic_when_answer_is_correct(self):
        from src.harness.agent import AgentTrace
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop
        from src.schemas import AgentResponse

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        spec = {
            "name": "plain sum",
            "scenario": "Explicit same-unit arithmetic.",
            "task": "Compute 47 + 19.",
            "should_activate": False,
            "expected_behavior": "Answer directly.",
            "expected_answer": "66",
        }
        trace = AgentTrace(
            duration_ms=1,
            total_cost_usd=0.0,
            num_turns=1,
            usage={},
            result="Used source-lock contract. 47 + 19 = 66.",
            is_error=False,
            output=AgentResponse(final_answer="66", reasoning=""),
            messages=[],
        )

        result = loop._score_executable_skill_test(
            skill_name="source-lock",
            spec=spec,
            trace=trace,
        )

        assert result.passed is True
        assert result.activated is True
        assert "diagnostic over-trigger" in result.reason

    def test_concrete_positive_missing_rubric_is_diagnostic_only(self):
        from src.harness.agent import AgentTrace
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop
        from src.schemas import AgentResponse

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        spec = {
            "name": "span sum",
            "scenario": "Inclusive span sum.",
            "source_data": "| Week | North |\n| W1 | 10 |\n| W2 | 12 |\n| W3 | 15 |\n| W4 | 9 |",
            "task": "What is the total North value for W2 through W4?",
            "should_activate": True,
            "expected_behavior": "Sum W2-W4.",
            "expected_answer": "36",
            "must_check": ["explicitly says exclude W1"],
        }
        trace = AgentTrace(
            duration_ms=1,
            total_cost_usd=0.0,
            num_turns=1,
            usage={},
            result="Used source-lock contract. 12 + 15 + 9 = 36.",
            is_error=False,
            output=AgentResponse(
                final_answer="Used source-lock contract. 12 + 15 + 9 = 36.",
                reasoning="",
            ),
            messages=[],
        )

        result = loop._score_executable_skill_test(
            skill_name="source-lock",
            spec=spec,
            trace=trace,
        )

        assert result.passed is True
        assert result.missing_checks == ["explicitly says exclude W1"]

    def test_forbidden_outputs_only_check_final_answer_exactly(self):
        from src.harness.agent import AgentTrace
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop
        from src.schemas import AgentResponse

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        spec = {
            "name": "unit conversion",
            "scenario": "Convert then round.",
            "source_data": "A = 845 g",
            "task": "Convert A to kg, rounded to 2 decimals.",
            "should_activate": True,
            "expected_behavior": "Round at end.",
            "expected_answer": "0.85 kg",
            "forbidden_outputs": ["0.845 kg"],
        }
        trace = AgentTrace(
            duration_ms=1,
            total_cost_usd=0.0,
            num_turns=1,
            usage={},
            result="Used source-lock contract. 845 g = 0.845 kg, rounded to 0.85 kg.",
            is_error=False,
            output=AgentResponse(
                final_answer="0.85 kg",
                reasoning="845 g = 0.845 kg before rounding.",
            ),
            messages=[],
        )

        result = loop._score_executable_skill_test(
            skill_name="source-lock",
            spec=spec,
            trace=trace,
        )

        assert result.passed is True
        assert result.missing_rejections == []


class TestPickLeastShown:
    @staticmethod
    def _loop():
        from src.loop.runner import SelfImprovingLoop

        loop = object.__new__(SelfImprovingLoop)
        loop._failure_shown_count = {}
        return loop

    @staticmethod
    def _fail(qid):
        # 6-tuple shape used across the loop: (_, question, _, _, category, ftype)
        return (None, qid, "", "", "cat", "")

    def test_sweeps_distinct_failures_before_repeating(self):
        loop = self._loop()
        pool = [self._fail(f"q{i}") for i in range(4)]
        first = loop._pick_least_shown(pool, 2)
        second = loop._pick_least_shown(pool, 2)
        third = loop._pick_least_shown(pool, 2)
        assert [f[1] for f in first] == ["q0", "q1"]
        assert [f[1] for f in second] == ["q2", "q3"]  # unshown picked next
        # All shown once now; rotation repeats from the front.
        assert [f[1] for f in third] == ["q0", "q1"]

    def test_handles_empty_and_zero(self):
        loop = self._loop()
        assert loop._pick_least_shown([], 2) == []
        assert loop._pick_least_shown([self._fail("q0")], 0) == []


class TestSampleProposalFailuresBandit:
    """Phase 3 port: _sample_proposal_failures records bandit keys and biases by gain."""

    @staticmethod
    def _loop(failure_selection="bandit", categories=("a", "b")):
        from src.loop.runner import SelfImprovingLoop
        from src.loop.config import LoopConfig
        from src.loop.curriculum import CategoryBandit

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig(
            failure_selection=failure_selection,
            categories_per_batch=1,
            samples_per_category=1,
        )
        loop._failure_shown_count = {}
        loop._category_offset = 0
        loop._last_bandit_keys = []
        loop._bandit = CategoryBandit(list(categories), epsilon=0.05, ema=1.0)
        return loop

    @staticmethod
    def _fail(qid, cat):
        # 6-tuple: (_, question, _, _, category, failure_type="")
        return (None, qid, "", "", cat, "")

    def test_records_sampled_category_keys(self):
        loop = self._loop()
        pool = [self._fail("qa", "a"), self._fail("qb", "b")]
        loop._sample_proposal_failures(pool, ["a", "b"], failure_types=[])
        assert loop._last_bandit_keys  # populated
        assert set(loop._last_bandit_keys) <= {"a", "b"}

    def test_bandit_biases_toward_high_gain_category(self):
        loop = self._loop()
        # Make both categories "seen", then reward "a" strongly.
        loop._bandit.update("a", accepted=True, score_delta=0.9)
        loop._bandit.update("b", accepted=False, score_delta=0.0)
        pool = [self._fail("qa", "a"), self._fail("qb", "b")]
        picks = {"a": 0, "b": 0}
        for _ in range(200):
            loop._sample_proposal_failures(pool, ["a", "b"], failure_types=[])
            for k in loop._last_bandit_keys:
                picks[k] += 1
        assert picks["a"] > picks["b"] * 2

    def test_round_robin_mode_still_records_keys(self):
        loop = self._loop(failure_selection="round_robin")
        pool = [self._fail("qa", "a"), self._fail("qb", "b")]
        loop._sample_proposal_failures(pool, ["a", "b"], failure_types=[])
        assert loop._last_bandit_keys == ["a"]  # deterministic first category


class TestSampleInductionSuccesses:
    """Skill-induction sampling: prefer hard categories, rotate across calls."""

    @staticmethod
    def _loop(prefer_hard=True):
        from src.loop.runner import SelfImprovingLoop
        from src.loop.config import LoopConfig

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig(success_induction_prefer_hard=prefer_hard)
        loop._induction_offset = 0
        return loop

    @staticmethod
    def _succ(qid, cat):
        # 7-tuple: (trace, question, answer, gt, category, ftype, feedback)
        return (None, qid, "ans", "gt", cat, "regression_pass", "fb")

    def test_prefers_hard_category_success(self):
        loop = self._loop(prefer_hard=True)
        successes = [self._succ("se", "easy"), self._succ("sh", "hard")]
        # "hard" has many failures, "easy" has none -> a hard success is rarer/valuable.
        failures = [(None, f"f{i}", "", "", "hard", "") for i in range(5)]
        picked = loop._sample_induction_successes(successes, failures, count=1)
        assert [p[1] for p in picked] == ["sh"]

    def test_empty_when_no_successes(self):
        loop = self._loop()
        assert loop._sample_induction_successes([], [], count=2) == []

    def test_rotation_advances_offset(self):
        loop = self._loop(prefer_hard=False)
        successes = [self._succ("s0", "c"), self._succ("s1", "c"), self._succ("s2", "c")]
        first = loop._sample_induction_successes(successes, [], count=1)
        second = loop._sample_induction_successes(successes, [], count=1)
        assert first[0][1] != second[0][1]  # rotation picks a different success


class TestSelectBtAnchors:
    def test_at_most_parent_plus_frontier_best(self):
        from unittest.mock import MagicMock
        from src.loop.runner import SelfImprovingLoop

        loop = object.__new__(SelfImprovingLoop)
        loop.manager = MagicMock()
        loop.manager.get_best_from_frontier.return_value = "iter-skill-1"
        loop.manager.list_programs.return_value = [
            "base", "parent", "iter-skill-1", "iter-skill-2", "child",
        ]
        # Frontier intentionally rich; selection must still cap at 2 (parent+best).
        loop.manager.get_frontier_with_scores.return_value = [
            ("iter-skill-1", 0.7), ("iter-skill-2", 0.6), ("parent", 0.65),
        ]
        anchors = loop._select_bt_anchor_nodes("parent", "child")
        assert anchors == ["parent", "iter-skill-1"]
        assert "base" not in anchors  # base is the global BT anchor, not re-judged

    def test_single_anchor_when_best_is_parent(self):
        from unittest.mock import MagicMock
        from src.loop.runner import SelfImprovingLoop

        loop = object.__new__(SelfImprovingLoop)
        loop.manager = MagicMock()
        loop.manager.get_best_from_frontier.return_value = "parent"
        loop.manager.list_programs.return_value = ["base", "parent", "child"]
        loop.manager.get_frontier_with_scores.return_value = [("parent", 0.65)]
        assert loop._select_bt_anchor_nodes("parent", "child") == ["parent"]


class TestMeanRootCauseFitAndWeakest:
    @staticmethod
    def _result(matches):
        from types import SimpleNamespace

        return SimpleNamespace(matches=matches)

    @staticmethod
    def _match(score, fit, reasoning="", valid=True):
        from types import SimpleNamespace

        return SimpleNamespace(
            match_score=score,
            skill_addresses_root_cause=fit,
            reasoning=reasoning,
            valid=valid,
        )

    def test_mean_root_cause_fit_ignores_invalid(self):
        from src.loop.runner import SelfImprovingLoop

        res = self._result([
            self._match(0.6, 0.8),
            self._match(0.4, 0.2),
            self._match(0.9, 0.9, valid=False),  # ignored
        ])
        assert SelfImprovingLoop._mean_root_cause_fit(res) == pytest.approx(0.5)

    def test_mean_root_cause_fit_empty(self):
        from src.loop.runner import SelfImprovingLoop

        assert SelfImprovingLoop._mean_root_cause_fit(self._result([])) == 0.0

    def test_weakest_case_reasoning(self):
        from src.loop.runner import SelfImprovingLoop

        res = self._result([
            self._match(0.9, 0.8, reasoning="strong"),
            self._match(0.1, 0.2, reasoning="weak"),
        ])
        assert SelfImprovingLoop._weakest_case_reasoning(res) == "weak"


class TestJudgeAndRevisionQueries:
    def test_build_judge_query_includes_proposer_root_cause_and_fields(self):
        from src.loop.helpers import build_judge_query

        q = build_judge_query(
            "trace", "question?", "wrong", "expected",
            proposer_root_cause="agent read the wrong row",
            candidate_skill_tests='[{"name":"synthetic regression","should_activate":false}]',
        )
        assert "agent read the wrong row" in q
        assert "proposer_root_cause_correct" in q
        assert "skill_addresses_root_cause" in q
        assert "failure_mechanism_encoding" in q
        assert "executable_specificity" in q
        assert "high_risk_blacklist" in q
        assert "generalization_transfer" in q
        assert "Candidate-Generated Skill Tests" in q
        assert "self_test_pass_rate" in q
        assert "synthetic regression" in q
        assert "polished but generic skill text" in q
        assert "EXECUTABLE_SKILL_TEST_RESULTS" in q

    def test_build_judge_query_omits_section_when_no_root_cause(self):
        from src.loop.helpers import build_judge_query

        q = build_judge_query("trace", "question?", "wrong", "expected")
        assert "Proposer's Claimed Root Cause" not in q

    def test_artifact_quality_gate_caps_generic_candidate_win(self):
        from src.loop.runner import SelfImprovingLoop

        generic_quality = SelfImprovingLoop._judge_artifact_quality(
            {
                "skill_addresses_root_cause": 0.2,
                "failure_mechanism_encoding": 0.0,
                "executable_specificity": 0.0,
                "high_risk_blacklist": 0.0,
                "generalization_transfer": 0.0,
                "self_test_pass_rate": 0.0,
            }
        )
        assert generic_quality == pytest.approx(0.0)
        assert SelfImprovingLoop._apply_artifact_quality_gate(0.9, generic_quality) == pytest.approx(0.5)

        concrete_quality = SelfImprovingLoop._judge_artifact_quality(
            {
                "failure_mechanism_encoding": 1.0,
                "executable_specificity": 1.0,
                "high_risk_blacklist": 1.0,
                "generalization_transfer": 1.0,
                "self_test_pass_rate": 1.0,
            }
        )
        assert concrete_quality == pytest.approx(1.0)
        assert SelfImprovingLoop._apply_artifact_quality_gate(0.9, concrete_quality) == pytest.approx(0.9)

    def test_build_skill_revision_query_separates_intent_and_verdict(self):
        from src.loop.helpers import build_skill_revision_query

        q = build_skill_revision_query(
            target_skill="my-skill",
            current_skill_markdown="---\nname: my-skill\n---\nbody",
            proposer_root_cause="tried to fix the formula",
            original_proposal="formula gate",
            judge_root_cause="real divergence was wrong row selection",
            judge_blockers=["does not pin the row label"],
            judge_reasoning="contract built but row not constrained",
        )
        assert "REVISE existing skill: my-skill" in q
        assert "tried to fix the formula" in q          # proposer intent
        assert "real divergence was wrong row selection" in q  # judge verdict
        assert "does not pin the row label" in q         # blocker
        assert "name: my-skill" in q                     # current md echoed


# ===========================================================================
# Champion-gated staged dueling
# ===========================================================================

class TestChampionGateConfig:
    def test_defaults(self):
        from src.loop.config import LoopConfig

        c = LoopConfig()
        assert c.judge_champion_gate is True
        assert c.judge_stage2_anchors == 2
        assert c.judge_duel_delta == 0.1
        assert c.judge_duel_min_cases == 2


class TestDuelDecision:
    @staticmethod
    def _loop():
        from src.loop.runner import SelfImprovingLoop
        from src.loop.config import LoopConfig

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        return loop

    @staticmethod
    def _result(scores):
        from types import SimpleNamespace

        return SimpleNamespace(
            matches=[SimpleNamespace(match_score=s, valid=True) for s in scores]
        )

    def test_clear_win(self):
        assert self._loop()._duel_decision(self._result([0.9, 0.85, 0.88, 0.92])) == "win"

    def test_clear_loss(self):
        assert self._loop()._duel_decision(self._result([0.1, 0.15, 0.12, 0.08])) == "loss"

    def test_close_when_straddling_half(self):
        assert self._loop()._duel_decision(self._result([0.55, 0.45, 0.52, 0.48])) == "close"

    def test_too_few_cases_is_close(self):
        assert self._loop()._duel_decision(self._result([0.95])) == "close"

    def test_none_is_close(self):
        assert self._loop()._duel_decision(None) == "close"

    def test_std_floor_blocks_tiny_n_overconfidence(self):
        # Two near-identical scores barely above 0.5 must NOT be a confident win:
        # the std floor widens the CI so it stays close.
        assert self._loop()._duel_decision(self._result([0.51, 0.515])) == "close"


class TestRandomFrontierAnchors:
    def test_excludes_and_caps(self):
        from unittest.mock import MagicMock
        from src.loop.runner import SelfImprovingLoop

        loop = object.__new__(SelfImprovingLoop)
        loop.manager = MagicMock()
        loop.manager.list_programs.return_value = ["base", "a", "b", "c", "child", "champ"]
        loop.manager.get_frontier_with_scores.return_value = [
            ("champ", 0.8), ("a", 0.7), ("b", 0.6), ("c", 0.5), ("child", 0.55),
        ]
        picks = loop._random_frontier_anchors({"champ", "child"}, 2)
        assert len(picks) == 2
        assert "champ" not in picks and "child" not in picks
        assert set(picks).issubset({"a", "b", "c"})

    def test_zero_and_empty(self):
        from unittest.mock import MagicMock
        from src.loop.runner import SelfImprovingLoop

        loop = object.__new__(SelfImprovingLoop)
        loop.manager = MagicMock()
        loop.manager.list_programs.return_value = ["base", "champ", "child"]
        loop.manager.get_frontier_with_scores.return_value = [("champ", 0.8), ("child", 0.5)]
        assert loop._random_frontier_anchors({"champ", "child"}, 2) == []  # none left
        loop.manager.get_frontier_with_scores.return_value = [("a", 0.8)]
        assert loop._random_frontier_anchors(set(), 0) == []  # n=0


# ===========================================================================
# Randomized-slot position de-bias
# ===========================================================================

class TestJudgeOrientationDebias:
    def test_unswap_is_symmetric_across_slots(self):
        # The SAME underlying judgement ("candidate clearly better") must yield the
        # SAME canonical scores whether the candidate was shown in slot A or B.
        from src.loop.runner import SelfImprovingLoop as S

        as_b = S._canonicalize_judge_orientation(
            {"set_a_success_prob": 0.2, "set_b_success_prob": 0.9, "b_over_a_score": 0.85},
            "B",
        )
        as_a = S._canonicalize_judge_orientation(
            {"set_a_success_prob": 0.9, "set_b_success_prob": 0.2, "b_over_a_score": 0.15},
            "A",
        )
        for key in ("candidate_success_prob", "parent_success_prob", "match_score"):
            assert as_b[key] == pytest.approx(as_a[key])
        assert as_b["candidate_success_prob"] == pytest.approx(0.9)
        assert as_b["match_score"] == pytest.approx(0.85)
        assert as_b["would_succeed"] is True

    def test_candidate_in_a_inverts_b_over_a(self):
        from src.loop.runner import SelfImprovingLoop as S

        d = S._canonicalize_judge_orientation(
            {"set_a_success_prob": 0.3, "set_b_success_prob": 0.7, "b_over_a_score": 0.8},
            "A",
        )
        # candidate is Set A here, so match (candidate over parent) = 1 - 0.8 = 0.2
        assert d["candidate_success_prob"] == pytest.approx(0.3)
        assert d["parent_success_prob"] == pytest.approx(0.7)
        assert d["match_score"] == pytest.approx(0.2)
        assert d["would_succeed"] is False

    def test_randomize_orientation_default_on(self):
        from src.loop.config import LoopConfig

        c = LoopConfig()
        assert c.judge_randomize_orientation is True
        assert c.judge_position_swap is False


class TestJudgePromptNoDrawLean:
    def test_prompt_is_decisive_not_skeptical_default(self):
        from src.loop.helpers import build_judge_query

        q = build_judge_query("t", "q?", "wrong", "expected")
        # The 0.5-defaulting language must be gone; decisive language present.
        assert "stay near 0.5" not in q
        assert "Score above 0.5 only" not in q
        assert "decisive" in q
        assert "0.5 ONLY when" in q

    def test_compact_prompt_explicitly_labels_candidate_slot(self):
        from src.loop.helpers import build_compact_judge_query

        q = build_compact_judge_query(
            "trace",
            "question",
            "wrong",
            "expected",
            parent_skill_summary="candidate summary in A",
            candidate_skill_summary="parent summary in B",
            skill_diff="+ candidate change",
            candidate_slot="A",
        )

        assert "CANDIDATE containing the Change Being Evaluated is Skill Set A" in q
        assert "unchanged PARENT is Skill Set B" in q
        assert "Present in Skill Set A; absent from Skill Set B" in q


class TestPromptCompaction:
    def test_compact_relevant_text_keeps_matching_lines(self):
        from src.loop.helpers import compact_relevant_text

        text = "\n".join(
            ["unrelated filler " + str(i) for i in range(60)]
            + [
                "Tool call selected wrong artifact scope=v2",
                "Computed result from wrong artifact",
            ]
            + ["more unrelated filler " + str(i) for i in range(60)]
        )

        compacted = compact_relevant_text(
            text,
            "artifact scope wrong",
            max_chars=500,
            context_lines=0,
        )

        assert "wrong artifact" in compacted
        assert len(compacted) <= 650


class TestGenericEvolutionPrompts:
    def test_global_prompts_do_not_assume_officeqa_or_table_numeric_tasks(self):
        from src.agent_profiles.base_agent.prompt import BASE_AGENT_SYSTEM_PROMPT
        from src.agent_profiles.skill_generator.prompt import SKILL_GENERATOR_SYSTEM_PROMPT
        from src.agent_profiles.skill_proposer.prompt import SKILL_PROPOSER_SYSTEM_PROMPT
        from src.agent_profiles.skill_verifier.prompt import SKILL_VERIFIER_SYSTEM_PROMPT
        from src.loop.helpers import build_judge_query, build_proposer_query

        prompts = [
            BASE_AGENT_SYSTEM_PROMPT,
            SKILL_PROPOSER_SYSTEM_PROMPT,
            SKILL_GENERATOR_SYSTEM_PROMPT,
            SKILL_VERIFIER_SYSTEM_PROMPT,
            build_proposer_query([], "none"),
            build_judge_query("trace", "task", "wrong", "expected"),
        ]
        combined = "\n".join(prompts).lower()

        for forbidden in (
            "officeqa",
            "treasury",
            "top office firm",
            "source/table",
            "row/column",
            "same-unit",
            "rollup",
            "subcategory",
            "percentage-point",
            "numeric-workflow",
        ):
            assert forbidden not in combined

        for required in ("artifact", "tool", "state", "output contract"):
            assert required in combined


# ===========================================================================
# Scoring-accuracy + skill-completeness guards
# ===========================================================================

class TestScoringAndCompletenessConfig:
    def test_new_guard_defaults(self):
        from src.loop.config import LoopConfig

        c = LoopConfig()
        assert c.judge_max_output_tokens == 1536
        assert c.judge_regression_penalty_weight == 1.5
        assert c.judge_regression_sample_multiplier == 2.0
        assert c.discard_on_negative_self_test_failure is False
        assert c.puct_overlap_prior_penalty == 0.6
        assert c.puct_overlap_similarity_threshold == 0.35
        assert c.distill_frontier_at_end is True
        assert c.frontier_distill_top_k == 4
        assert c.skill_edit_min_retention_ratio == 0.5
        assert c.enforce_skill_sections is True
        assert c.max_active_skills == 8
        assert "Invariants" in c.required_skill_sections


class TestSkillSectionHelpers:
    def test_section_headers_lowercased(self):
        from src.loop.runner import SelfImprovingLoop

        md = "# When To Use\nx\n## Procedure\ny\n### Invariants\nz"
        assert SelfImprovingLoop._section_headers(md) == {
            "when to use",
            "procedure",
            "invariants",
        }

    def test_missing_required_sections(self):
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig()
        md = "## When To Use\na\n## Procedure\nb"
        missing = loop._missing_required_sections(md)
        assert "When To Use" not in missing
        assert "Procedure" not in missing
        assert "Invariants" in missing
        assert "Regression Risks" in missing

    def test_missing_required_sections_disabled_when_empty(self):
        from src.loop.runner import SelfImprovingLoop
        import types

        loop = object.__new__(SelfImprovingLoop)
        loop.config = types.SimpleNamespace(required_skill_sections=())
        assert loop._missing_required_sections("## Anything") == []


class TestMisleadingWorkedExampleGeneralized:
    def test_flags_grid_size_scaffold(self):
        from src.loop.runner import SelfImprovingLoop

        md = "grid_size: 5 x 4 = 20\nfinal_answer: 20"
        assert SelfImprovingLoop._has_misleading_worked_example(md) is True

    def test_distinct_answer_still_clean(self):
        from src.loop.runner import SelfImprovingLoop

        md = (
            "ledger_count_check: expected 7 categories * 2 periods = 14 threshold tests\n"
            "Answer: 3 categories\n"
        )
        assert SelfImprovingLoop._has_misleading_worked_example(md) is False


class TestProposerSkillBudget:
    def test_budget_reached_forces_edit(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        skills = tmp_path / ".claude" / "skills"
        for name in ("a", "b"):
            (skills / name).mkdir(parents=True)
            (skills / name / "SKILL.md").write_text("---\nname: %s\n---\n" % name)

        q = build_proposer_query(
            [], "none", project_root=tmp_path, max_active_skills=2
        )
        assert "active-skill budget is reached" in q
        assert 'action="edit"' in q

    def test_budget_not_reached_allows_create(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        skills = tmp_path / ".claude" / "skills"
        (skills / "a").mkdir(parents=True)
        (skills / "a" / "SKILL.md").write_text("---\nname: a\n---\n")

        q = build_proposer_query(
            [], "none", project_root=tmp_path, max_active_skills=8
        )
        assert "active-skill budget is reached" not in q


class TestRevisionCompletenessDirectives:
    def test_structural_and_self_test_gaps_appended(self):
        from src.loop.helpers import build_skill_revision_query

        q = build_skill_revision_query(
            target_skill="s",
            current_skill_markdown="md",
            proposer_root_cause="rc",
            original_proposal="prop",
            judge_root_cause="jrc",
            judge_blockers=["b"],
            structural_gaps=["Invariants", "Procedure"],
            self_test_gap=True,
        )
        assert "Completeness Corrections" in q
        assert "Invariants, Procedure" in q
        assert "negative (should_activate=false)" in q

    def test_executable_self_test_feedback_appended(self):
        from src.loop.helpers import build_skill_revision_query

        q = build_skill_revision_query(
            target_skill="s",
            current_skill_markdown="md",
            proposer_root_cause="rc",
            original_proposal="prop",
            judge_root_cause="jrc",
            judge_blockers=["b"],
            executable_self_test_feedback=(
                "actual_pass_rate: 0.400 (2/5)\n"
                "- FAIL s/negative-boundary: should_activate=False, activated=True"
            ),
        )
        assert "Executable Self-Test Feedback" in q
        assert "actual_pass_rate: 0.400" in q
        assert "should_activate=False, activated=True" in q

    def test_no_directives_when_complete(self):
        from src.loop.helpers import build_skill_revision_query

        q = build_skill_revision_query(
            target_skill="s",
            current_skill_markdown="md",
            proposer_root_cause="rc",
            original_proposal="prop",
            judge_root_cause="jrc",
            judge_blockers=["b"],
            structural_gaps=[],
            self_test_gap=False,
        )
        assert "Completeness Corrections" not in q


class TestExecutableSelfTestRevisionFeedback:
    def test_failed_negative_self_test_is_detected(self):
        from src.loop.runner import (
            ExecutableSkillTestCaseResult,
            ExecutableSkillTestSuiteResult,
            SelfImprovingLoop,
        )

        result = ExecutableSkillTestSuiteResult(
            passed=1,
            total=2,
            pass_rate=0.5,
            cases=[
                ExecutableSkillTestCaseResult(
                    skill_name="s",
                    test_name="positive",
                    should_activate=True,
                    activated=True,
                    passed=True,
                    reason="ok",
                ),
                ExecutableSkillTestCaseResult(
                    skill_name="s",
                    test_name="negative",
                    should_activate=False,
                    activated=True,
                    passed=False,
                    reason="over-triggered",
                ),
            ],
        )

        assert SelfImprovingLoop._has_failed_negative_self_test(result) is True
        assert SelfImprovingLoop._has_failed_negative_self_test(None) is False

    def test_passed_negative_overtrigger_is_not_a_hard_failure(self):
        from src.loop.runner import (
            ExecutableSkillTestCaseResult,
            ExecutableSkillTestSuiteResult,
            SelfImprovingLoop,
        )

        result = ExecutableSkillTestSuiteResult(
            passed=1,
            total=1,
            pass_rate=1.0,
            cases=[
                ExecutableSkillTestCaseResult(
                    skill_name="s",
                    test_name="negative-correct-answer",
                    should_activate=False,
                    activated=True,
                    passed=True,
                    reason="concrete task solved; diagnostic over-trigger",
                ),
            ],
        )

        assert SelfImprovingLoop._has_failed_negative_self_test(result) is False

    def test_compacts_failures_for_revision_prompt(self):
        from src.loop.runner import (
            ExecutableSkillTestCaseResult,
            ExecutableSkillTestSuiteResult,
            SelfImprovingLoop,
        )

        result = ExecutableSkillTestSuiteResult(
            passed=1,
            total=2,
            pass_rate=0.5,
            cases=[
                ExecutableSkillTestCaseResult(
                    skill_name="s",
                    test_name="positive",
                    should_activate=True,
                    activated=True,
                    passed=True,
                    reason="ok",
                ),
                ExecutableSkillTestCaseResult(
                    skill_name="s",
                    test_name="negative",
                    should_activate=False,
                    activated=True,
                    passed=False,
                    reason="activated on direct lookup",
                    missing_checks=["direct lookup boundary"],
                    missing_rejections=["do not build ledger"],
                    agent_answer="built a long protocol block anyway",
                ),
            ],
        )

        text = SelfImprovingLoop._self_test_revision_feedback(result)
        assert "actual_pass_rate: 0.500 (1/2)" in text
        assert "- FAIL s/negative" in text
        assert "should_activate=False, activated=True" in text
        assert "missing_checks: direct lookup boundary" in text
        assert "missing_rejections: do not build ledger" in text
        assert "agent_answer: built a long protocol block anyway" in text


class TestPuctProposalDiversity:
    def test_existing_children_advance_diversity_strategy_across_iterations(self):
        from src.loop.runner import ProgramSearchNode, SelfImprovingLoop

        parent = ProgramSearchNode("base", None)
        prior_child = ProgramSearchNode(
            "iter-skill-1",
            parent,
            proposal="Edit source binding gate",
            proposal_action="edit",
            proposal_skill="source-binding-gate",
        )

        hint = SelfImprovingLoop._build_child_diversity_hint(
            0,
            [],
            existing_children=[prior_child],
        )

        assert "alternative root-cause hypothesis" in hint
        assert "Earlier child actions: edit" in hint
        assert "source-binding-gate" in hint

    def test_overlapping_proposal_gets_lower_prior(self):
        from src.loop.config import LoopConfig
        from src.loop.runner import SelfImprovingLoop

        loop = object.__new__(SelfImprovingLoop)
        loop.config = LoopConfig(
            puct_overlap_similarity_threshold=0.1,
            puct_overlap_prior_penalty=0.8,
        )
        proposal = "Bind the exact source table row and column before arithmetic."
        unrelated = "Choose a browser download directory for generated reports."

        overlapping = loop._proposal_policy_prior(
            0.8,
            proposal=proposal,
            existing_skill_content=proposal,
        )
        distinct = loop._proposal_policy_prior(
            0.8,
            proposal=proposal,
            existing_skill_content=unrelated,
        )

        assert overlapping < distinct
        assert distinct == pytest.approx(0.8)
