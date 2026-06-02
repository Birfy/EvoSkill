"""Tests for src/loop/ — LoopConfig, helpers (build_proposer_query, feedback management)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock


# ===========================================================================
# LoopConfig — dataclass defaults and field types
# ===========================================================================

class TestLoopConfig:
    def test_default_max_iterations(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.max_iterations == 5

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
        )
        assert "agent read the wrong row" in q
        assert "proposer_root_cause_correct" in q
        assert "skill_addresses_root_cause" in q
        assert "failure_mechanism_encoding" in q
        assert "executable_specificity" in q
        assert "high_risk_blacklist" in q
        assert "generalization_transfer" in q
        assert "polished but generic skill text" in q

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
