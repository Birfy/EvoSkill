"""Tests for src/schemas/ — Pydantic model validation, defaults, and required fields."""

import pytest
from pydantic import ValidationError


def _skill_boundary_fields() -> dict[str, str]:
    return {
        "should_apply_when": "Use when numeric table extraction needs unit conversion or binding checks.",
        "should_not_apply_when": "Do not use when the source already matches the requested units and binding is explicit.",
        "invariants_to_preserve": "Preserve explicit row, column, period, formula, unit, and rounding instructions.",
        "regression_risks": "Could over-apply to already-normalized values; keep the trigger narrow.",
    }


# ===========================================================================
# AgentResponse
# ===========================================================================

class TestAgentResponse:
    def test_valid_construction(self):
        from src.schemas import AgentResponse

        resp = AgentResponse(final_answer="42", reasoning="Because math.")
        assert resp.final_answer == "42"
        assert resp.reasoning == "Because math."

    def test_missing_final_answer_raises(self):
        from src.schemas import AgentResponse

        with pytest.raises(ValidationError):
            AgentResponse(reasoning="no answer provided")  # type: ignore[call-arg]

    def test_missing_reasoning_raises(self):
        from src.schemas import AgentResponse

        with pytest.raises(ValidationError):
            AgentResponse(final_answer="42")  # type: ignore[call-arg]

    def test_empty_strings_accepted(self):
        from src.schemas import AgentResponse

        resp = AgentResponse(final_answer="", reasoning="")
        assert resp.final_answer == ""

    def test_long_strings_accepted(self):
        from src.schemas import AgentResponse

        long_str = "x" * 10_000
        resp = AgentResponse(final_answer=long_str, reasoning=long_str)
        assert len(resp.final_answer) == 10_000

    def test_json_serialization_round_trip(self):
        from src.schemas import AgentResponse

        original = AgentResponse(final_answer="42", reasoning="Because math.")
        dumped = original.model_dump()
        restored = AgentResponse.model_validate(dumped)
        assert restored == original


# ===========================================================================
# StoredTrajectory
# ===========================================================================

class TestStoredTrajectory:
    def test_extended_tuple_includes_failure_feedback(self):
        from src.schemas.trajectory import StoredTrajectory

        trajectory = StoredTrajectory(
            question="What is the total?",
            ground_truth="100",
            category="finance",
            agent_answer="42",
            failure_type="numeric_mismatch",
            failure_feedback="predicted 42 but expected 100",
        )

        extended = trajectory.to_extended_tuple()
        assert extended[1] == "What is the total?"
        assert extended[5] == "numeric_mismatch"
        assert extended[6] == "predicted 42 but expected 100"

    def test_extracts_codex_skill_file_reads(self):
        from src.schemas.trajectory import _extract_skill_calls

        messages = [
            """{"items":[{"type":"command_execution","command":"/bin/bash -lc \\"sed -n '1,80p' /repo/.claude/skills/officeqa-aggregation-lock/SKILL.md\\"","aggregated_output":"---\\nname: officeqa-aggregation-lock\\n---"}]}"""
        ]

        calls = _extract_skill_calls(messages)

        assert calls == [
            {
                "message_index": 0,
                "event_type": "codex_skill_file_read",
                "skill_name": "officeqa-aggregation-lock",
                "payload": {
                    "command": "/bin/bash -lc \"sed -n '1,80p' /repo/.claude/skills/officeqa-aggregation-lock/SKILL.md\""
                },
                "text": "/bin/bash -lc \"sed -n '1,80p' /repo/.claude/skills/officeqa-aggregation-lock/SKILL.md\"",
            }
        ]


# ===========================================================================
# ProposerResponse
# ===========================================================================

class TestProposerResponse:
    def test_valid_prompt_mode(self):
        from src.schemas import ProposerResponse

        resp = ProposerResponse(
            optimize_prompt_or_skill="prompt",
            proposed_skill_or_prompt="Improve CoT reasoning",
            justification="Agent failed on multi-step problems",
        )
        assert resp.optimize_prompt_or_skill == "prompt"

    def test_valid_skill_mode(self):
        from src.schemas import ProposerResponse

        resp = ProposerResponse(
            optimize_prompt_or_skill="skill",
            proposed_skill_or_prompt="Build a numeric extraction skill",
            justification="Agent missed decimal numbers",
        )
        assert resp.optimize_prompt_or_skill == "skill"

    def test_invalid_mode_raises(self):
        from src.schemas import ProposerResponse

        with pytest.raises(ValidationError):
            ProposerResponse(
                optimize_prompt_or_skill="invalid_value",  # type: ignore[arg-type]
                proposed_skill_or_prompt="something",
                justification="reason",
            )

    def test_missing_justification_raises(self):
        from src.schemas import ProposerResponse

        with pytest.raises(ValidationError):
            ProposerResponse(
                optimize_prompt_or_skill="skill",
                proposed_skill_or_prompt="something",
                # justification missing
            )  # type: ignore[call-arg]


# ===========================================================================
# SkillProposerResponse
# ===========================================================================

class TestMultiSkillProposal:
    @staticmethod
    def _edit(**kw):
        from src.schemas import SkillEdit

        base = dict(
            action="create",
            proposed_skill="s",
            should_apply_when="a",
            should_not_apply_when="b",
            invariants_to_preserve="c",
            regression_risks="d",
        )
        base.update(kw)
        return SkillEdit(**base)

    def test_skill_edits_mirror_primary_into_top_level(self):
        from src.schemas import SkillProposerResponse

        r = SkillProposerResponse(
            justification="j",
            skill_edits=[
                self._edit(action="edit", target_skill="foo", proposed_skill="p1"),
                self._edit(action="create", proposed_skill="p2"),
            ],
        )
        assert r.action == "edit"
        assert r.target_skill == "foo"
        assert r.proposed_skill == "p1"
        assert len(r.extra_edits()) == 1
        assert r.extra_edits()[0].proposed_skill == "p2"

    def test_single_skill_still_works_and_has_no_extras(self):
        from src.schemas import SkillProposerResponse

        r = SkillProposerResponse(
            proposed_skill="x",
            justification="j",
            should_apply_when="a",
            should_not_apply_when="b",
            invariants_to_preserve="c",
            regression_risks="d",
        )
        assert r.extra_edits() == []

    def test_skill_edit_edit_requires_target(self):
        from src.schemas import SkillEdit

        with pytest.raises(ValidationError):
            SkillEdit(
                action="edit",
                proposed_skill="s",
                should_apply_when="a",
                should_not_apply_when="b",
                invariants_to_preserve="c",
                regression_risks="d",
            )

    def test_skill_edit_requires_boundaries(self):
        from src.schemas import SkillEdit

        with pytest.raises(ValidationError):
            SkillEdit(action="create", proposed_skill="s")


class TestSkillProposerResponse:
    def test_default_action_is_create(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            proposed_skill="A numeric comparison skill",
            justification="Agent misses unit differences",
            **_skill_boundary_fields(),
        )
        assert resp.action == "create"

    def test_edit_action_with_target_skill(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            action="edit",
            target_skill="numeric-extraction",
            proposed_skill="Extend to handle percentages",
            justification="Missing percentage support",
            **_skill_boundary_fields(),
        )
        assert resp.action == "edit"
        assert resp.target_skill == "numeric-extraction"

    def test_target_skill_defaults_to_none(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            proposed_skill="New skill",
            justification="New capability needed",
            **_skill_boundary_fields(),
        )
        assert resp.target_skill is None

    def test_related_iterations_defaults_to_empty_list(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            proposed_skill="New skill",
            justification="reason",
            **_skill_boundary_fields(),
        )
        assert resp.related_iterations == []

    def test_optional_analysis_fields_default_to_empty_strings(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            proposed_skill="New skill",
            justification="reason",
            **_skill_boundary_fields(),
        )
        assert resp.root_cause_analysis == ""
        assert resp.coverage_plan == ""
        assert resp.regression_risks

    def test_related_iterations_accepts_list(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            proposed_skill="New skill",
            justification="reason",
            related_iterations=["iter-1", "iter-5"],
            root_cause_analysis="Failure 1 diverged at row binding.",
            coverage_plan="Covers failures 1 and 2.",
            **{
                **_skill_boundary_fields(),
                "regression_risks": "Could over-apply to decimal values.",
            },
        )
        assert resp.related_iterations == ["iter-1", "iter-5"]
        assert "row binding" in resp.root_cause_analysis
        assert "failures 1 and 2" in resp.coverage_plan
        assert "over-apply" in resp.regression_risks

    def test_reinforce_without_target_id_but_with_text_becomes_add(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            proposed_skill="New skill",
            justification="reason",
            bullet_ops=[
                {
                    "op": "reinforce",
                    "text": "Check source scope before extracting values.",
                }
            ],
            **_skill_boundary_fields(),
        )

        assert resp.bullet_ops[0].op == "add"
        assert "source scope" in resp.bullet_ops[0].text

    def test_invalid_action_raises(self):
        from src.schemas import SkillProposerResponse

        with pytest.raises(ValidationError):
            SkillProposerResponse(
                action="delete",  # type: ignore[arg-type]
                proposed_skill="something",
                justification="reason",
                **_skill_boundary_fields(),
            )

    def test_edit_action_requires_target_skill(self):
        from src.schemas import SkillProposerResponse

        with pytest.raises(ValidationError, match="target_skill is required"):
            SkillProposerResponse(
                action="edit",
                proposed_skill="Extend existing skill",
                justification="Need to update current behavior",
                **_skill_boundary_fields(),
            )

    def test_missing_proposed_skill_raises(self):
        from src.schemas import SkillProposerResponse

        with pytest.raises(ValidationError):
            SkillProposerResponse(justification="reason", **_skill_boundary_fields())  # type: ignore[call-arg]

    def test_missing_skill_boundary_fields_raises(self):
        from src.schemas import SkillProposerResponse

        with pytest.raises(ValidationError, match="skill boundary fields"):
            SkillProposerResponse(
                proposed_skill="New skill",
                justification="reason",
                should_apply_when="Use for unit conversion cases.",
                should_not_apply_when="Do not use for already-normalized values.",
                invariants_to_preserve="Preserve explicit units and rounding.",
                regression_risks="",
            )


# ===========================================================================
# PromptProposerResponse
# ===========================================================================

class TestPromptProposerResponse:
    def test_valid_construction(self):
        from src.schemas import PromptProposerResponse

        resp = PromptProposerResponse(
            proposed_prompt_change="Add step-by-step reasoning instructions",
            justification="Agent skips intermediate steps",
        )
        assert "step-by-step" in resp.proposed_prompt_change

    def test_missing_proposed_prompt_change_raises(self):
        from src.schemas import PromptProposerResponse

        with pytest.raises(ValidationError):
            PromptProposerResponse(justification="reason")  # type: ignore[call-arg]

    def test_missing_justification_raises(self):
        from src.schemas import PromptProposerResponse

        with pytest.raises(ValidationError):
            PromptProposerResponse(proposed_prompt_change="change")  # type: ignore[call-arg]


# ===========================================================================
# ToolGeneratorResponse
# ===========================================================================

class TestToolGeneratorResponse:
    def test_valid_construction(self):
        from src.schemas import ToolGeneratorResponse

        resp = ToolGeneratorResponse(
            generated_skill="# SKILL\n\nExtract numbers from text.",
            reasoning="The agent needs number extraction.",
            skill_tests=[
                {
                    "name": "synthetic extraction boundary",
                    "scenario": "Question requires copying one labeled number.",
                    "source_data": "| label | value |\n| A | 4 |",
                    "task": "Return A.",
                    "should_activate": True,
                    "expected_behavior": "Bind the exact label before answering.",
                    "expected_answer": "4",
                }
            ],
        )
        assert "SKILL" in resp.generated_skill
        assert resp.skill_tests is not None
        assert resp.skill_tests[0].should_activate is True
        assert resp.skill_tests[0].task == "Return A."
        assert resp.skill_tests[0].expected_answer == "4"

    def test_missing_generated_skill_raises(self):
        from src.schemas import ToolGeneratorResponse

        with pytest.raises(ValidationError):
            ToolGeneratorResponse(reasoning="reason")  # type: ignore[call-arg]

    def test_missing_reasoning_raises(self):
        from src.schemas import ToolGeneratorResponse

        with pytest.raises(ValidationError):
            ToolGeneratorResponse(generated_skill="skill text")  # type: ignore[call-arg]


# ===========================================================================
# PromptGeneratorResponse
# ===========================================================================

class TestPromptGeneratorResponse:
    def test_valid_construction(self):
        from src.schemas import PromptGeneratorResponse

        resp = PromptGeneratorResponse(
            optimized_prompt="You are a helpful agent. Always show your reasoning.",
            reasoning="Added explicit reasoning instruction.",
        )
        assert "helpful" in resp.optimized_prompt

    def test_missing_optimized_prompt_raises(self):
        from src.schemas import PromptGeneratorResponse

        with pytest.raises(ValidationError):
            PromptGeneratorResponse(reasoning="reason")  # type: ignore[call-arg]

    def test_missing_reasoning_raises(self):
        from src.schemas import PromptGeneratorResponse

        with pytest.raises(ValidationError):
            PromptGeneratorResponse(optimized_prompt="prompt text")  # type: ignore[call-arg]

    def test_json_round_trip(self):
        from src.schemas import PromptGeneratorResponse

        original = PromptGeneratorResponse(
            optimized_prompt="prompt", reasoning="reasoning"
        )
        restored = PromptGeneratorResponse.model_validate(original.model_dump())
        assert restored == original
