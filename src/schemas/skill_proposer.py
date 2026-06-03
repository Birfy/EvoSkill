from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BulletOp(BaseModel):
    """One edit to a skill's ACE-style bullet playbook (Phase 1 playbook mode).

    Proposals are applied as small deltas instead of monolithic rewrites:
    - ``add``: introduce a new rule (``text`` required); deduped against existing
      bullets, reinforcing the closest match if near-identical.
    - ``reinforce``: bump an existing bullet's helpful counter (``target_id``).
    - ``retire``: remove a bullet that no longer helps (``target_id``).
    """

    op: Literal["add", "reinforce", "retire"] = "add"
    target_id: str | None = None
    text: str = ""

    @model_validator(mode="after")
    def validate_op(self) -> "BulletOp":
        if self.op == "add" and not self.text.strip():
            raise ValueError("bullet op 'add' requires non-empty text")
        if self.op in ("reinforce", "retire") and not (self.target_id or "").strip():
            raise ValueError(f"bullet op '{self.op}' requires target_id")
        return self


class SkillProposerResponse(BaseModel):
    """Response from the skill proposer agent.

    This proposer analyzes agent failures and proposes skill additions
    or modifications to existing skills to address capability gaps.
    """

    action: Literal["create", "edit"] = "create"
    """Whether to create a new skill or edit an existing one."""

    target_skill: str | None = None
    """Name of existing skill to modify. Required if action="edit"."""

    proposed_skill: str
    """High-level description of the skill needed or modifications to make."""

    justification: str
    """Explanation of why this skill/modification addresses the identified gap."""

    root_cause_analysis: str = ""
    """Per-failure root cause analysis that led to the proposal."""

    coverage_plan: str = ""
    """How the proposal covers each sampled failure and the common pattern."""

    should_apply_when: str
    """Concrete conditions where the generated skill should be used."""

    should_not_apply_when: str
    """Concrete conditions where the generated skill must not be used."""

    invariants_to_preserve: str
    """Existing correct behavior, bindings, units, formulas, or rounding rules that must not change."""

    regression_risks: str
    """Known ways the proposal could regress previously correct behavior."""

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    """Proposer confidence that this change addresses the observed failures."""

    related_iterations: list[str] = Field(default_factory=list)
    """List of relevant past iterations referenced in the proposal (e.g., ["iter-4", "iter-9"])."""

    bullet_ops: list[BulletOp] = Field(default_factory=list)
    """Playbook delta (Phase 1 playbook mode). When non-empty and use_playbook is
    on, these add/reinforce/retire ops are applied deterministically to the target
    skill's bullet playbook instead of regenerating the whole SKILL.md."""

    @model_validator(mode="after")
    def validate_required_fields(self) -> "SkillProposerResponse":
        if self.action == "edit" and not self.target_skill:
            raise ValueError("target_skill is required when action='edit'")
        required_boundary_fields = {
            "should_apply_when": self.should_apply_when,
            "should_not_apply_when": self.should_not_apply_when,
            "invariants_to_preserve": self.invariants_to_preserve,
            "regression_risks": self.regression_risks,
        }
        missing = [
            name
            for name, value in required_boundary_fields.items()
            if not value or not value.strip()
        ]
        if missing:
            raise ValueError(
                "skill boundary fields are required and must be non-empty: "
                + ", ".join(missing)
            )
        return self
