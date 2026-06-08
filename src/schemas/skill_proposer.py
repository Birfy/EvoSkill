from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SuspectSkill(BaseModel):
    """A skill implicated in a failure, with attribution weight and evidence category.

    Weight ranges follow SkillAdaptor convention:
      0.8–1.0  fully responsible
      0.5–0.7  partially responsible
      0.2–0.4  weakly related
      0.0–0.1  irrelevant / coincidentally active
    """

    skill_name: str
    weight: float = Field(ge=0.0, le=1.0)
    reason: Literal[
        "direct_instruction",   # skill explicitly told the agent to do the wrong thing
        "context_mismatch",     # skill fired in a context it shouldn't apply to
        "omission",             # skill omitted a step the agent needed
        "misleading_wording",   # skill wording led agent to a wrong interpretation
    ]


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
            if self.text.strip():
                # Models sometimes describe a new rule but label it reinforce.
                # Preserve the usable proposal by treating that payload as add.
                self.op = "add"
            else:
                raise ValueError(f"bullet op '{self.op}' requires target_id")
        return self


class SkillEdit(BaseModel):
    """One skill create/edit within a (possibly multi-skill) proposal.

    A single proposal may target several distinct skills at once; each entry is
    independently generated, validated, and (in playbook mode) gets its own
    bullet delta applied to its own SKILL.md.
    """

    action: Literal["create", "edit"] = "create"
    target_skill: str | None = None
    proposed_skill: str = ""
    should_apply_when: str = ""
    should_not_apply_when: str = ""
    invariants_to_preserve: str = ""
    regression_risks: str = ""
    bullet_ops: list[BulletOp] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_edit(self) -> "SkillEdit":
        if self.action == "edit" and not self.target_skill:
            raise ValueError("target_skill is required when action='edit'")
        if not self.proposed_skill.strip():
            raise ValueError("proposed_skill is required for each skill edit")
        missing = [
            name
            for name in (
                "should_apply_when",
                "should_not_apply_when",
                "invariants_to_preserve",
                "regression_risks",
            )
            if not (getattr(self, name) or "").strip()
        ]
        if missing:
            raise ValueError(
                "skill edit boundary fields are required and must be non-empty: "
                + ", ".join(missing)
            )
        return self


class SkillProposerResponse(BaseModel):
    """Response from the skill proposer agent.

    This proposer analyzes agent failures and proposes skill additions
    or modifications to existing skills to address capability gaps.

    A proposal may target a SINGLE skill (the top-level action/target_skill/
    proposed_skill/boundary fields — the common case) or MULTIPLE skills at once
    via ``skill_edits``. When ``skill_edits`` is non-empty, its first entry is
    mirrored into the top-level fields (so the single-skill code path handles the
    primary edit unchanged) and the remaining entries are applied as extra skills
    to the same child program.
    """

    action: Literal["create", "edit"] = "create"
    """Whether to create a new skill or edit an existing one."""

    target_skill: str | None = None
    """Name of existing skill to modify. Required if action="edit"."""

    proposed_skill: str = ""
    """High-level description of the skill needed or modifications to make."""

    justification: str
    """Explanation of why this skill/modification addresses the identified gap."""

    root_cause_analysis: str = ""
    """Per-failure root cause analysis that led to the proposal."""

    coverage_plan: str = ""
    """How the proposal covers each sampled failure and the common pattern."""

    should_apply_when: str = ""
    """Concrete conditions where the generated skill should be used."""

    should_not_apply_when: str = ""
    """Concrete conditions where the generated skill must not be used."""

    invariants_to_preserve: str = ""
    """Existing correct behavior, bindings, units, formulas, or rounding rules that must not change."""

    regression_risks: str = ""
    """Known ways the proposal could regress previously correct behavior."""

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    """Proposer confidence that this change addresses the observed failures."""

    fault_type: Literal["skill_wrong", "skill_missing"] = "skill_missing"
    """Whether an active skill caused the error (skill_wrong → EDIT) or no skill
    covered the scenario (skill_missing → CREATE)."""

    suspect_skills: list[SuspectSkill] = Field(default_factory=list)
    """Skills implicated in the failure, ordered by descending weight.
    Non-empty when fault_type=skill_wrong; action=edit targets the highest-weight entry."""

    related_iterations: list[str] = Field(default_factory=list)
    """List of relevant past iterations referenced in the proposal (e.g., ["iter-4", "iter-9"])."""

    bullet_ops: list[BulletOp] = Field(default_factory=list)
    """Playbook delta (Phase 1 playbook mode). When non-empty and use_playbook is
    on, these add/reinforce/retire ops are applied deterministically to the target
    skill's bullet playbook instead of regenerating the whole SKILL.md."""

    skill_edits: list[SkillEdit] = Field(default_factory=list)
    """Optional: when this proposal targets MULTIPLE distinct skills, list each as
    a SkillEdit. The first is mirrored into the top-level fields; extras are applied
    additionally. Leave empty for a single-skill proposal."""

    def extra_edits(self) -> list[SkillEdit]:
        """Edits beyond the primary (top-level) one, applied as additional skills."""
        return list(self.skill_edits[1:]) if self.skill_edits else []

    @model_validator(mode="after")
    def validate_required_fields(self) -> "SkillProposerResponse":
        # Multi-skill: mirror the first edit into the top-level fields so the
        # single-skill code path handles the primary edit unchanged.
        if self.skill_edits:
            primary = self.skill_edits[0]
            self.action = primary.action
            self.target_skill = primary.target_skill
            if not self.proposed_skill.strip():
                self.proposed_skill = primary.proposed_skill
            for name in (
                "should_apply_when",
                "should_not_apply_when",
                "invariants_to_preserve",
                "regression_risks",
            ):
                if not (getattr(self, name) or "").strip():
                    setattr(self, name, getattr(primary, name))

        if self.action == "edit" and not self.target_skill:
            raise ValueError("target_skill is required when action='edit'")
        if not self.proposed_skill.strip():
            raise ValueError("proposed_skill is required")
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
