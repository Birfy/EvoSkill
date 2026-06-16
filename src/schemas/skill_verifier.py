from pydantic import BaseModel, Field

from src.schemas.tool_generator import SkillTest


class SkillVerifierResponse(BaseModel):
    """Output of the information-isolated skill verifier (CoEvoSkills-style).

    A separate agent from the skill generator authors adversarial test cases for a
    candidate skill WITHOUT seeing ground-truth answers, so the tests cannot be
    gamed by whoever wrote the skill. The tests reuse the SkillTest schema and are
    consumed by the existing executable-test / judge self-test-pass-rate path.
    """

    tests: list[SkillTest] = Field(default_factory=list)
    """Independently authored test cases probing activation, over-triggering, and
    whether the skill actually performs its claimed procedure."""

    probe_reasoning: str = ""
    """How the verifier tried to falsify the skill's claims (fed to refine)."""
