"""Static acceptance gate for skill proposals."""

from __future__ import annotations

from dataclasses import dataclass

from .evolver import SkillProposal


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    score: float


class ParetoGate:
    """Two-part gate: LLM score improvement plus SKILL.md validity."""

    def __init__(self, baseline_score: float = 0.5, min_score_delta: float = 0.05):
        self.baseline_score = baseline_score
        self.min_score_delta = min_score_delta

    def evaluate(self, proposal: SkillProposal, new_q_score: float) -> GateDecision:
        if new_q_score < self.baseline_score + self.min_score_delta:
            return GateDecision(
                accepted=False,
                reason=(
                    f"Score {new_q_score:.3f} < baseline "
                    f"{self.baseline_score:.3f} + delta {self.min_score_delta:.3f}"
                ),
                score=new_q_score,
            )

        validity = self._check_skill_validity(proposal.new_content)
        if not validity["valid"]:
            return GateDecision(
                accepted=False,
                reason=f"Invalid skill content: {validity['reason']}",
                score=new_q_score,
            )

        return GateDecision(
            accepted=True,
            reason=f"Accepted: score {new_q_score:.3f}, {validity['reason']}",
            score=new_q_score,
        )

    def _check_skill_validity(self, content: str) -> dict[str, object]:
        lowered = content.lower()
        checks = {
            "has_name": "name:" in lowered,
            "has_trigger": "trigger_conditions:" in lowered,
            "has_procedure": "## procedure" in lowered or "procedure:" in lowered,
            "not_empty": len(content.strip()) > 100,
            "has_failure_patterns": "failure" in lowered,
        }
        failed = [name for name, ok in checks.items() if not ok]
        return {
            "valid": not failed,
            "reason": "All checks passed" if not failed else f"Failed: {failed}",
        }
