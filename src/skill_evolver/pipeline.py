"""Legacy standalone offline skill evolution pipeline.

Prefer ``scripts/run_loop.py`` for current offline evolution. That path uses
the shared run-loop program manager, PUCT tree search, trajectory handling,
and LLM-judge scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .evolver import SkillEvolver, SkillProposal
from .grouper import SkillGrouper
from .pareto_gate import ParetoGate
from .puct_search import PUCTSearch
from .scorer import LLMScorer


class SkillEvolutionPipeline:
    def __init__(
        self,
        skill_library_path: str | Path,
        trajectory_path: str | Path,
        n_rounds: int = 5,
        model: str = "claude-sonnet-4-6",
        *,
        min_score_delta: float = 0.05,
    ):
        self.skill_lib = Path(skill_library_path)
        self.grouper = SkillGrouper(trajectory_path)
        self.evolver = SkillEvolver(self.skill_lib, model)
        self.scorer = LLMScorer(model)
        self.gate = ParetoGate(min_score_delta=min_score_delta)
        self.search = PUCTSearch(self.evolver, self.scorer)
        self.n_rounds = n_rounds
        self.evolution_log: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        groups = self.grouper.group_by_skill()
        if not groups:
            return {"evolution_log": [], "rounds": self.n_rounds, "message": "no skill groups found"}

        for round_id in range(self.n_rounds):
            print(f"\n=== Evolution Round {round_id + 1}/{self.n_rounds} ===")
            for skill_name, group in groups.items():
                if not group["failures"]:
                    continue
                print(f"  Processing skill: {skill_name} ({len(group['failures'])} failures)")

                current_path = self.skill_lib / skill_name / "SKILL.md"
                if not current_path.exists():
                    print(f"    Skipping: {current_path} does not exist")
                    continue
                current_content = current_path.read_text()
                failure_summary = self.grouper.get_skill_signal_summary(skill_name)

                baseline_score = self.scorer.batch_score(
                    evolved_skill_content=current_content,
                    original_skill_content=current_content,
                    skill_name=skill_name,
                    trajectory_group=group,
                )
                self.gate.baseline_score = baseline_score

                top_proposals = self.search.search(
                    skill_name=skill_name,
                    current_skill_content=current_content,
                    failure_summary=failure_summary,
                    trajectory_group=group,
                )

                accepted = False
                for proposal in top_proposals:
                    new_score = self.scorer.batch_score(
                        evolved_skill_content=proposal.new_content,
                        original_skill_content=current_content,
                        skill_name=skill_name,
                        trajectory_group=group,
                    )
                    decision = self.gate.evaluate(proposal, new_score)
                    if decision.accepted:
                        self._commit_skill(skill_name, proposal, new_score, round_id)
                        accepted = True
                        print(
                            f"    Accepted: {proposal.proposal_type} "
                            f"(score: {baseline_score:.3f} -> {new_score:.3f})"
                        )
                        break
                    self.evolver.add_to_rejected_buffer(proposal)
                    print(f"    Rejected: {decision.reason}")

                if not accepted:
                    print(f"    No improvement found for {skill_name} in round {round_id + 1}")

            groups = self.grouper.group_by_skill()

        return {"evolution_log": self.evolution_log, "rounds": self.n_rounds}

    def _commit_skill(
        self,
        skill_name: str,
        proposal: SkillProposal,
        new_score: float,
        round_id: int,
    ) -> None:
        target_name = proposal.skill_name if proposal.proposal_type == "create" else skill_name
        skill_dir = self.skill_lib / target_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        current_path = skill_dir / "SKILL.md"
        if current_path.exists():
            backup_path = skill_dir / f"SKILL.v{round_id}.md"
            backup_path.write_text(current_path.read_text())

        current_path.write_text(proposal.new_content)
        self.evolution_log.append(
            {
                "round": round_id,
                "skill": target_name,
                "type": proposal.proposal_type,
                "score": new_score,
                "rationale": proposal.rationale,
            }
        )
