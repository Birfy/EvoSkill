"""Generate candidate skill edits from offline failure signals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.harness.provider_auth import ensure_provider_api_key


ProposalType = Literal["refine", "create", "trigger_update"]


@dataclass
class SkillProposal:
    proposal_type: ProposalType
    skill_name: str
    new_content: str
    rationale: str
    prior_confidence: float


EVOLVER_PROMPT = """You are an expert at improving agent skill specifications based on failure analysis.

## Current Skill
{skill_content}

## Failure Analysis
{failure_summary}

## Task
Generate {k} diverse improvement proposals. Each proposal must be one of:
- "refine": improve the existing skill's procedure or instructions
- "create": create a new complementary skill to handle uncovered cases
- "trigger_update": refine trigger conditions to prevent misapplication

For each proposal, provide the complete new SKILL.md content, rationale, and confidence.
Respond ONLY as a JSON array:
[
  {{
    "type": "refine|create|trigger_update",
    "skill_name": "<name>",
    "new_content": "<complete SKILL.md>",
    "rationale": "<why>",
    "confidence": <float>
  }}
]
"""


class SkillEvolver:
    """LLM-backed proposal generator with a rejected-edit buffer."""

    def __init__(self, skill_library_path: str | Path, model: str = "claude-sonnet-4-6"):
        self.skill_lib = Path(skill_library_path)
        self.model = model
        self.rejected_buffer: list[dict[str, str]] = []

    def generate_proposals(
        self,
        skill_name: str,
        failure_summary: str,
        k: int = 6,
    ) -> list[SkillProposal]:
        skill_path = self.skill_lib / skill_name / "SKILL.md"
        skill_content = skill_path.read_text() if skill_path.exists() else ""
        if self.rejected_buffer:
            skill_content += "\n\n## Previously Rejected Directions\n"
            for rejected in self.rejected_buffer[-5:]:
                skill_content += f"- {rejected.get('rationale', '')}\n"

        prompt = EVOLVER_PROMPT.format(
            skill_content=skill_content or "No existing skill content.",
            failure_summary=failure_summary,
            k=k,
        )
        raw = self._call_anthropic(prompt, max_tokens=4096)
        try:
            data = json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []

        proposals: list[SkillProposal] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            proposal_type = item.get("type", "refine")
            if proposal_type not in {"refine", "create", "trigger_update"}:
                proposal_type = "refine"
            proposals.append(
                SkillProposal(
                    proposal_type=proposal_type,
                    skill_name=str(item.get("skill_name") or skill_name),
                    new_content=str(item.get("new_content") or ""),
                    rationale=str(item.get("rationale") or ""),
                    prior_confidence=_clamp_float(item.get("confidence"), default=0.5),
                )
            )
        return proposals

    def add_to_rejected_buffer(self, proposal: SkillProposal) -> None:
        self.rejected_buffer.append(
            {
                "skill_name": proposal.skill_name,
                "type": proposal.proposal_type,
                "rationale": proposal.rationale,
            }
        )
        if len(self.rejected_buffer) > 20:
            self.rejected_buffer = self.rejected_buffer[-20:]

    def _call_anthropic(self, prompt: str, max_tokens: int) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=ensure_provider_api_key("anthropic"))
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:])
        text = text.rsplit("```", 1)[0]
    return text.replace("```json", "").replace("```", "").strip()


def _clamp_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
