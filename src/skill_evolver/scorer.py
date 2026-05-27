"""Offline LLM scorer for candidate skill edits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.harness.provider_auth import ensure_provider_api_key


@dataclass
class ScoreResult:
    overall: float
    step_quality: float
    skill_alignment: float
    failure_avoidance: float
    reasoning: str


SCORER_PROMPT = """You are evaluating whether an evolved agent skill improves action quality.

## Original Skill
{original_skill}

## Evolved Skill (candidate)
{evolved_skill}

## Task Context
{task_description}

## Original Trajectory Step (failed)
Step action: {original_action}
Failure reason: {failure_reason}

## Re-generated Action with Evolved Skill
{new_action}

## Evaluation
Score the re-generated action on three dimensions from 0.0 to 1.0:
1. step_quality: Is the new action clear, specific, and executable?
2. skill_alignment: Does the new action follow the evolved skill's procedure?
3. failure_avoidance: Does the new action avoid the original failure pattern?

Respond ONLY in JSON:
{{
  "step_quality": <float>,
  "skill_alignment": <float>,
  "failure_avoidance": <float>,
  "reasoning": "<one sentence>"
}}
"""


class LLMScorer:
    """Score candidate skills without executing the task environment again."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def regenerate_action(
        self,
        task_description: str,
        evolved_skill_content: str,
        failed_step: dict[str, Any],
        context_steps: list[dict[str, Any]],
    ) -> str:
        context = "\n".join(
            f"Step {step.get('step_id', i)}: {step.get('action', '')}"
            for i, step in enumerate(context_steps[-5:], start=1)
        )
        prompt = f"""You are an agent solving a task. Based on the skill below and the context, generate the action you would take at the current step.

# Skill
{evolved_skill_content}

# Task
{task_description}

# Context (previous steps)
{context or "No prior successful steps recorded."}

# Current Step
The previous attempt at this step resulted in: {failed_step.get("failure_reason", "unknown error")}

Generate a single action to handle this step correctly. Respond with just the action, no explanation."""
        return self._call_anthropic(prompt, max_tokens=512).strip()

    def score_evolution(
        self,
        original_skill: str,
        evolved_skill: str,
        task_description: str,
        failed_step: dict[str, Any],
        new_action: str,
    ) -> ScoreResult:
        prompt = SCORER_PROMPT.format(
            original_skill=original_skill,
            evolved_skill=evolved_skill,
            task_description=task_description,
            original_action=failed_step.get("action", ""),
            failure_reason=failed_step.get("failure_reason", ""),
            new_action=new_action,
        )
        try:
            raw = _strip_json_fences(self._call_anthropic(prompt, max_tokens=512))
            data = json.loads(raw)
            step_quality = _clamp_float(data.get("step_quality"), default=0.0)
            skill_alignment = _clamp_float(data.get("skill_alignment"), default=0.0)
            failure_avoidance = _clamp_float(data.get("failure_avoidance"), default=0.0)
            return ScoreResult(
                overall=step_quality * 0.3 + skill_alignment * 0.4 + failure_avoidance * 0.3,
                step_quality=step_quality,
                skill_alignment=skill_alignment,
                failure_avoidance=failure_avoidance,
                reasoning=str(data.get("reasoning") or ""),
            )
        except Exception as exc:
            return ScoreResult(
                overall=0.0,
                step_quality=0.0,
                skill_alignment=0.0,
                failure_avoidance=0.0,
                reasoning=f"Scoring failed: {exc}",
            )

    def batch_score(
        self,
        evolved_skill_content: str,
        original_skill_content: str,
        skill_name: str,
        trajectory_group: dict[str, Any],
        top_k_failures: int = 5,
    ) -> float:
        del skill_name
        failures = trajectory_group.get("failures", [])
        if not failures:
            return 0.5

        scores: list[float] = []
        for traj in failures:
            if len(scores) >= top_k_failures:
                break
            failed_steps = self._failed_steps_for(traj)
            context_steps = [step for step in traj.get("steps", []) if step.get("success") is not False]
            for step in failed_steps[:2]:
                if len(scores) >= top_k_failures:
                    break
                new_action = self.regenerate_action(
                    task_description=self._task_description_for(traj),
                    evolved_skill_content=evolved_skill_content,
                    failed_step=step,
                    context_steps=context_steps,
                )
                score = self.score_evolution(
                    original_skill=original_skill_content,
                    evolved_skill=evolved_skill_content,
                    task_description=self._task_description_for(traj),
                    failed_step=step,
                    new_action=new_action,
                )
                scores.append(score.overall)
        return sum(scores) / len(scores) if scores else 0.0

    def _failed_steps_for(self, traj: dict[str, Any]) -> list[dict[str, Any]]:
        failed = [step for step in traj.get("steps", []) if step.get("success") is False]
        if failed:
            return failed
        return [
            {
                "step_id": 0,
                "action": traj.get("agent_answer") or traj.get("trace_result") or "",
                "failure_reason": "; ".join(traj.get("error_log") or [])
                or traj.get("trace_parse_error")
                or "final answer did not match ground truth",
            }
        ]

    def _task_description_for(self, traj: dict[str, Any]) -> str:
        return str(
            traj.get("task_description")
            or traj.get("question")
            or traj.get("task_id")
            or ""
        )

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
