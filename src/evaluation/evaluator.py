"""Three-layer SkillLearnBench-style evaluator."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.harness.provider_auth import ensure_provider_api_key


@dataclass
class SkillEvalResult:
    task_id: str
    domain: str
    skill_quality_score: float
    trajectory_alignment: float
    task_success: bool
    notes: str = ""


class SkillLearnBenchEvaluator:
    def __init__(self, benchmark_path: str | Path, model: str = "claude-sonnet-4-6"):
        self.bench_path = Path(benchmark_path)
        self.model = model

    def evaluate_skill_quality(self, skill_content: str, task: dict[str, Any]) -> float:
        domain = str(task.get("domain", "")).lower()
        checks = {
            "has_trigger": "trigger_conditions:" in skill_content,
            "has_procedure": "## Procedure" in skill_content or "## procedure" in skill_content.lower(),
            "has_examples": "## Examples" in skill_content or "## examples" in skill_content.lower(),
            "has_failure": "## Failure" in skill_content or "failure" in skill_content.lower(),
            "covers_domain": not domain or domain in skill_content.lower(),
            "reasonable_length": 200 < len(skill_content) < 6000,
        }
        return sum(1 for ok in checks.values() if ok) / len(checks)

    def evaluate_trajectory_alignment(self, trajectory: dict[str, Any], skill_content: str) -> float:
        steps = trajectory.get("steps") or []
        if not steps:
            return 0.0
        steps_summary = "\n".join(
            f"Step {step.get('step_id', i)}: {str(step.get('action', ''))[:160]}"
            for i, step in enumerate(steps, start=1)
        )
        prompt = f"""Rate how well the following trajectory follows the skill from 0.0 to 1.0.

Skill:
{skill_content[:1500]}

Trajectory:
{steps_summary}

Respond ONLY with JSON: {{"score": <float>, "reason": "<brief>"}}"""
        try:
            data = json.loads(_strip_json_fences(self._call_anthropic(prompt, max_tokens=256)))
            return _clamp_float(data.get("score"), default=0.0)
        except Exception:
            return 0.0

    def evaluate_task_outcome(self, task: dict[str, Any], trajectory: dict[str, Any]) -> bool:
        keypoints_path = self.bench_path / "eval_keypoints" / f"{task['id']}.json"
        if not keypoints_path.exists():
            return bool(trajectory.get("final_success", False))

        keypoints = json.loads(keypoints_path.read_text())
        trajectory_text = " ".join(
            str(step.get("action", "")) for step in trajectory.get("steps", [])
        ).lower()
        required = keypoints.get("required_actions", [])
        if not required:
            return bool(trajectory.get("final_success", False))
        matched = sum(1 for item in required if str(item).lower() in trajectory_text)
        return matched >= len(required) * 0.7

    def run_full_evaluation(
        self,
        skill_library_path: str | Path,
        trajectory_path: str | Path,
        condition_name: str = "ours",
    ) -> dict[str, Any]:
        trajectories = self._load_trajectories(trajectory_path)
        tasks = json.loads((self.bench_path / "tasks.json").read_text())
        skill_lib = Path(skill_library_path)

        results: list[SkillEvalResult] = []
        for task in tasks:
            task_id = str(task.get("id", ""))
            if task_id not in trajectories:
                continue
            traj = trajectories[task_id]
            skill_content = self._skill_content_for(skill_lib, traj)
            result = SkillEvalResult(
                task_id=task_id,
                domain=str(task.get("domain") or traj.get("domain") or "unknown"),
                skill_quality_score=self.evaluate_skill_quality(skill_content, task),
                trajectory_alignment=self.evaluate_trajectory_alignment(traj, skill_content),
                task_success=self.evaluate_task_outcome(task, traj),
            )
            results.append(result)

        n = len(results)
        if n == 0:
            return {"condition": condition_name, "n_tasks": 0, "results": []}
        return {
            "condition": condition_name,
            "n_tasks": n,
            "skill_quality": sum(r.skill_quality_score for r in results) / n,
            "trajectory_alignment": sum(r.trajectory_alignment for r in results) / n,
            "task_success_rate": sum(1 for r in results if r.task_success) / n,
            "by_domain": self._aggregate_by_domain(results),
            "results": [r.__dict__ for r in results],
        }

    def _load_trajectories(self, path: str | Path) -> dict[str, dict[str, Any]]:
        path = Path(path)
        trajectories: dict[str, dict[str, Any]] = {}
        with open(path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                traj = json.loads(line)
                task_id = str(traj.get("task_id") or traj.get("id") or idx)
                trajectories[task_id] = traj
        return trajectories

    def _skill_content_for(self, skill_lib: Path, trajectory: dict[str, Any]) -> str:
        parts: list[str] = []
        for skill_name in trajectory.get("skills_used") or []:
            skill_path = skill_lib / str(skill_name) / "SKILL.md"
            if skill_path.exists():
                parts.append(skill_path.read_text())
        return "\n\n".join(parts)

    def _aggregate_by_domain(self, results: list[SkillEvalResult]) -> dict[str, float]:
        grouped: dict[str, list[bool]] = defaultdict(list)
        for result in results:
            grouped[result.domain].append(result.task_success)
        return {
            domain: sum(1 for value in values if value) / len(values)
            for domain, values in grouped.items()
        }

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
