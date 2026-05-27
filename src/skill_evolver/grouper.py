"""Group offline trajectories by referenced skill."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class SkillGrouper:
    """SkillClaw-style G(s) grouping.

    G(s) = {trajectory in D | trajectory referenced skill s}.  The loader
    accepts both the rich trajectory JSONL described in plan.md and the
    StoredTrajectory JSONL produced by scripts/collect_trajectories.py.
    """

    def __init__(self, trajectory_path: str | Path):
        self.trajectory_path = Path(trajectory_path)
        self.trajectories = self._load(self.trajectory_path)

    def _load(self, path: Path) -> list[dict[str, Any]]:
        if path.is_dir():
            path = path / "trajectories.jsonl"
        trajectories: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(json.loads(line))
        return trajectories

    def group_by_skill(self) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "successes": [],
                "failures": [],
                "error_patterns": Counter(),
            }
        )

        for traj in self.trajectories:
            skills = self._skills_for(traj)
            success = self._is_success(traj)
            for skill_name in skills:
                if success:
                    groups[skill_name]["successes"].append(traj)
                    continue
                groups[skill_name]["failures"].append(traj)
                for err in self._errors_for(traj):
                    err_type = str(err).split(":", 1)[0].strip() or "UNKNOWN"
                    groups[skill_name]["error_patterns"][err_type] += 1

        return dict(groups)

    def get_skill_signal_summary(self, skill_name: str) -> str:
        groups = self.group_by_skill()
        if skill_name not in groups:
            return "No data available for this skill."

        group = groups[skill_name]
        n_success = len(group["successes"])
        n_fail = len(group["failures"])
        total = n_success + n_fail
        success_rate = (n_success / total * 100.0) if total else 0.0

        lines = [
            f"Skill: {skill_name}",
            f"Usage: {n_success} successes, {n_fail} failures",
            f"Success rate: {success_rate:.1f}%",
            "Top failure patterns:",
        ]
        top_errors = group["error_patterns"].most_common(3)
        if top_errors:
            lines.extend(f"  - {err_type}: {count} occurrences" for err_type, count in top_errors)
        else:
            lines.append("  - none")

        sample_lines = self._sample_failure_lines(group["failures"][:3])
        if sample_lines:
            lines.append("")
            lines.append("Sample failure steps:")
            lines.extend(sample_lines)
        return "\n".join(lines)

    def _skills_for(self, traj: dict[str, Any]) -> list[str]:
        skills = traj.get("skills_used") or []
        if skills:
            return [str(s) for s in skills]

        step_skills: list[str] = []
        for step in traj.get("steps") or []:
            step_skills.extend(str(s) for s in step.get("skills_active") or [])
        if step_skills:
            return sorted(set(step_skills))

        category = traj.get("category") or traj.get("domain") or "unassigned"
        return [str(category)]

    def _is_success(self, traj: dict[str, Any]) -> bool:
        if traj.get("final_success") is not None:
            return bool(traj["final_success"])
        if traj.get("trace_is_error"):
            return False
        answer = str(traj.get("agent_answer") or "")
        return answer != "[PARSE FAILED]"

    def _errors_for(self, traj: dict[str, Any]) -> list[str]:
        errors = [str(e) for e in traj.get("error_log") or [] if e]
        if errors:
            return errors
        parse_error = traj.get("trace_parse_error")
        if parse_error:
            return [f"PARSE_ERROR: {parse_error}"]
        if traj.get("trace_is_error"):
            return ["TRACE_ERROR: agent run failed"]
        return ["INCORRECT_ANSWER: final answer did not match ground truth"]

    def _sample_failure_lines(self, failures: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for traj in failures:
            failed_steps = [
                step for step in traj.get("steps") or []
                if step.get("success") is False
            ]
            if not failed_steps and traj.get("trace_result"):
                snippet = str(traj["trace_result"]).replace("\n", " ")[:240]
                lines.append(f"  > Trace result: {snippet}")
                continue
            for step in failed_steps[:2]:
                reason = step.get("failure_reason") or step.get("action") or "unknown failure"
                lines.append(f"  > {reason}")
        return lines
