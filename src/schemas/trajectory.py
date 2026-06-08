"""StoredTrajectory — serialisable record of a single agent run.

Trajectories are collected once by scripts/collect_trajectories.py and saved
as JSONL.  The evolution loop loads them from disk instead of re-running the
agent, which eliminates redundant inference during skill evolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
import re
import json

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.harness.agent import AgentTrace
    from src.schemas.agent import AgentResponse


class StoredTrajectory(BaseModel):
    """A single agent run serialised to disk."""

    # Optional plan-level trajectory fields. Existing collector output only
    # needs the dataset/trace fields below, but the offline skill evolver can
    # consume richer JSONL records when step-level data is available.
    task_id: str = ""
    task_description: str = ""
    domain: str = "unknown"
    skills_used: list[str] = []
    steps: list[dict[str, Any]] = []
    final_success: bool | None = None
    error_log: list[str] = []
    skill_calls: list[dict[str, Any]] = []

    # Dataset fields
    question: str
    ground_truth: str
    category: str
    agent_answer: str
    failure_type: str = ""
    failure_feedback: str = ""
    failure_analysis: dict[str, Any] | None = None

    # Step-level fault localization (populated by classify_failures.py)
    fault_step: int | None = None        # 0-based index of first fault message
    failure_behavior: str = ""           # observed failure at fault_step (π)
    fault_type: str = ""                 # "skill_wrong" | "skill_missing" | ""

    # Subset of AgentTrace needed by the LLM judge (everything summarize() uses)
    trace_model: str = ""
    trace_num_turns: int = 0
    trace_duration_ms: int = 0
    trace_total_cost_usd: float = 0.0
    trace_is_error: bool = False
    trace_parse_error: str | None = None
    trace_result: str = ""          # full agent output text (used in summarize())
    trace_output: dict[str, Any] | None = None  # serialised AgentResponse
    trace_messages: list[str] = []

    # ------------------------------------------------------------------ #

    def to_agent_trace(self) -> "AgentTrace[AgentResponse]":
        """Reconstruct a lightweight AgentTrace (no messages) for judge use."""
        from src.harness.agent import AgentTrace
        from src.schemas.agent import AgentResponse

        output = None
        if self.trace_output:
            try:
                output = AgentResponse(**self.trace_output)
            except Exception:
                pass

        return AgentTrace(
            uuid="",
            session_id="",
            model=self.trace_model,
            tools=[],
            duration_ms=self.trace_duration_ms,
            total_cost_usd=self.trace_total_cost_usd,
            num_turns=self.trace_num_turns,
            usage={},
            result=self.trace_result,
            is_error=self.trace_is_error,
            output=output,
            parse_error=self.trace_parse_error,
            raw_structured_output=self.trace_output,
            messages=[],
        )

    def to_extended_tuple(self):
        """Return (AgentTrace, question, agent_answer, ground_truth, category, failure_type, failure_feedback)."""
        feedback = self.failure_feedback
        if not feedback and self.failure_analysis:
            try:
                feedback = json.dumps(self.failure_analysis, ensure_ascii=False, default=str)
            except Exception:
                feedback = str(self.failure_analysis)
        return (
            self.to_agent_trace(),
            self.question,
            self.agent_answer,
            self.ground_truth,
            self.category,
            self.failure_type,
            feedback,
        )

    # ------------------------------------------------------------------ #

    @classmethod
    def from_trace(
        cls,
        trace: "AgentTrace[AgentResponse]",
        question: str,
        ground_truth: str,
        category: str,
        agent_answer: str,
        task_id: str = "",
        domain: str = "",
        skills_used: list[str] | None = None,
    ) -> "StoredTrajectory":
        """Build a StoredTrajectory from a live AgentTrace."""
        output_dict: dict[str, Any] | None = None
        if trace.output is not None:
            try:
                output_dict = trace.output.model_dump()
            except Exception:
                pass
        trace_messages = [_message_to_text(message) for message in trace.messages]
        skill_calls = _extract_skill_calls(trace_messages)
        detected_skills = sorted(
            {
                str(call.get("skill_name"))
                for call in skill_calls
                if call.get("skill_name")
            }
        )

        return cls(
            task_id=task_id,
            task_description=question,
            domain=domain or category,
            skills_used=skills_used if skills_used is not None else detected_skills,
            skill_calls=skill_calls,
            steps=[
                {
                    "step_id": i,
                    "state": "skill_call",
                    "skills_active": [call["skill_name"]] if call.get("skill_name") else [],
                    "action": call.get("text", ""),
                    "action_type": "skill",
                    "tool_name": "Skill",
                    "tool_input": call,
                    "observation": "",
                    "success": True,
                    "failure_reason": None,
                }
                for i, call in enumerate(skill_calls)
            ],
            question=question,
            ground_truth=ground_truth,
            category=category,
            agent_answer=agent_answer,
            trace_model=trace.model,
            trace_num_turns=trace.num_turns,
            trace_duration_ms=trace.duration_ms,
            trace_total_cost_usd=trace.total_cost_usd,
            trace_is_error=trace.is_error,
            trace_parse_error=trace.parse_error,
            trace_result=str(trace.result) if trace.result else "",
            trace_output=output_dict,
            trace_messages=trace_messages,
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def load_jsonl(path: str | Path) -> list["StoredTrajectory"]:
        """Load a JSONL file of StoredTrajectory records."""
        trajectories: list[StoredTrajectory] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(StoredTrajectory.model_validate_json(line))
        return trajectories

    @staticmethod
    def save_jsonl(trajectories: list["StoredTrajectory"], path: str | Path) -> None:
        """Save a list of StoredTrajectory records as JSONL."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for t in trajectories:
                f.write(t.model_dump_json() + "\n")


def _message_to_text(message: Any) -> str:
    """Convert an SDK message/event object into durable text."""
    if isinstance(message, str):
        return message
    for method_name in ("model_dump_json", "json"):
        method = getattr(message, method_name, None)
        if callable(method):
            try:
                return str(method())
            except Exception:
                pass
    for method_name in ("model_dump", "dict"):
        method = getattr(message, method_name, None)
        if callable(method):
            try:
                return str(method())
            except Exception:
                pass
    return str(message)


def _extract_skill_calls(messages: list[str]) -> list[dict[str, Any]]:
    """Best-effort extraction of explicit skill invocation records."""
    calls: list[dict[str, Any]] = []
    for idx, text in enumerate(messages):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if obj is not None:
            calls.extend(_extract_skill_calls_from_obj(obj, idx))
        else:
            # Text-based fallback only when JSON parsing fails to avoid double-counting.
            lowered = text.lower()
            if "skill(" in lowered or ("tool_name" in lowered and '"skill"' in lowered):
                snippet = text[:2000]
                calls.append(
                    {
                        "message_index": idx,
                        "event_type": "skill_tool_call",
                        "skill_name": _extract_skill_name(snippet),
                        "text": snippet,
                    }
                )
    return _dedupe_skill_calls(calls)


def _extract_skill_calls_from_obj(obj: Any, message_index: int) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if isinstance(obj, list):
        for item in obj:
            calls.extend(_extract_skill_calls_from_obj(item, message_index))
        return calls
    if not isinstance(obj, dict):
        return calls

    for skill in obj.get("activated_skills") or []:
        if isinstance(skill, dict):
            skill_name = skill.get("name") or skill.get("skill_name")
            payload = skill
        else:
            skill_name = str(skill)
            payload = {"name": skill_name}
        calls.append(
            {
                "message_index": message_index,
                "event_type": "activated_skill",
                "skill_name": str(skill_name) if skill_name else None,
                "payload": payload,
                "text": str(payload)[:2000],
            }
        )

    tool_name = str(obj.get("tool_name") or "").lower()
    tool_call = obj.get("tool_call") if isinstance(obj.get("tool_call"), dict) else {}
    tool_call_name = str(tool_call.get("name") or "").lower()
    if tool_name == "skill" or tool_call_name == "skill":
        payload = obj.get("action") or tool_call.get("arguments") or tool_call
        text = str(payload)[:2000]
        calls.append(
            {
                "message_index": message_index,
                "event_type": "skill_tool_call",
                "skill_name": _extract_skill_name(text),
                "payload": payload,
                "text": text,
            }
        )

    command = obj.get("command")
    if isinstance(command, str) and "SKILL.md" in command:
        skill_name = _extract_skill_name_from_path(command)
        if skill_name is None:
            skill_name = _extract_frontmatter_skill_name(str(obj.get("aggregated_output") or ""))
        if skill_name:
            calls.append(
                {
                    "message_index": message_index,
                    "event_type": "codex_skill_file_read",
                    "skill_name": skill_name,
                    "payload": {"command": command},
                    "text": command[:2000],
                }
            )

    for value in obj.values():
        if isinstance(value, (dict, list)):
            calls.extend(_extract_skill_calls_from_obj(value, message_index))
    return calls


def _extract_skill_name(text: str) -> str | None:
    patterns = [
        r"skill[_ -]?name['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_.-]+)",
        r"name['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_.-]+).*skill",
        r"Skill\(([^)]+)\)",
        r"Skill:\s*([A-Za-z0-9_.-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip("'\" ")
    return None


def _extract_skill_name_from_path(text: str) -> str | None:
    """Extract a skill name from a command or path that references SKILL.md."""
    pattern = r"\.(?:claude|agents|codex)/skills/(?:\.system/)?([A-Za-z0-9_.-]+)/SKILL\.md"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def _extract_frontmatter_skill_name(text: str) -> str | None:
    """Extract a YAML-frontmatter skill name from a SKILL.md file body."""
    match = re.search(r"(?m)^name:\s*['\"]?([A-Za-z0-9_.-]+)['\"]?\s*$", text)
    if match:
        return match.group(1)
    return None


def _dedupe_skill_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first copy of each extracted skill event."""
    seen: set[tuple[Any, Any, Any]] = set()
    deduped: list[dict[str, Any]] = []
    for call in calls:
        key = (
            call.get("message_index"),
            call.get("event_type"),
            call.get("skill_name"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return deduped
