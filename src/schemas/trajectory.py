"""StoredTrajectory — serialisable record of a single agent run.

Trajectories are collected once by scripts/collect_trajectories.py and saved
as JSONL.  The evolution loop loads them from disk instead of re-running the
agent, which eliminates redundant inference during skill evolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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

    # Dataset fields
    question: str
    ground_truth: str
    category: str
    agent_answer: str

    # Subset of AgentTrace needed by the LLM judge (everything summarize() uses)
    trace_model: str = ""
    trace_num_turns: int = 0
    trace_duration_ms: int = 0
    trace_total_cost_usd: float = 0.0
    trace_is_error: bool = False
    trace_parse_error: str | None = None
    trace_result: str = ""          # full agent output text (used in summarize())
    trace_output: dict[str, Any] | None = None  # serialised AgentResponse

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
        """Return (AgentTrace, question, agent_answer, ground_truth, category)."""
        return (
            self.to_agent_trace(),
            self.question,
            self.agent_answer,
            self.ground_truth,
            self.category,
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
    ) -> "StoredTrajectory":
        """Build a StoredTrajectory from a live AgentTrace."""
        output_dict: dict[str, Any] | None = None
        if trace.output is not None:
            try:
                output_dict = trace.output.model_dump()
            except Exception:
                pass

        return cls(
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
