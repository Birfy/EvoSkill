#!/usr/bin/env python3
"""Create compact trajectory JSONL with scoring feedback.

The raw StoredTrajectory file keeps full SDK messages and can be very large.
This script preserves enough signal for offline skill evolution:
task metadata, answer comparison, score/pass/failure feedback, cost, a compact
action trace, and exact skill activation/tool-call records when present.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.loop.runner import _score_multi_tolerance
from src.schemas.trajectory import _extract_skill_calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw StoredTrajectory JSONL")
    parser.add_argument("--output", required=True, help="Compact JSONL output")
    parser.add_argument("--summary", default=None, help="Optional summary JSON output")
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--max-text", type=int, default=1200)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compact_rows: list[dict[str, Any]] = []
    with input_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            compact_rows.append(compact_record(raw, args.max_actions, args.max_text))

    with output_path.open("w") as f:
        for row in compact_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(compact_rows)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def compact_record(raw: dict[str, Any], max_actions: int, max_text: int) -> dict[str, Any]:
    question = str(raw.get("question") or raw.get("task_description") or "")
    predicted = str(raw.get("agent_answer") or "")
    expected = str(raw.get("ground_truth") or "")
    score = _score_multi_tolerance(question, predicted.strip().lower(), expected.strip().lower())
    passed = score >= 0.8
    failure_reason = None if passed else classify_failure(raw, predicted, expected, score)

    trace_messages = [str(m) for m in raw.get("trace_messages") or []]
    skill_calls = _extract_skill_calls(trace_messages)
    skills_used = sorted(
        {
            str(call.get("skill_name"))
            for call in skill_calls
            if call.get("skill_name")
        }
    )
    actions = extract_actions(trace_messages, max_actions=max_actions, max_text=max_text)

    return {
        "task_id": raw.get("task_id"),
        "category": raw.get("category") or raw.get("domain"),
        "question": question,
        "expected_answer": expected,
        "predicted_answer": predicted,
        "eval_score": score,
        "passed": passed,
        "failure_reason": failure_reason,
        "failure_feedback": build_failure_feedback(predicted, expected, score, failure_reason),
        "trace_model": raw.get("trace_model"),
        "trace_num_turns": raw.get("trace_num_turns"),
        "trace_duration_ms": raw.get("trace_duration_ms"),
        "trace_total_cost_usd": raw.get("trace_total_cost_usd"),
        "trace_is_error": raw.get("trace_is_error"),
        "trace_parse_error": raw.get("trace_parse_error"),
        "skills_used": skills_used,
        "skill_calls": skill_calls,
        "actions": actions,
        "trace_summary": {
            "num_raw_messages": len(trace_messages),
            "num_actions": len(actions),
            "final_result": truncate(str(raw.get("trace_result") or ""), max_text),
        },
    }


def classify_failure(raw: dict[str, Any], predicted: str, expected: str, score: float) -> str:
    if raw.get("trace_parse_error"):
        return "PARSE_ERROR"
    if raw.get("trace_is_error"):
        return "TRACE_ERROR"
    if not predicted or predicted == "[PARSE FAILED]":
        return "NO_ANSWER"
    pred_nums = extract_numbers(predicted)
    exp_nums = extract_numbers(expected)
    if pred_nums and exp_nums:
        return "NUMERIC_MISMATCH_CLOSE" if score > 0 else "NUMERIC_MISMATCH"
    return "TEXT_MISMATCH"


def build_failure_feedback(
    predicted: str,
    expected: str,
    score: float,
    failure_reason: str | None,
) -> str | None:
    if failure_reason is None:
        return None
    return (
        f"{failure_reason}: predicted {predicted!r}, expected {expected!r}, "
        f"score={score:.3f}."
    )


def extract_actions(messages: list[str], *, max_actions: int, max_text: int) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for idx, text in enumerate(messages):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "tool_name" not in obj and "action" not in obj and "observation" not in obj:
            continue
        action = obj.get("action")
        observation = obj.get("observation")
        summary = obj.get("summary")
        thought = obj.get("thought")
        actions.append(
            {
                "message_index": idx,
                "source": obj.get("source"),
                "kind": obj.get("kind"),
                "tool_name": obj.get("tool_name"),
                "summary": truncate(_textify(summary), max_text // 2),
                "thought": truncate(_textify(thought), max_text),
                "action": truncate(_textify(action), max_text),
                "observation": truncate(_textify(observation), max_text),
            }
        )
        if len(actions) >= max_actions:
            break
    return actions


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    scores = [float(row["eval_score"]) for row in rows]
    failures = Counter(row["failure_reason"] for row in rows if not row["passed"])
    categories = Counter(row.get("category") or "unknown" for row in rows)
    return {
        "n": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": passed / n if n else 0.0,
        "avg_score": sum(scores) / n if n else 0.0,
        "failure_reasons": dict(failures),
        "categories": dict(categories),
        "cost_total_usd": sum(float(row.get("trace_total_cost_usd") or 0) for row in rows),
    }


def extract_numbers(text: str) -> list[float]:
    nums: list[float] = []
    normalized = text.replace(",", "").replace("−", "-")
    for match in re.finditer(r"-?\d+(?:\.\d+)?", normalized):
        try:
            nums.append(float(match.group(0)))
        except ValueError:
            pass
    return nums


def _textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 40
    return f"{text[:head]} ... <truncated {len(text) - head - tail} chars> ... {text[-tail:]}"


if __name__ == "__main__":
    main()
