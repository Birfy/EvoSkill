#!/usr/bin/env python3
"""Classify failure trajectories by root cause type.

Reads a trajectories.jsonl file, calls a cheap LLM for each failed trajectory
(score < 0.8) that lacks a failure_type label, writes a short snake_case label
back into the field, and saves the updated JSONL in-place. Optionally rewrites
the trajectory category so successes become ``success`` and failures are grouped
by their inferred failure_type.

Usage:
    uv run python scripts/classify_failures.py \\
        --trajectories <dir_or_file> \\
        [--model claude-haiku-4-5-20251001] \\
        [--provider anthropic] \\
        [--score-threshold 0.8] \\
        [--concurrency 8] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.trajectory import StoredTrajectory
from src.loop.runner import _score_multi_tolerance  # type: ignore[attr-defined]

CLASSIFY_PROMPT = """\
You are analyzing why an AI agent failed to answer a question correctly.
Identify: (1) the root-cause failure type, (2) the FIRST message index where
things went wrong, (3) what the agent did wrong at that step, and (4) whether
an active skill misguided the agent or no skill covered the scenario.

## Question
{question}

## Expected Answer
{ground_truth}

## Agent's Wrong Answer
{agent_answer}

## Agent Trace ({num_steps} messages, excerpt)
{trace_summary}

## Root-Cause Label
Pick exactly one failure_type from this batch-specific taxonomy:
{taxonomy}

Use one of those labels exactly. Put the specific mechanism in failure_behavior,
not in failure_type. Do not use benchmark difficulty, dataset split, task id,
file name, answer value, or domain-specific constants as labels.

Reply with ONLY a JSON object — no markdown fences:
{{
  "failure_type": "<snake_case_label>",
  "brief_reason": "<one sentence>",
  "fault_step": <0-based message index of first fault, or null if unclear>,
  "failure_behavior": "<what the agent did wrong at that step, one sentence>",
  "fault_type": "<skill_wrong or skill_missing>"
}}

fault_type rules:
- "skill_wrong"   — an active skill gave incorrect or misleading guidance
- "skill_missing" — no skill covered this scenario; agent had no relevant guidance
"""

CATEGORY_PROMPT = """\
You are grouping failed agent trajectories into broad reusable root-cause
categories for later skill evolution.

Read all failure summaries below. Propose 6 to 14 fine-grained failure_type labels that
cover this batch. Labels must be generic and reusable across datasets, not tied
to any benchmark, file, table, date, answer value, or domain constant.
Prefer splitting broad categories (e.g. transformation_error) into more specific
sub-types when the data supports it (e.g. formula_selection_error, unit_conversion_error,
sign_error, magnitude_error, rounding_error, integration_error).

Good label style:
- input_selection_error: wrong source/evidence/context/item/field/entity/period/scope
- information_extraction_error: right context but wrong needed value/fact copied or bound
- transformation_error: wrong computation/comparison/conversion/derivation/algorithm
- representation_error: right-ish content but wrong unit/scale/order/format/rounding/schema
- perception_error: visual/layout/chart/spatial structure misread
- execution_error: skipped steps, stopped early, tool/process failed
- grounding_error: unsupported or invented data/facts

Return ONLY JSON:
{{
  "categories": [
    {{"failure_type": "<snake_case>", "definition": "<short generic definition>"}}
  ]
}}

## Failure Summaries
{summaries}
"""


def _format_trace(traj: StoredTrajectory, max_chars: int = 6000) -> tuple[str, int]:
    """Format trace as numbered messages for step-level fault localization."""
    if traj.trace_messages:
        # The first messages are often huge SDK/tool init blocks. Keep a small
        # head plus a larger tail so the classifier sees both setup and the final
        # divergent answer path.
        indexed = list(enumerate(traj.trace_messages))
        head = indexed[:4]
        tail = indexed[-24:] if len(indexed) > 28 else indexed[4:]
        selected = head + [(-1, "... omitted middle messages ...")] + tail if len(indexed) > 28 else indexed
        lines = []
        for i, msg in selected:
            if i >= 0 and "SystemMessage(subtype='init'" in msg:
                excerpt = "System init/tool list omitted"
            else:
                excerpt = str(msg)[:420].replace("\n", " ")
            prefix = "[...]" if i < 0 else f"[{i}]"
            lines.append(f"{prefix} {excerpt}")
        return "\n".join(lines)[:max_chars], len(traj.trace_messages)
    if traj.trace_result:
        return traj.trace_result[:max_chars], 0
    return "", 0


def _is_placeholder_failure_type(value: str | None) -> bool:
    label = str(value or "").strip().lower()
    return label in {"", "unknown", "unclassified", "unclassified_failure"}


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", str(value or "").strip().lower()).strip("_")


def _failure_brief(idx: int, traj: StoredTrajectory, max_chars: int = 900) -> str:
    trace_summary, num_steps = _format_trace(traj, max_chars=1200)
    text = (
        f"### Failure idx={idx} task_id={traj.task_id}\n"
        f"Question: {traj.question[:360]}\n"
        f"Expected: {traj.ground_truth[:160]}\n"
        f"Agent answer: {traj.agent_answer[:160]}\n"
        f"Trace messages: {num_steps}\n"
        f"Trace excerpt:\n{trace_summary[:max_chars]}\n"
    )
    return text


def _taxonomy_text(categories: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- {item['failure_type']} — {item.get('definition', '').strip()}"
        for item in categories
    )


async def derive_failure_taxonomy(
    failures: list[tuple[int, StoredTrajectory]],
    provider: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
) -> list[dict[str, str]]:
    summaries = "\n".join(_failure_brief(idx, traj, max_chars=700) for idx, traj in failures)
    prompt = CATEGORY_PROMPT.format(summaries=summaries[:18000])
    raw = await asyncio.wait_for(
        _call_api(prompt, provider, model, api_key),
        timeout=timeout_seconds,
    )
    try:
        obj = json.loads(_strip_json_fences(raw))
    except Exception as exc:
        raise ValueError(f"Could not parse taxonomy JSON: {exc}; raw={raw[:1000]!r}") from exc
    items = obj.get("categories") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"Taxonomy JSON missing categories; raw={raw[:1000]!r}")
    categories: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        label = _normalize_label(str(item.get("failure_type") or ""))
        definition = str(item.get("definition") or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        categories.append({"failure_type": label, "definition": definition})
        if len(categories) >= 8:
            break
    if len(categories) < 2:
        raise ValueError(f"Taxonomy too small; raw={raw[:1000]!r}")
    return categories


async def classify_one(
    traj: StoredTrajectory,
    semaphore: asyncio.Semaphore,
    provider: str,
    model: str,
    api_key: str,
    taxonomy: list[dict[str, str]],
    idx: int | None = None,
) -> dict:
    """Return classification dict with failure_type, fault_step, failure_behavior, fault_type."""
    trace_summary, num_steps = _format_trace(traj)
    prompt = CLASSIFY_PROMPT.format(
        question=traj.question[:1000],
        ground_truth=traj.ground_truth[:500],
        agent_answer=traj.agent_answer[:500],
        num_steps=num_steps or "?",
        trace_summary=trace_summary,
        taxonomy=_taxonomy_text(taxonomy),
    )
    async with semaphore:
        raw = await _call_api(prompt, provider, model, api_key)

    try:
        cleaned = _strip_json_fences(raw)
        obj = json.loads(cleaned)
    except Exception:
        obj = {}
        m = re.search(r'"failure_type"\s*:\s*"([^"]+)"', raw)
        if m:
            obj["failure_type"] = m.group(1)
    if not obj.get("failure_type"):
        raise ValueError(f"Classifier returned no failure_type; raw={raw[:500]!r}")

    ft = _normalize_label(str(obj.get("failure_type", ""))) or "unknown"
    allowed = {item["failure_type"] for item in taxonomy}
    aliases = {
        "wrong_artifact_selection": "input_selection_error",
        "wrong_row_column_binding": "information_extraction_error",
        "wrong_unit_or_scale": "representation_error",
        "aggregation_scope_error": "input_selection_error",
        "formula_definition_error": "transformation_error",
        "chart_feature_misread": "perception_error",
        "output_order_mismatch": "representation_error",
        "precision_rounding_error": "representation_error",
        "grounding": "grounding_error",
        "data_retrieval": "input_selection_error",
        "computation": "transformation_error",
        "concept_definition": "transformation_error",
        "precision_rounding": "representation_error",
        "numeric_extraction_error": "information_extraction_error",
        "calculation_error": "transformation_error",
        "source_selection_error": "input_selection_error",
        "chart_interpretation_error": "perception_error",
        "output_format_error": "representation_error",
        "incomplete_execution": "execution_error",
        "unsupported_grounding": "grounding_error",
        "data_extraction_error": "information_extraction_error",
        "value_extraction_error": "information_extraction_error",
        "incorrect_value_extraction": "information_extraction_error",
        "wrong_value_extraction": "information_extraction_error",
        "table_extraction_error": "information_extraction_error",
        "field_binding_error": "information_extraction_error",
        "period_binding_error": "input_selection_error",
        "scope_selection_error": "input_selection_error",
        "wrong_source": "input_selection_error",
        "wrong_context": "input_selection_error",
        "arithmetic_error": "transformation_error",
        "formula_error": "transformation_error",
        "computation_error": "transformation_error",
        "derivation_error": "transformation_error",
        "statistical_calculation_error": "transformation_error",
        "unit_error": "representation_error",
        "scale_error": "representation_error",
        "format_error": "representation_error",
        "rounding_error": "representation_error",
        "ordering_error": "representation_error",
    }
    raw_ft = ft
    if ft not in allowed:
        ft = aliases.get(ft, ft)
    if ft not in allowed:
        raise ValueError(
            f"Classifier chose label outside taxonomy: raw={raw_ft!r}; "
            f"allowed={sorted(allowed)}"
        )

    fault_step_raw = obj.get("fault_step")
    fault_step = int(fault_step_raw) if isinstance(fault_step_raw, (int, float)) else None

    failure_behavior = str(obj.get("failure_behavior", "")).strip()

    fault_type_raw = str(obj.get("fault_type", "")).strip().lower()
    fault_type = fault_type_raw if fault_type_raw in ("skill_wrong", "skill_missing") else ""

    return {
        "failure_type": ft,
        "raw_failure_type": raw_ft,
        "fault_step": fault_step,
        "failure_behavior": failure_behavior,
        "fault_type": fault_type,
    }


async def _call_api(prompt: str, provider: str, model: str, api_key: str) -> str:
    if provider == "anthropic":
        import anthropic
        client_kwargs = {"api_key": api_key}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        client = anthropic.AsyncAnthropic(**client_kwargs)
        kwargs = {
            "model": model,
            "max_tokens": _anthropic_max_tokens(model, 2048),
            "system": (
                "Return exactly one valid JSON object. Do not output markdown, "
                "prose, or a hidden-only/thinking-only answer."
            ),
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = await client.messages.create(**kwargs)
        text = _extract_anthropic_message_text(resp)
        if text:
            return text
        if _anthropic_message_has_only_thinking(resp):
            retry_prompt = (
                "Return ONLY this JSON schema with one allowed failure_type. "
                "No markdown, no explanation outside JSON.\n"
                '{"failure_type":"transformation_error","brief_reason":"...",'
                '"fault_step":null,"failure_behavior":"...",'
                '"fault_type":"skill_missing"}\n\n'
                "Original request tail:\n"
                + prompt[-4000:]
            )
            retry_kwargs = dict(kwargs)
            retry_kwargs["max_tokens"] = _anthropic_max_tokens(model, 4096)
            retry_kwargs["messages"] = [{"role": "user", "content": retry_prompt}]
            retry_resp = await client.messages.create(**retry_kwargs)
            retry_text = _extract_anthropic_message_text(retry_resp)
            if retry_text:
                return retry_text
            raise RuntimeError(
                "Anthropic classifier returned thinking-only twice: "
                f"first={_summarize_anthropic_message_blocks(resp)} "
                f"retry={_summarize_anthropic_message_blocks(retry_resp)}"
            )
        raise RuntimeError(
            "Anthropic classifier returned no text content: "
            f"{_summarize_anthropic_message_blocks(resp)}"
        )
    if provider == "openai":
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
    raise ValueError(f"Unsupported provider: {provider!r}")


def _strip_json_fences(text: str) -> str:
    value = re.sub(r"```(?:json)?\s*|\s*```", "", str(text or "")).strip()
    if value.startswith("{") and value.endswith("}"):
        return value
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        return value[start : end + 1]
    return value


def _anthropic_max_tokens(model: str, configured: int) -> int:
    if model.lower().startswith("deepseek"):
        return max(configured, 4096)
    return configured


def _extract_anthropic_message_text(response: object) -> str:
    """Extract text from Anthropic-compatible content blocks.

    DeepSeek's Anthropic-compatible API may emit thinking blocks before the final
    text block. Thinking blocks do not expose ``.text``, so indexing
    ``content[0].text`` fails even when a valid JSON text block follows.
    """
    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        block_type = str(getattr(block, "type", "") or "").lower()
        if block_type in {"thinking", "redacted_thinking"}:
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text)
            continue
        payload = None
        for method_name in ("model_dump", "to_dict", "dict"):
            method = getattr(block, method_name, None)
            if callable(method):
                try:
                    payload = method()
                    break
                except Exception:
                    pass
        if isinstance(payload, dict):
            if str(payload.get("type") or "").lower() in {"thinking", "redacted_thinking"}:
                continue
            value = payload.get("text") or payload.get("content")
            if isinstance(value, str) and value.strip():
                parts.append(value)
                continue
        if isinstance(block, dict):
            if str(block.get("type") or "").lower() in {"thinking", "redacted_thinking"}:
                continue
            value = block.get("text") or block.get("content")
            if isinstance(value, str) and value.strip():
                parts.append(value)
    return "\n".join(parts).strip()


def _summarize_anthropic_message_blocks(response: object) -> str:
    blocks = getattr(response, "content", None) or []
    block_types = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if not block_type:
            block_type = type(block).__name__
        block_types.append(str(block_type))
    return (
        f"stop_reason={getattr(response, 'stop_reason', None)!r}, "
        f"block_types={block_types!r}, "
        f"usage={getattr(response, 'usage', None)!r}"
    )


def _anthropic_message_has_only_thinking(response: object) -> bool:
    blocks = list(getattr(response, "content", None) or [])
    if not blocks:
        return False
    seen_thinking = False
    for block in blocks:
        block_type = str(getattr(block, "type", "") or type(block).__name__).lower()
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            return False
        if "thinking" in block_type:
            seen_thinking = True
            continue
        payload = None
        for method_name in ("model_dump", "to_dict", "dict"):
            method = getattr(block, method_name, None)
            if callable(method):
                try:
                    payload = method()
                    break
                except Exception:
                    pass
        if isinstance(payload, dict):
            payload_type = str(payload.get("type") or "").lower()
            payload_text = payload.get("text") or payload.get("content")
            if isinstance(payload_text, str) and payload_text.strip():
                return False
            if "thinking" in payload_type:
                seen_thinking = True
                continue
        if isinstance(block, dict):
            payload_type = str(block.get("type") or "").lower()
            payload_text = block.get("text") or block.get("content")
            if isinstance(payload_text, str) and payload_text.strip():
                return False
            if "thinking" in payload_type:
                seen_thinking = True
                continue
        return False
    return seen_thinking



async def run(args: argparse.Namespace) -> None:
    # Resolve JSONL path
    traj_path = Path(args.trajectories)
    if traj_path.is_dir():
        traj_path = traj_path / "trajectories.jsonl"
    if not traj_path.exists():
        print(f"ERROR: trajectories file not found: {traj_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading trajectories from {traj_path}...")
    trajectories = StoredTrajectory.load_jsonl(traj_path)
    print(f"Loaded {len(trajectories)} trajectories")

    # Determine which need classification
    to_classify: list[tuple[int, StoredTrajectory]] = []
    scores: list[float] = []
    for idx, t in enumerate(trajectories):
        score = _score_multi_tolerance(t.question, t.agent_answer.strip().lower(), t.ground_truth.strip().lower())
        scores.append(score)
        if score < args.score_threshold and _is_placeholder_failure_type(t.failure_type):
            to_classify.append((idx, t))

    already_done = sum(
        1
        for t in trajectories
        if t.failure_type and not _is_placeholder_failure_type(t.failure_type)
    )
    print(
        f"Failures below threshold {args.score_threshold}: "
        f"{len(to_classify)} to classify, {already_done} already labelled"
    )

    if not to_classify and not args.rewrite_categories:
        print("Nothing to do.")
        return

    if args.dry_run:
        print(f"[dry-run] Would classify {len(to_classify)} trajectories. Exiting.")
        return

    updated = 0
    errors = 0
    if to_classify:
        # Resolve API key only when classification is actually needed.
        from src.harness.provider_auth import ensure_provider_api_key
        api_key = ensure_provider_api_key(args.provider)

        print(
            f"Deriving batch taxonomy from {len(to_classify)} failures with "
            f"{args.provider}/{args.model}...",
            flush=True,
        )
        taxonomy = await derive_failure_taxonomy(
            to_classify,
            args.provider,
            args.model,
            api_key,
            timeout_seconds=args.request_timeout_seconds,
        )
        print("Derived taxonomy:", flush=True)
        for item in taxonomy:
            print(f"  - {item['failure_type']}: {item.get('definition', '')}", flush=True)

        semaphore = asyncio.Semaphore(args.concurrency)
        queue: asyncio.Queue[tuple[int, StoredTrajectory] | None] = asyncio.Queue()
        for item in to_classify:
            queue.put_nowait(item)
        worker_count = max(1, min(args.concurrency, len(to_classify)))
        for _ in range(worker_count):
            queue.put_nowait(None)

        lock = asyncio.Lock()

        async def worker(worker_id: int) -> None:
            nonlocal updated, errors
            while True:
                item = await queue.get()
                if item is None:
                    return
                idx, traj = item
                print(
                    f"  [CLASSIFY request] worker={worker_id} idx={idx} task_id={traj.task_id}",
                    flush=True,
                )
                try:
                    result = await asyncio.wait_for(
                        classify_one(
                            traj,
                            semaphore,
                            args.provider,
                            args.model,
                            api_key,
                            taxonomy,
                            idx=idx,
                        ),
                        timeout=args.request_timeout_seconds,
                    )
                except Exception as exc:
                    print(
                        "  [WARN] classification task failed: "
                        f"idx={idx} task_id={traj.task_id} "
                        f"{type(exc).__name__}: {exc!r}",
                        flush=True,
                    )
                    async with lock:
                        errors += 1
                    continue
                print(
                    "  [CLASSIFY done] "
                    f"idx={idx} task_id={traj.task_id} "
                    f"raw={result.get('raw_failure_type')} "
                    f"failure_type={result.get('failure_type')}",
                    flush=True,
                )
                update = {k: v for k, v in result.items() if v is not None and v != ""}
                async with lock:
                    trajectories[idx] = trajectories[idx].model_copy(update=update)
                    updated += 1

        print(f"Classifying with {args.provider}/{args.model} (workers={worker_count})...")
        await asyncio.gather(*(worker(i + 1) for i in range(worker_count)))

    print(f"Classified {updated} trajectories ({errors} errors)")

    # Summary of label distribution
    from collections import Counter
    label_counts = Counter(t.failure_type for t in trajectories if t.failure_type)
    print("Label distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label:40s} {count}")

    if args.rewrite_categories:
        rewritten: list[StoredTrajectory] = []
        category_counts: Counter[str] = Counter()
        for traj, score in zip(trajectories, scores):
            if score >= args.score_threshold:
                category = "success"
                update = {
                    "category": category,
                    "domain": category,
                    "failure_type": "",
                }
            else:
                failure_type = (traj.failure_type or "unclassified_failure").strip() or "unclassified_failure"
                category = failure_type
                update = {
                    "category": category,
                    "domain": category,
                    "failure_type": failure_type,
                }
            category_counts[category] += 1
            rewritten.append(traj.model_copy(update=update))
        trajectories = rewritten
        print("Rewritten trajectory categories:")
        for label, count in category_counts.most_common():
            print(f"  {label:40s} {count}")

    # Save back
    print(f"Saving to {traj_path}...")
    StoredTrajectory.save_jsonl(trajectories, traj_path)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", required=True, help="Path to trajectories.jsonl or its parent directory")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="LLM model for classification")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"], help="API provider")
    parser.add_argument("--score-threshold", type=float, default=0.8, help="Trajectories scoring below this are classified")
    parser.add_argument("--concurrency", type=int, default=8, help="Max parallel API calls")
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=180.0,
        help="Timeout for one classification request",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without calling the API")
    parser.add_argument(
        "--rewrite-categories",
        action="store_true",
        help="Set successful trajectories to category=success and failed trajectories to category=failure_type",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
