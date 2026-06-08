"""Claude SDK execution and response parsing.

This module handles all Claude-specific logic:
    - Converting dict options to ClaudeAgentOptions (fallback)
    - Spawning ClaudeSDKClient and streaming messages
    - Parsing SystemMessage/ResultMessage into AgentTrace fields
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Type

from pydantic import BaseModel, ValidationError


def _claude_debug(message: str) -> None:
    print(f"[CLAUDE-DEBUG] {message}", flush=True)


def _get_option_model(options: Any) -> str:
    model = getattr(options, "model", None)
    if isinstance(model, str):
        return model
    if isinstance(options, dict):
        raw = options.get("model") or options.get("model_id") or ""
        return raw if isinstance(raw, str) else ""
    return ""


def _prepare_claude_provider_env(options: Any) -> None:
    """Normalize Claude Code env for Anthropic-compatible DeepSeek models.

    Claude Code reads Anthropic-style env vars. DeepSeek's Claude-Code
    integration uses the Anthropic-compatible route, so a DeepSeek key must be
    exposed as ``ANTHROPIC_API_KEY`` and the base URL must point at DeepSeek.
    Without this, Claude Code falls back to Anthropic/OAuth and can spin in a
    structured-output hook loop after repeated 401s.
    """
    model = _get_option_model(options)
    if not model.lower().startswith("deepseek"):
        return

    os.environ.setdefault("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        for name in ("ANTHROPIC_AUTH_TOKEN", "DEEPSEEK_API_KEY"):
            value = os.environ.get(name)
            if value:
                os.environ["ANTHROPIC_API_KEY"] = value
                break
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Claude Code DeepSeek model requires ANTHROPIC_API_KEY, "
            "ANTHROPIC_AUTH_TOKEN, or DEEPSEEK_API_KEY. Set one before running."
        )
    _claude_debug(
        "provider env ready "
        f"model={model} base_url={os.environ.get('ANTHROPIC_BASE_URL')} "
        f"api_key_set={bool(os.environ.get('ANTHROPIC_API_KEY'))}"
    )


async def execute_query(options: Any, query: str) -> list[Any]:
    """Execute a query via Claude SDK.

    Args:
        options: ClaudeAgentOptions or dict (auto-converted)
        query: The question to send to the agent

    Returns:
        List of messages: [SystemMessage, ..., ResultMessage]
    """
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    # If someone passed a dict to Claude SDK (e.g., from config_to_options),
    # convert it to ClaudeAgentOptions. This is a fallback — normally profiles
    # use build_claudecode_options() which returns the right type directly.
    if isinstance(options, dict):
        claude_opts = ClaudeAgentOptions(
            system_prompt=options.get("system"),
            allowed_tools=list(options.get("tools", {}).keys())
            if options.get("tools")
            else [],
            output_format=options.get("format"),
            setting_sources=["user", "project"],
            permission_mode="acceptEdits",
        )
        if "model_id" in options and "claude" in options["model_id"].lower():
            claude_opts.model = options["model_id"]
        options = claude_opts

    _prepare_claude_provider_env(options)
    model = _get_option_model(options)
    tools = getattr(options, "allowed_tools", None) or []
    cwd = getattr(options, "cwd", None) or getattr(options, "working_directory", None) or ""
    project_dir = getattr(options, "permission_prompt_tool_name", None)
    _claude_debug(
        "execute_query start "
        f"model={model or '<default>'} query_chars={len(query)} "
        f"tools={len(tools)} cwd={cwd or '<sdk-default>'} "
        f"permission_tool={project_dir or '<none>'}"
    )

    # ClaudeSDKClient spawns a Claude Code process, sends the query,
    # and streams back messages (SystemMessage, AssistantMessages, ResultMessage)
    started = time.monotonic()
    async with ClaudeSDKClient(options) as client:
        _claude_debug(
            f"client entered model={model or '<default>'} "
            f"elapsed={time.monotonic() - started:.1f}s"
        )
        await client.query(query)
        _claude_debug(
            f"query submitted model={model or '<default>'} "
            f"elapsed={time.monotonic() - started:.1f}s"
        )
        messages = []
        async for msg in client.receive_response():
            messages.append(msg)
            msg_type = type(msg).__name__
            subtype = getattr(msg, "subtype", "")
            is_error = getattr(msg, "is_error", None)
            _claude_debug(
                f"received message #{len(messages)} type={msg_type} "
                f"subtype={subtype or '<none>'} is_error={is_error} "
                f"elapsed={time.monotonic() - started:.1f}s"
            )
        _claude_debug(
            f"receive complete messages={len(messages)} "
            f"elapsed={time.monotonic() - started:.1f}s"
        )
        return messages


def parse_response(
    messages: list[Any],
    response_model: Type[BaseModel],
) -> dict[str, Any]:
    """Parse Claude SDK messages into AgentTrace field values.

    Claude returns: [SystemMessage, ..., ResultMessage]
        - First message has identity info (uuid, model, tools)
        - Last message has results (cost, duration, structured_output)

    Returns:
        Dict of field values ready to pass to AgentTrace(**fields).
    """
    first = messages[0]
    last = messages[-1]

    # Try to parse structured output from the ResultMessage
    output = None
    parse_error = None
    raw_structured_output = last.structured_output
    result_text = last.result if isinstance(last.result, str) else ""

    if raw_structured_output is not None:
        try:
            output = response_model.model_validate(raw_structured_output)
        except (ValidationError, json.JSONDecodeError, TypeError) as e:
            parse_error = f"{type(e).__name__}: {str(e)}"
    else:
        if last.result is None:
            parse_error = "No structured output or text result returned by Claude SDK"
        else:
            # No structured output usually means the agent hit the context limit.
            parse_error = (
                "No structured output returned (context limit likely exceeded)"
            )

    return dict(
        uuid=first.data.get("uuid") or "",
        session_id=last.session_id or "",
        model=first.data.get("model") or "",
        tools=first.data.get("tools") or [],
        duration_ms=last.duration_ms,
        total_cost_usd=last.total_cost_usd,
        num_turns=last.num_turns,
        usage=last.usage,
        result=result_text,
        is_error=last.is_error or parse_error is not None,
        output=output,
        parse_error=parse_error,
        raw_structured_output=raw_structured_output,
        messages=messages,
    )
