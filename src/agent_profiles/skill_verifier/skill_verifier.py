from __future__ import annotations

from typing import Any

from src.harness import build_options
from src.schemas import SkillVerifierResponse
from src.agent_profiles.skill_verifier.prompt import SKILL_VERIFIER_SYSTEM_PROMPT


# Information isolation: the verifier must NOT be able to read ground-truth data
# files (dataset answers, gold outputs). It receives the SKILL.md content in its
# query, so it needs no filesystem/shell tools. An empty toolset both enforces the
# isolation and makes each verifier call a single completion (no tool-use loop) —
# materially fewer tokens and round-trips than an agentic session.
SKILL_VERIFIER_TOOLS: list[str] = []


def get_skill_verifier_options(
    model: str | None = None,
    project_root: str | None = None,
) -> Any:
    return build_options(
        system=SKILL_VERIFIER_SYSTEM_PROMPT.strip(),
        schema=SkillVerifierResponse.model_json_schema(),
        tools=SKILL_VERIFIER_TOOLS,
        project_root=project_root,
        model=model,
    )


def make_skill_verifier_options(
    *,
    project_root: str | None = None,
    model: str | None = None,
):
    return get_skill_verifier_options(model=model, project_root=project_root)


skill_verifier_options = get_skill_verifier_options()
