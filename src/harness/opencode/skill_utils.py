"""OpenCode skill file utilities.

Handles YAML frontmatter normalization for SKILL.md files so that
OpenCode can discover project-local skills. Claude SDK reads SKILL.md
natively and doesn't need this processing.
"""

import re
from pathlib import Path
from typing import Any, Iterable

import yaml


_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)

DEFAULT_OPENHANDS_SKILL_TRIGGERS = [
    "task",
    "artifact",
    "file",
    "document",
    "data",
    "tool",
    "command",
    "API",
    "calculate",
    "compute",
    "transform",
    "validate",
    "verify",
    "format",
    "output",
    "edit",
    "update",
]


def _normalize_skill_description(description: str) -> str:
    cleaned = " ".join(description.split())
    if not cleaned:
        return "Reusable benchmark skill."
    if len(cleaned) <= 1024:
        return cleaned
    truncated = cleaned[:1021].rstrip()
    return f"{truncated}..."


def _normalize_skill_triggers(triggers: Iterable[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for trigger in triggers or []:
        cleaned = " ".join(str(trigger).split())
        key = cleaned.lower()
        if cleaned and key not in seen:
            normalized.append(cleaned)
            seen.add(key)
    return normalized


def ensure_skill_frontmatter(
    skill_path: Path,
    *,
    description: str,
    compatibility: str | None = None,
    triggers: Iterable[str] | None = None,
) -> bool:
    """Ensure a SKILL.md file has discoverable YAML frontmatter.

    Returns True if the file was rewritten.
    """
    if not skill_path.exists():
        return False

    skill_name = skill_path.parent.name
    normalized_description = _normalize_skill_description(description)
    normalized_triggers = _normalize_skill_triggers(triggers)
    original_text = skill_path.read_text()
    body = original_text
    metadata: dict[str, Any] = {}

    match = _FRONTMATTER_RE.match(original_text)
    if match:
        body = original_text[match.end() :].lstrip("\n")
        try:
            parsed = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            parsed = {}
        if isinstance(parsed, dict):
            metadata = {str(key): value for key, value in parsed.items()}

    changed = False
    if str(metadata.get("name", "")) != skill_name:
        metadata["name"] = skill_name
        changed = True
    if not str(metadata.get("description", "")).strip():
        metadata["description"] = normalized_description
        changed = True
    if compatibility and not str(metadata.get("compatibility", "")).strip():
        metadata["compatibility"] = compatibility
        changed = True
    if normalized_triggers and not metadata.get("triggers"):
        metadata["triggers"] = normalized_triggers
        changed = True

    if not match and not changed:
        return False
    if not match:
        changed = True

    if not changed:
        return False

    frontmatter = yaml.safe_dump(metadata, sort_keys=False).strip()
    skill_path.write_text(f"---\n{frontmatter}\n---\n\n{body.lstrip()}")
    return True


def normalize_project_skill_frontmatter(
    project_root: Path,
    *,
    descriptions: dict[str, str] | None = None,
    fallback_description: str = "Reusable benchmark skill.",
    compatibility: str | None = None,
    triggers: Iterable[str] | None = None,
    triggers_by_skill: dict[str, Iterable[str]] | None = None,
) -> list[str]:
    """Normalize every project-local SKILL.md file under .claude/skills."""
    skills_dir = project_root / ".claude" / "skills"
    if not skills_dir.exists():
        return []

    descriptions = descriptions or {}
    triggers_by_skill = triggers_by_skill or {}
    normalized: list[str] = []
    for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
        skill_name = skill_path.parent.name
        description = descriptions.get(skill_name, fallback_description)
        skill_triggers = triggers_by_skill.get(skill_name, triggers)
        if ensure_skill_frontmatter(
            skill_path,
            description=description,
            compatibility=compatibility,
            triggers=skill_triggers,
        ):
            normalized.append(skill_name)
    return normalized
