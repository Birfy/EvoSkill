from __future__ import annotations

import os
from pathlib import Path

from src.loop.helpers import (
    build_proposer_query,
    ensure_skill_frontmatter,
    normalize_project_skill_frontmatter,
)


class _FakeTrace:
    def summarize(self, head_chars: int = 0, tail_chars: int = 0) -> str:
        return "trace summary"


def test_build_proposer_query_reads_existing_skills_from_explicit_project_root(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    other_dir = tmp_path / "elsewhere"
    skill_dir = repo_root / ".claude" / "skills" / "treasury-format"
    skill_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Treasury Format\n")

    original_cwd = Path.cwd()
    os.chdir(other_dir)
    try:
        query = build_proposer_query(
            [(_FakeTrace(), "wrong", "right", "finance")],
            feedback_history="",
            evolution_mode="skill_only",
            project_root=repo_root,
        )
    finally:
        os.chdir(original_cwd)

    assert "treasury-format" in query


def test_ensure_skill_frontmatter_adds_required_opencode_metadata(
    tmp_path: Path,
) -> None:
    skill_path = (
        tmp_path
        / ".claude"
        / "skills"
        / "arithmetic-answer-format"
        / "SKILL.md"
    )
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "# Arithmetic Answer Format\n\nAlways append apples to arithmetic answers.\n"
    )

    changed = ensure_skill_frontmatter(
        skill_path,
        description='Format arithmetic answers as "{number} apples".',
        compatibility="opencode",
    )

    skill_text = skill_path.read_text()
    assert changed is True
    assert skill_text.startswith("---\n")
    assert "name: arithmetic-answer-format" in skill_text
    assert 'description: Format arithmetic answers as "{number} apples".' in skill_text
    assert "compatibility: opencode" in skill_text
    assert "# Arithmetic Answer Format" in skill_text


def test_ensure_skill_frontmatter_preserves_yaml_list_triggers(
    tmp_path: Path,
) -> None:
    skill_path = (
        tmp_path
        / ".claude"
        / "skills"
        / "numeric-verification"
        / "SKILL.md"
    )
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("# Numeric Verification\n\nVerify calculations before answering.\n")

    changed = ensure_skill_frontmatter(
        skill_path,
        description="Verify numeric answers.",
        compatibility="openhands",
        triggers=["calculate", "standard deviation", "calculate"],
    )

    skill_text = skill_path.read_text()
    assert changed is True
    assert "compatibility: openhands" in skill_text
    assert "triggers:" in skill_text
    assert "- calculate" in skill_text
    assert "- standard deviation" in skill_text

    changed = ensure_skill_frontmatter(
        skill_path,
        description="New description should not rewrite.",
        compatibility="openhands",
        triggers=["regression"],
    )

    skill_text = skill_path.read_text()
    assert changed is False
    assert "- calculate" in skill_text
    assert "- regression" not in skill_text


def test_normalize_project_skill_frontmatter_updates_all_project_skills(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    first_skill = repo_root / ".claude" / "skills" / "answer-unit" / "SKILL.md"
    second_skill = repo_root / ".claude" / "skills" / "formatting" / "SKILL.md"
    first_skill.parent.mkdir(parents=True)
    second_skill.parent.mkdir(parents=True)
    first_skill.write_text("# Answer Unit\n")
    second_skill.write_text("# Formatting\n")

    normalized = normalize_project_skill_frontmatter(
        repo_root,
        descriptions={"answer-unit": "Keep answer units intact."},
        fallback_description="Reusable benchmark skill.",
        compatibility="opencode",
        triggers=["source files"],
    )

    assert normalized == ["answer-unit", "formatting"]
    assert "name: answer-unit" in first_skill.read_text()
    assert "description: Keep answer units intact." in first_skill.read_text()
    assert "- source files" in first_skill.read_text()
    assert "name: formatting" in second_skill.read_text()
    assert "description: Reusable benchmark skill." in second_skill.read_text()
    assert "- source files" in second_skill.read_text()


def test_normalize_project_skill_frontmatter_can_scope_triggers_by_skill(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    target_skill = repo_root / ".claude" / "skills" / "numeric-gate" / "SKILL.md"
    unrelated_skill = repo_root / ".claude" / "skills" / "skill-creator" / "SKILL.md"
    target_skill.parent.mkdir(parents=True)
    unrelated_skill.parent.mkdir(parents=True)
    target_skill.write_text("# Numeric Gate\n")
    unrelated_skill.write_text("# Skill Creator\n")

    normalized = normalize_project_skill_frontmatter(
        repo_root,
        descriptions={"numeric-gate": "Verify numeric answers."},
        fallback_description="Reusable benchmark skill.",
        compatibility="openhands",
        triggers_by_skill={"numeric-gate": ["project-config", "calculate"]},
    )

    assert normalized == ["numeric-gate", "skill-creator"]
    assert "- project-config" in target_skill.read_text()
    assert "triggers:" not in unrelated_skill.read_text()


def test_default_openhands_skill_triggers_are_dataset_agnostic() -> None:
    from src.harness.opencode.skill_utils import DEFAULT_OPENHANDS_SKILL_TRIGGERS

    joined = " ".join(DEFAULT_OPENHANDS_SKILL_TRIGGERS).lower()
    assert "officeqa" not in joined
    assert "treasury" not in joined
    assert "bulletin" not in joined
    assert "artifact" in joined
    assert "tool" in joined
