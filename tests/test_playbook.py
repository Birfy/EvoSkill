"""Tests for the ACE-style skill playbook (Phase 1)."""

from pathlib import Path

import pytest

from src.loop.playbook import (
    BEGIN,
    END,
    Bullet,
    Playbook,
    apply_playbook_delta,
    merge_delta,
    parse_playbook,
    prune,
    render_playbook,
    update_bullet_counters,
    write_playbook,
)
from src.schemas import BulletOp, SkillProposerResponse


def test_parse_render_roundtrip():
    text = f"""# Skill

Some prose.

{BEGIN}
- [b1] (helpful:3 harmful:1) use difference for percentage points
- [b2] (helpful:0 harmful:0) fall back to ALTO coordinates
{END}
"""
    pb = parse_playbook(text)
    assert [b.id for b in pb.bullets] == ["b1", "b2"]
    assert pb.by_id("b1").helpful == 3 and pb.by_id("b1").harmful == 1
    # Re-rendering and re-parsing preserves everything.
    pb2 = parse_playbook(write_playbook(text, pb))
    assert pb2.bullets[0].render() == pb.bullets[0].render()


def test_parse_empty_when_no_block():
    assert parse_playbook("# Skill\n\nno playbook here").bullets == []


def test_add_creates_new_bullet_and_reports_id():
    pb = Playbook()
    pb, added = merge_delta(pb, [{"op": "add", "text": "never use an external source"}])
    assert added == ["b1"]
    assert pb.bullets[0].text == "never use an external source"


def test_add_dedup_credits_existing_without_bumping():
    pb = Playbook([Bullet(id="b1", text="use difference for percentage points")])
    pb, credited = merge_delta(
        pb,
        [{"op": "add", "text": "use difference for percentage points"}],
        dedup_threshold=0.85,
    )
    assert len(pb.bullets) == 1  # no duplicate created
    assert credited == ["b1"]  # existing bullet credited for post-gate scoring
    assert pb.by_id("b1").helpful == 0  # NOT bumped pre-gate (gate-driven only)


def test_reinforce_credits_without_bumping_and_retire_drops():
    pb = Playbook([Bullet(id="b1", text="rule one"), Bullet(id="b2", text="rule two")])
    pb, credited = merge_delta(pb, [{"op": "reinforce", "target_id": "b1"}])
    assert credited == ["b1"]
    assert pb.by_id("b1").helpful == 0  # reinforce only marks for gate credit
    pb, credited = merge_delta(pb, [{"op": "retire", "target_id": "b2"}])
    assert pb.by_id("b2") is None
    assert credited == []


def test_prune_drops_net_harmful_first():
    pb = Playbook(
        [
            Bullet(id="b1", text="good", helpful=5, harmful=0),
            Bullet(id="b2", text="bad", helpful=0, harmful=4),  # net harmful
            Bullet(id="b3", text="ok", helpful=2, harmful=1),
        ]
    )
    pb = prune(pb, max_bullets=2)
    ids = {b.id for b in pb.bullets}
    assert "b2" not in ids  # net-harmful dropped first
    assert ids == {"b1", "b3"}


def test_next_id_avoids_collisions():
    pb = Playbook([Bullet(id="b1", text="x"), Bullet(id="b3", text="y")])
    assert pb.next_id() == "b2"


def _write_skill(root: Path, name: str, body: str) -> Path:
    d = root / ".claude" / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text(body)
    return p


def test_apply_delta_and_counter_update_on_disk(tmp_path):
    p = _write_skill(tmp_path, "demo", "# Demo\n\nprose only\n")
    added = apply_playbook_delta(
        tmp_path, "demo", [{"op": "add", "text": "guard the source binding"}]
    )
    assert added == ["b1"]
    assert BEGIN in p.read_text()

    # Helpful outcome bumps the counter.
    update_bullet_counters(tmp_path, "demo", added, helpful=True)
    pb = parse_playbook(p.read_text())
    assert pb.by_id("b1").helpful == 1 and pb.by_id("b1").harmful == 0

    # Harmful outcome bumps the other counter.
    update_bullet_counters(tmp_path, "demo", added, helpful=False)
    pb = parse_playbook(p.read_text())
    assert pb.by_id("b1").harmful == 1


def test_apply_delta_noop_when_skill_missing(tmp_path):
    assert apply_playbook_delta(tmp_path, "ghost", [{"op": "add", "text": "x"}]) == []


def test_apply_delta_prune_drops_credited_id(tmp_path):
    # A skill already at capacity with strong bullets; a weak new add gets pruned,
    # so it must not be reported as a surviving added id.
    body_lines = [BEGIN] + [
        f"- [b{i}] (helpful:9 harmful:0) strong rule {i}" for i in range(1, 4)
    ] + [END]
    _write_skill(tmp_path, "full", "# Full\n\n" + "\n".join(body_lines) + "\n")
    added = apply_playbook_delta(
        tmp_path, "full", [{"op": "add", "text": "weak new rule"}], max_bullets=3
    )
    assert added == []  # the new (helpful:0) bullet loses pruning and is not credited


def test_credit_playbook_bumps_every_applied_skill(tmp_path):
    from src.loop.runner import SelfImprovingLoop, MutationResult

    # Two skills, each with a fresh bullet to credit.
    _write_skill(tmp_path, "skA", "# A\n")
    _write_skill(tmp_path, "skB", "# B\n")
    ba = apply_playbook_delta(tmp_path, "skA", [{"op": "add", "text": "rule a"}])
    bb = apply_playbook_delta(tmp_path, "skB", [{"op": "add", "text": "rule b"}])

    loop = object.__new__(SelfImprovingLoop)
    loop._project_root = tmp_path
    mr = MutationResult(
        child_name="c", proposal="p", justification="j", proposer_confidence=0.5,
        applied_skills=[
            {"skill_name": "skA", "credited_bullets": ba},
            {"skill_name": "skB", "credited_bullets": bb},
        ],
    )
    total = loop._credit_playbook(mr)
    assert total == 2
    assert parse_playbook((tmp_path / ".claude/skills/skA/SKILL.md").read_text()).by_id(ba[0]).helpful == 1
    assert parse_playbook((tmp_path / ".claude/skills/skB/SKILL.md").read_text()).by_id(bb[0]).helpful == 1


def test_credit_playbook_falls_back_to_singular_fields(tmp_path):
    from src.loop.runner import SelfImprovingLoop, MutationResult

    _write_skill(tmp_path, "solo", "# Solo\n")
    bullets = apply_playbook_delta(tmp_path, "solo", [{"op": "add", "text": "x"}])
    loop = object.__new__(SelfImprovingLoop)
    loop._project_root = tmp_path
    mr = MutationResult(
        child_name="c", proposal="p", justification="j", proposer_confidence=0.5,
        target_skill="solo", credited_bullets=bullets,  # no applied_skills
    )
    assert loop._credit_playbook(mr) == 1


def test_schema_bullet_op_validation():
    # add requires text
    with pytest.raises(Exception):
        BulletOp(op="add", text="")
    # reinforce requires target_id
    with pytest.raises(Exception):
        BulletOp(op="reinforce")
    ok = SkillProposerResponse(
        proposed_skill="x",
        justification="y",
        should_apply_when="a",
        should_not_apply_when="b",
        invariants_to_preserve="c",
        regression_risks="d",
        bullet_ops=[BulletOp(op="add", text="a concrete transferable rule")],
    )
    assert ok.bullet_ops[0].text
