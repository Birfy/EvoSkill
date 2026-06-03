"""ACE-style bullet playbook for skills.

Instead of letting the generator rewrite a whole SKILL.md on every proposal —
which causes "context collapse" (details eroded by repeated rewrites) and
"brevity bias" (specific heuristics compressed into vague "verify more" prose) —
each skill carries a *managed playbook*: an itemized list of bullets, each with
an id and helpful/harmful counters. Proposals are applied as small deterministic
deltas (add / reinforce / retire), de-duplicated by token overlap, and bullets
the held-out gate finds harmful are pruned. Counters give bullet-level credit
assignment, finer than document-level skill edits.

The playbook lives inside SKILL.md between sentinel markers so it round-trips
losslessly and leaves the rest of the (human-authored) skill prose untouched.
The merge logic is intentionally non-LLM and dependency-free: similarity uses
Jaccard over lowercased word tokens, which is deterministic and good enough for
near-duplicate detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

BEGIN = "<!-- ACE-PLAYBOOK (managed; do not hand-edit) -->"
END = "<!-- /ACE-PLAYBOOK -->"

# - [b3] (helpful:5 harmful:0) text...
_BULLET_RE = re.compile(
    r"^- \[(?P<id>b\d+)\] \(helpful:(?P<h>\d+) harmful:(?P<n>\d+)\) (?P<text>.*)$"
)
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Bullet:
    id: str
    text: str
    helpful: int = 0
    harmful: int = 0

    def render(self) -> str:
        return f"- [{self.id}] (helpful:{self.helpful} harmful:{self.harmful}) {self.text}"


@dataclass
class Playbook:
    bullets: list[Bullet] = field(default_factory=list)

    def next_id(self) -> str:
        used = {int(b.id[1:]) for b in self.bullets if b.id[1:].isdigit()}
        n = 1
        while n in used:
            n += 1
        return f"b{n}"

    def by_id(self, bid: str) -> Bullet | None:
        return next((b for b in self.bullets if b.id == bid), None)


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def parse_playbook(skill_text: str) -> Playbook:
    """Extract the managed playbook block from a SKILL.md body."""
    start = skill_text.find(BEGIN)
    end = skill_text.find(END)
    if start == -1 or end == -1 or end < start:
        return Playbook()
    block = skill_text[start + len(BEGIN) : end]
    bullets: list[Bullet] = []
    for line in block.splitlines():
        m = _BULLET_RE.match(line.strip())
        if m:
            bullets.append(
                Bullet(
                    id=m.group("id"),
                    text=m.group("text").strip(),
                    helpful=int(m.group("h")),
                    harmful=int(m.group("n")),
                )
            )
    return Playbook(bullets)


def render_playbook(pb: Playbook) -> str:
    """Render the managed block (markers + bullets)."""
    lines = [BEGIN]
    for b in pb.bullets:
        lines.append(b.render())
    lines.append(END)
    return "\n".join(lines)


def write_playbook(skill_text: str, pb: Playbook) -> str:
    """Return ``skill_text`` with its managed block replaced/inserted."""
    rendered = render_playbook(pb)
    start = skill_text.find(BEGIN)
    end = skill_text.find(END)
    if start != -1 and end != -1 and end >= start:
        return skill_text[:start] + rendered + skill_text[end + len(END) :]
    sep = "" if skill_text.endswith("\n") or not skill_text else "\n\n"
    return f"{skill_text}{sep}{rendered}\n"


def merge_delta(
    pb: Playbook,
    ops: list[dict],
    *,
    dedup_threshold: float = 0.85,
    max_bullets: int = 12,
) -> tuple[Playbook, list[str]]:
    """Apply add/reinforce/retire ops deterministically.

    Returns the updated playbook and the *credited* bullet ids — the bullets this
    delta added or reinforced. Counters are NOT bumped here: ``helpful`` is driven
    only by the held-out gate (the caller bumps the credited set when, and only
    when, the resulting child improves), so the signal stays gate-driven rather
    than reflecting mere proposer intent.

    - ``add``: append a new bullet unless it near-duplicates an existing one
      (>= ``dedup_threshold`` Jaccard), in which case the existing bullet is
      credited instead of duplicated.
    - ``reinforce`` (with target_id): credit the named existing bullet.
    - ``retire`` (with target_id): drop the named bullet.

    After applying ops the playbook is pruned to ``max_bullets``, dropping
    net-harmful bullets first, then the least-helpful.
    """
    credited: list[str] = []
    for op in ops:
        kind = (op.get("op") or "").strip().lower()
        text = (op.get("text") or "").strip()
        target = (op.get("target_id") or "").strip()

        if kind == "retire" and target:
            pb.bullets = [b for b in pb.bullets if b.id != target]
            continue
        if kind == "reinforce" and target:
            if pb.by_id(target) is not None:
                credited.append(target)
            continue
        if kind == "add" and text:
            dup = max(
                pb.bullets,
                key=lambda b: _similarity(b.text, text),
                default=None,
            )
            if dup is not None and _similarity(dup.text, text) >= dedup_threshold:
                credited.append(dup.id)  # same rule already present; credit it
                continue
            bid = pb.next_id()
            pb.bullets.append(Bullet(id=bid, text=text))
            credited.append(bid)

    pb = prune(pb, max_bullets)
    # De-dupe (preserving order) and drop ids pruning removed.
    live = {b.id for b in pb.bullets}
    credited = [bid for bid in dict.fromkeys(credited) if bid in live]
    return pb, credited


def prune(pb: Playbook, max_bullets: int) -> Playbook:
    """Keep at most ``max_bullets``: net-harmful first to go, then least helpful."""
    if max_bullets <= 0 or len(pb.bullets) <= max_bullets:
        return pb
    # Sort worst-first: net-harmful (harmful>helpful) before net-positive; within
    # that, lower (helpful-harmful) and lower helpful are dropped first.
    ordered = sorted(
        pb.bullets,
        key=lambda b: (b.helpful - b.harmful, b.helpful),
    )
    drop = set(id(b) for b in ordered[: len(pb.bullets) - max_bullets])
    pb.bullets = [b for b in pb.bullets if id(b) not in drop]
    return pb


def _skill_path(project_root: Path, skill_name: str) -> Path:
    return project_root / ".claude" / "skills" / skill_name / "SKILL.md"


def apply_playbook_delta(
    project_root: Path,
    skill_name: str,
    ops: list[dict],
    *,
    dedup_threshold: float = 0.85,
    max_bullets: int = 12,
) -> list[str]:
    """Load a skill, apply a playbook delta, write it back.

    Returns the credited bullet ids (added or reinforced). Safe no-op (returns
    []) when the skill file does not exist or there are no ops.
    """
    if not ops:
        return []
    path = _skill_path(project_root, skill_name)
    if not path.exists():
        return []
    text = path.read_text()
    pb = parse_playbook(text)
    pb, credited = merge_delta(
        pb, ops, dedup_threshold=dedup_threshold, max_bullets=max_bullets
    )
    path.write_text(write_playbook(text, pb))
    return credited


def update_bullet_counters(
    project_root: Path,
    skill_name: str,
    bullet_ids: list[str],
    *,
    helpful: bool,
) -> None:
    """Bump helpful or harmful counters for the given bullets in a skill."""
    if not skill_name or not bullet_ids:
        return
    path = _skill_path(project_root, skill_name)
    if not path.exists():
        return
    text = path.read_text()
    pb = parse_playbook(text)
    touched = False
    for bid in bullet_ids:
        b = pb.by_id(bid)
        if b is None:
            continue
        if helpful:
            b.helpful += 1
        else:
            b.harmful += 1
        touched = True
    if touched:
        path.write_text(write_playbook(text, pb))
