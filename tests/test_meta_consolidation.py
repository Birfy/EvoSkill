"""Tests for the epoch slow/meta consolidation (SkillOpt port into the judge loop)."""

from src.loop.helpers import consolidate_meta, read_meta, write_meta


def _rec(outcome, delta, blockers=None, source="failure", skill="sk"):
    return {
        "outcome": outcome,
        "delta": delta,
        "blockers": blockers or [],
        "root_cause": "",
        "source": source,
        "skill": skill,
    }


def test_empty_records_returns_empty():
    assert consolidate_meta([]) == ""


def test_buckets_improvements_kept_regressions():
    recs = [
        _rec("improved", 0.1),
        _rec("kept", 0.0),
        _rec("discarded", -0.2),
    ]
    out = consolidate_meta(recs)
    assert "improvements: 1 | kept: 1 | regressions: 1" in out


def test_persistent_blockers_ranked_by_frequency():
    recs = [
        _rec("discarded", -0.1, blockers=["unit scale", "rounding"]),
        _rec("not_frontier", -0.05, blockers=["unit scale"]),
    ]
    out = consolidate_meta(recs, top_k=5)
    # most frequent blocker listed first with its count
    assert "unit scale (x2)" in out
    assert "rounding (x1)" in out
    assert out.index("unit scale") < out.index("rounding")


def test_winning_skills_only_from_improvements():
    recs = [
        _rec("improved", 0.1, skill="good"),
        _rec("improved", 0.2, skill="good"),
        _rec("discarded", -0.1, skill="bad"),  # must not appear in "what worked"
    ]
    out = consolidate_meta(recs)
    assert "good (x2)" in out
    assert "bad" not in out.split("persistent")[0]  # not in the "what worked" line


def test_source_mix_and_induction_winrate():
    recs = [
        _rec("improved", 0.1, source="induction", skill="s"),
        _rec("discarded", -0.1, source="induction", skill="s"),
        _rec("improved", 0.1, source="failure", skill="s"),
    ]
    out = consolidate_meta(recs)
    assert "failure 1 / induction 2" in out
    assert "induction wins: 1/2" in out


def test_top_k_caps_listed_blockers():
    recs = [_rec("discarded", -0.1, blockers=[f"b{i}"]) for i in range(10)]
    out = consolidate_meta(recs, top_k=3)
    blocker_line = [l for l in out.splitlines() if "persistent unresolved" in l][0]
    assert blocker_line.count("(x1)") == 3  # only top_k listed


def test_meta_file_roundtrip(tmp_path):
    p = tmp_path / ".evoskill" / "meta.md"
    assert read_meta(p) == ""  # absent -> empty
    write_meta(p, "### meta\n- x")
    assert read_meta(p) == "### meta\n- x"
