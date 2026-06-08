"""Tests for the judge-loop QD coverage vector (Phase 2 ported into _run_with_llm_judge)."""

from types import SimpleNamespace

from src.loop.runner import SelfImprovingLoop


def _match(category, match_score, valid=True):
    return SimpleNamespace(category=category, match_score=match_score, valid=valid)


def _judge(matches):
    return SimpleNamespace(matches=matches)


def test_coverage_means_per_category():
    jr = _judge([
        _match("tax", 0.8),
        _match("tax", 0.6),
        _match("unit", 0.4),
    ])
    cov = SelfImprovingLoop._judge_coverage_vector(jr)
    assert cov["tax"] == 0.7
    assert cov["unit"] == 0.4


def test_coverage_skips_invalid_matches():
    jr = _judge([
        _match("tax", 0.9),
        _match("tax", 0.1, valid=False),  # ignored
    ])
    cov = SelfImprovingLoop._judge_coverage_vector(jr)
    assert cov["tax"] == 0.9


def test_coverage_empty_when_no_matches():
    assert SelfImprovingLoop._judge_coverage_vector(_judge([])) == {}


def test_coverage_uncategorized_bucket():
    cov = SelfImprovingLoop._judge_coverage_vector(_judge([_match("", 0.5)]))
    assert cov == {"uncategorized": 0.5}
