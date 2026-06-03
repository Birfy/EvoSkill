"""Tests for the curriculum bandit (Phase 3)."""

import random

from src.loop.curriculum import CategoryBandit


def test_cold_start_returns_untried_first():
    b = CategoryBandit(["a", "b", "c"])
    picked = b.select(["a", "b", "c"], 3, rng=random.Random(0))
    assert set(picked) == {"a", "b", "c"}  # all untried, all returned


def test_select_respects_k_and_distinctness():
    b = CategoryBandit(["a", "b", "c", "d"])
    picked = b.select(["a", "b", "c", "d"], 2, rng=random.Random(1))
    assert len(picked) == 2 and len(set(picked)) == 2


def test_update_only_credits_accepted_gains():
    b = CategoryBandit(["a", "b"], ema=1.0)
    b.update("a", accepted=True, score_delta=0.4)
    b.update("b", accepted=False, score_delta=0.9)  # not accepted -> zero gain
    assert b.value_of("a") == 0.4
    assert b.value_of("b") == 0.0


def test_negative_delta_does_not_punish():
    b = CategoryBandit(["a"], ema=1.0)
    b.update("a", accepted=True, score_delta=-0.3)
    assert b.value_of("a") == 0.0  # max(0, delta)


def test_bandit_prefers_high_gain_category():
    rng = random.Random(42)
    b = CategoryBandit(["a", "b"], epsilon=0.05, ema=1.0)
    # Make both "seen" so cold-start does not force selection.
    b.update("a", accepted=True, score_delta=0.8)
    b.update("b", accepted=False, score_delta=0.0)
    counts = {"a": 0, "b": 0}
    for _ in range(400):
        pick = b.select(["a", "b"], 1, rng=rng)[0]
        counts[pick] += 1
    assert counts["a"] > counts["b"] * 2  # strongly favors the high-gain category


def test_ema_smooths_updates():
    b = CategoryBandit(["a"], ema=0.5)
    b.update("a", accepted=True, score_delta=1.0)  # 0.5
    assert abs(b.value_of("a") - 0.5) < 1e-9
    b.update("a", accepted=True, score_delta=0.0)  # 0.25
    assert abs(b.value_of("a") - 0.25) < 1e-9
