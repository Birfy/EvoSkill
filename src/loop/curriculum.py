"""Curriculum bandit for failure-category selection.

The default loop sweeps failure categories round-robin, treating every category
as equally worth proposing on. In practice some categories yield proposals that
repeatedly improve the held-out score while others never move it (too easy —
already solved, or too hard — no proposal helps). This bandit concentrates
sampling on the categories with the highest recent *learning gain*, defined as
the score delta of accepted proposals, while keeping an exploration floor so no
category is starved.

It is intentionally simple and dependency-free: a per-category EMA of observed
gain plus epsilon-floored weighted sampling without replacement. Untried
categories are always tried first so the bandit starts from full coverage.
"""

from __future__ import annotations

import random


class CategoryBandit:
    """Tracks per-category learning gain and samples high-gain categories.

    Args:
        categories: All known failure categories.
        epsilon: Exploration floor added to every category's sampling weight,
            as a fraction of the current max value (keeps low-value categories
            reachable). Clamped to a small absolute floor too.
        ema: Smoothing factor in [0, 1] for value updates. Higher = faster to
            track recent outcomes; lower = more stable.
    """

    def __init__(
        self,
        categories: list[str],
        *,
        epsilon: float = 0.15,
        ema: float = 0.5,
    ) -> None:
        self._epsilon = max(0.0, epsilon)
        self._ema = min(1.0, max(0.0, ema))
        self._value: dict[str, float] = {c: 0.0 for c in categories}
        self._seen: set[str] = set()

    def known_categories(self) -> list[str]:
        return list(self._value.keys())

    def value_of(self, category: str) -> float:
        return self._value.get(category, 0.0)

    def update(self, category: str, accepted: bool, score_delta: float) -> None:
        """Fold one proposal outcome into a category's value.

        Gain is the improvement an accepted proposal produced; a rejected or
        non-improving proposal contributes zero (not negative — we want to
        deprioritize unproductive categories, not punish exploration).
        """
        if category not in self._value:
            self._value[category] = 0.0
        gain = max(0.0, score_delta) if accepted else 0.0
        prev = self._value[category]
        self._value[category] = (1.0 - self._ema) * prev + self._ema * gain
        self._seen.add(category)

    def select(
        self,
        available: list[str],
        k: int,
        rng: random.Random | None = None,
    ) -> list[str]:
        """Pick up to ``k`` distinct categories, gain-weighted with exploration.

        Untried categories are returned first (round-robin-like cold start), then
        remaining slots are filled by sampling without replacement with weight
        ``value + floor``.
        """
        rng = rng or random
        k = max(0, min(k, len(available)))
        if k == 0:
            return []

        chosen: list[str] = []
        # Cold start: exhaust untried categories before exploiting values.
        untried = [c for c in available if c not in self._seen]
        rng.shuffle(untried)
        for c in untried:
            if len(chosen) >= k:
                return chosen
            chosen.append(c)

        remaining = [c for c in available if c not in chosen]
        if not remaining or len(chosen) >= k:
            return chosen

        max_val = max((self._value.get(c, 0.0) for c in remaining), default=0.0)
        floor = max(self._epsilon * max_val, 1e-6)
        while remaining and len(chosen) < k:
            weights = [self._value.get(c, 0.0) + floor for c in remaining]
            pick = rng.choices(remaining, weights=weights, k=1)[0]
            chosen.append(pick)
            remaining.remove(pick)
        return chosen
