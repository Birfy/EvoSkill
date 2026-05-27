import pytest
import asyncio
import sys
import types

from src.loop.config import LoopConfig
from src.loop.runner import SelfImprovingLoop
from src.loop.runner import ProgramSearchNode
from src.loop.runner import JudgeMatchResult
from src.loop.runner import BradleyTerryMatch


def test_judge_confidence_maps_to_elo_match_score():
    assert SelfImprovingLoop._judge_binary_to_match_score(True, 1.0) == 1.0
    assert SelfImprovingLoop._judge_binary_to_match_score(False, 1.0) == 0.0
    assert SelfImprovingLoop._judge_binary_to_match_score(True, 0.0) == 0.5
    assert SelfImprovingLoop._judge_binary_to_match_score(False, 0.0) == 0.5
    assert SelfImprovingLoop._judge_binary_to_match_score(True, 0.8) == pytest.approx(0.9)
    assert SelfImprovingLoop._judge_binary_to_match_score(False, 0.8) == pytest.approx(0.1)


def test_judge_probability_maps_to_match_score_with_root_cause_credit():
    assert SelfImprovingLoop._judge_probability_to_match_score(0.8, 0.8) == pytest.approx(0.8)
    assert SelfImprovingLoop._judge_probability_to_match_score(0.4, 0.8) == pytest.approx(0.5)
    assert SelfImprovingLoop._judge_probability_to_match_score(0.2, 1.0) == pytest.approx(0.4)


def test_elo_expected_probability_is_symmetric():
    equal = SelfImprovingLoop._elo_expected(1500.0, 1500.0, 400.0)
    stronger = SelfImprovingLoop._elo_expected(1700.0, 1500.0, 400.0)
    weaker = SelfImprovingLoop._elo_expected(1500.0, 1700.0, 400.0)

    assert equal == 0.5
    assert stronger > 0.5
    assert weaker < 0.5
    assert round(stronger + weaker, 12) == 1.0


def test_bradley_terry_fit_is_order_invariant_for_soft_matches():
    matches = [
        BradleyTerryMatch("skill-a", "base", 0.75),
        BradleyTerryMatch("skill-a", "base", 0.65),
        BradleyTerryMatch("skill-b", "base", 0.55),
        BradleyTerryMatch("skill-a", "skill-b", 0.70),
    ]

    ratings = SelfImprovingLoop._fit_bradley_terry_ratings(
        matches,
        ["base", "skill-a", "skill-b"],
        iterations=200,
    )
    reversed_ratings = SelfImprovingLoop._fit_bradley_terry_ratings(
        list(reversed(matches)),
        ["base", "skill-a", "skill-b"],
        iterations=200,
    )

    assert ratings["base"] == pytest.approx(1500.0)
    assert ratings["skill-a"] > ratings["skill-b"] > ratings["base"]
    assert ratings == pytest.approx(reversed_ratings)


def test_rating_to_score_uses_base_as_global_anchor():
    score = SelfImprovingLoop._rating_to_score(
        rating=1600.0,
        anchor_rating=1500.0,
        base_score=0.65,
        scale=400.0,
    )

    assert 0.65 < score < 1.0
    assert SelfImprovingLoop._rating_to_score(1500.0, 1500.0, 0.65, 400.0) == 0.65


def test_puct_node_balances_q_prior_and_visits():
    parent_visits = 10
    high_q = ProgramSearchNode("high-q", None, prior=0.1, score=0.8, visit_count=4, total_q=3.2)
    high_prior = ProgramSearchNode("high-prior", None, prior=0.9, score=0.3, visit_count=0, total_q=0.0)

    assert high_prior.puct_score(c_puct=0.5, parent_visits=parent_visits) > high_q.q_value
    assert high_q.puct_score(c_puct=0.0, parent_visits=parent_visits) == high_q.q_value


def test_puct_selection_returns_expandable_node_before_saturated_path():
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig(puct_children_per_node=1, puct_max_depth=2, puct_c=0.5)

    root = ProgramSearchNode("base", None, score=0.6, visit_count=2, total_q=1.2)
    first = ProgramSearchNode("iter-1", root, prior=0.9, score=0.8, visit_count=1, total_q=0.8, depth=1)
    root.children.append(first)

    selected = loop._select_puct_node(root)

    assert selected is first


def test_puct_selection_descends_before_filling_unsaturated_root():
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig(puct_children_per_node=4, puct_max_depth=2, puct_c=0.5)

    root = ProgramSearchNode("base", None, score=0.6, visit_count=2, total_q=1.2)
    first = ProgramSearchNode("iter-1", root, prior=0.9, score=0.8, visit_count=1, total_q=0.8, depth=1)
    root.children.append(first)

    selected = loop._select_puct_node(root)

    assert selected is first


def test_invalid_judge_match_is_neutral_and_marked_invalid():
    match = JudgeMatchResult(
        index=1,
        category="easy",
        would_succeed=False,
        confidence=0.0,
        match_score=0.5,
        expected_before=0.5,
        child_rating_after=1500.0,
        opponent_rating_after=1500.0,
        hypothetical_action="",
        reasoning="Judge call failed",
        raw_response="",
        valid=False,
    )

    assert match.valid is False
    assert match.match_score == 0.5
    assert match.confidence == 0.0


def test_openai_gpt5_judge_uses_max_completion_tokens(monkeypatch):
    loop = object.__new__(SelfImprovingLoop)
    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content='{"would_succeed": true}')
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key):
            self.chat = FakeChat()

    fake_openai = types.SimpleNamespace(AsyncOpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(
        "src.harness.provider_auth.ensure_provider_api_key",
        lambda provider: "test-key",
    )

    asyncio.run(loop._call_judge_api("prompt", "openai", "gpt-5.4-nano"))

    assert captured["max_completion_tokens"] == 512
    assert "max_tokens" not in captured


def test_judge_api_usage_updates_tokens_and_configured_cost():
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig(
        judge_input_cost_per_1m=1.0,
        judge_output_cost_per_1m=2.0,
        judge_log_details=False,
    )
    loop._iter_cost = 0.0
    loop._judge_cost_usd = 0.0
    loop._judge_prompt_tokens = 0
    loop._judge_completion_tokens = 0
    loop._judge_total_tokens = 0

    usage = types.SimpleNamespace(prompt_tokens=1000, completion_tokens=2000, total_tokens=3000)

    loop._record_judge_api_usage("openai", "test-model", usage)

    assert loop._judge_prompt_tokens == 1000
    assert loop._judge_completion_tokens == 2000
    assert loop._judge_total_tokens == 3000
    assert loop._judge_cost_usd == pytest.approx(0.005)
    assert loop._iter_cost == pytest.approx(0.005)
