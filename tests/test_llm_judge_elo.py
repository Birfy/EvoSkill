import pytest
import asyncio
import sys
import types

from src.loop.config import LoopConfig
from src.loop.runner import SelfImprovingLoop
from src.loop.runner import ProgramSearchNode
from src.loop.runner import JudgeMatchResult
from src.loop.runner import BradleyTerryMatch


def _entry(question: str, category: str):
    # (trace, question, agent_answer, ground_truth, category, failure_type, feedback)
    return (None, question, "ans", "gt", category, "wrong_answer", "")


def test_split_holdout_produces_disjoint_pools():
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig(judge_holdout_ratio=0.5)
    failures = [_entry(f"q{i}", "math") for i in range(4)] + [
        _entry(f"f{i}", "finance") for i in range(4)
    ]

    proposer, judge = loop._split_holdout(failures)

    assert proposer is not judge
    p_q = {e[1] for e in proposer}
    j_q = {e[1] for e in judge}
    assert p_q and j_q
    assert p_q.isdisjoint(j_q)  # judge never sees a proposer-visible case
    assert p_q | j_q == {e[1] for e in failures}


def test_split_holdout_disabled_shares_all_cases():
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig(judge_holdout_ratio=0.0)
    failures = [_entry(f"q{i}", "math") for i in range(4)]

    proposer, judge = loop._split_holdout(failures)

    assert proposer is judge is failures


def test_split_holdout_keeps_singleton_category_for_proposer():
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig(judge_holdout_ratio=0.5)
    failures = [_entry("only", "rare")] + [_entry(f"q{i}", "math") for i in range(4)]

    proposer, judge = loop._split_holdout(failures)

    assert "only" in {e[1] for e in proposer}
    assert "only" not in {e[1] for e in judge}


def test_sample_judge_failures_draws_only_from_held_out_pool():
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig()
    loop._judge_cat_offset = {}
    judge_pool = [_entry(f"j{i}", "math") for i in range(3)]

    sampled = loop._sample_judge_failures(judge_pool, ["math"], count=2)

    assert len(sampled) == 2
    assert {e[1] for e in sampled}.issubset({e[1] for e in judge_pool})


def test_judge_confidence_maps_to_elo_match_score():
    assert SelfImprovingLoop._judge_binary_to_match_score(True, 1.0) == 1.0
    assert SelfImprovingLoop._judge_binary_to_match_score(False, 1.0) == 0.0
    assert SelfImprovingLoop._judge_binary_to_match_score(True, 0.0) == 0.5
    assert SelfImprovingLoop._judge_binary_to_match_score(False, 0.0) == 0.5
    assert SelfImprovingLoop._judge_binary_to_match_score(True, 0.8) == pytest.approx(0.9)
    assert SelfImprovingLoop._judge_binary_to_match_score(False, 0.8) == pytest.approx(0.1)


def test_judge_bool_parser_handles_string_values():
    assert SelfImprovingLoop._parse_judge_bool(True) is True
    assert SelfImprovingLoop._parse_judge_bool(False) is False
    assert SelfImprovingLoop._parse_judge_bool("true") is True
    assert SelfImprovingLoop._parse_judge_bool("false") is False
    assert SelfImprovingLoop._parse_judge_bool("0") is False
    assert SelfImprovingLoop._parse_judge_bool("1") is True
    assert SelfImprovingLoop._parse_judge_bool("unexpected", default=True) is True


def test_judge_probability_maps_to_match_score_with_root_cause_credit():
    assert SelfImprovingLoop._judge_probability_to_match_score(0.8, 0.8) == pytest.approx(0.8)
    assert SelfImprovingLoop._judge_probability_to_match_score(0.4, 0.8) == pytest.approx(0.5)
    assert SelfImprovingLoop._judge_probability_to_match_score(0.2, 1.0) == pytest.approx(0.4)


def test_judge_relative_score_centers_on_parent_probability():
    score, parent, candidate, advantage = SelfImprovingLoop._judge_relative_to_match_score(
        {
            "parent_success_prob": 0.60,
            "candidate_success_prob": 0.80,
        }
    )

    assert parent == pytest.approx(0.60)
    assert candidate == pytest.approx(0.80)
    assert advantage == pytest.approx(0.20)
    assert score == pytest.approx(0.60)


def test_judge_relative_score_uses_explicit_match_score():
    score, _parent, _candidate, advantage = SelfImprovingLoop._judge_relative_to_match_score(
        {
            "parent_success_prob": 0.80,
            "candidate_success_prob": 0.70,
            "relative_advantage": -0.10,
            "match_score": 0.42,
        }
    )

    assert score == pytest.approx(0.42)
    assert advantage == pytest.approx(-0.10)


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


def test_rating_to_score_keeps_weaker_nodes_below_base():
    score = SelfImprovingLoop._rating_to_score(
        rating=1400.0,
        anchor_rating=1500.0,
        base_score=0.65,
        scale=400.0,
    )

    assert 0.0 < score < 0.65


def test_rating_to_score_with_uncertainty_applies_penalty():
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig(judge_bt_uncertainty_penalty=0.05)

    raw = SelfImprovingLoop._rating_to_score(1600.0, 1500.0, 0.65, 400.0)
    penalized = loop._rating_to_score_with_uncertainty(
        rating=1600.0,
        anchor_rating=1500.0,
        base_score=0.65,
        scale=400.0,
        uncertainty=0.5,
    )

    assert penalized == pytest.approx(raw - 0.025)


def test_bt_player_uncertainty_reflects_sparse_matches():
    one_match = [BradleyTerryMatch("skill-a", "base", 0.7)]
    four_matches = [
        BradleyTerryMatch("skill-a", "base", 0.7),
        BradleyTerryMatch("skill-a", "base", 0.6),
        BradleyTerryMatch("skill-a", "skill-b", 0.55),
        BradleyTerryMatch("skill-a", "skill-c", 0.65),
    ]

    assert SelfImprovingLoop._bt_player_uncertainty("skill-a", one_match) == pytest.approx(1.0)
    assert SelfImprovingLoop._bt_player_uncertainty("skill-a", four_matches) == pytest.approx(0.5)


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


def test_codex_sdk_uses_codex_judge_provider(monkeypatch):
    from src.harness.sdk_config import set_sdk

    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig(judge_model="openai/gpt-5.4-mini")

    set_sdk("codex")
    try:
        provider, model = loop._detect_judge_provider_and_model()
    finally:
        set_sdk("claude")

    assert provider == "codex"
    assert model == "gpt-5.4-mini"


def test_codex_judge_call_uses_codex_executor(monkeypatch, tmp_path):
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig()
    loop._project_root = tmp_path
    loop._judge_prompt_tokens = 0
    loop._judge_completion_tokens = 0
    loop._judge_total_tokens = 0
    loop._judge_cost_usd = 0.0
    captured = {}

    class Usage:
        input_tokens = 11
        output_tokens = 7

    class Turn:
        usage = Usage()
        final_response = '{"match_score": 0.5}'

    async def fake_execute_query(options, query):
        captured["options"] = options
        captured["query"] = query
        return [Turn()]

    monkeypatch.setattr(
        "src.harness.codex.executor.execute_query",
        fake_execute_query,
    )

    text = asyncio.run(loop._call_judge_api("judge prompt", "codex", "gpt-5.4-mini"))

    assert text == '{"match_score": 0.5}'
    assert captured["options"]["model"] == "gpt-5.4-mini"
    assert captured["options"]["working_directory"] == str(tmp_path)
    assert captured["options"]["output_schema"]["properties"]["b_over_a_score"]["type"] == "number"
    assert captured["options"]["output_schema"]["properties"]["failure_mechanism_encoding"]["type"] == "number"
    assert captured["options"]["output_schema"]["properties"]["executable_specificity"]["type"] == "number"
    assert captured["options"]["output_schema"]["properties"]["high_risk_blacklist"]["type"] == "number"
    assert captured["options"]["output_schema"]["properties"]["generalization_transfer"]["type"] == "number"
    assert "failure_mechanism_encoding" in captured["options"]["output_schema"]["required"]
    assert "executable_specificity" in captured["options"]["output_schema"]["required"]
    assert "high_risk_blacklist" in captured["options"]["output_schema"]["required"]
    assert "generalization_transfer" in captured["options"]["output_schema"]["required"]
    assert captured["query"] == "judge prompt"
    assert loop._judge_prompt_tokens == 11
    assert loop._judge_completion_tokens == 7
    assert loop._judge_total_tokens == 18
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


def test_judge_api_usage_uses_default_openai_pricing():
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig(judge_log_details=False)
    loop._iter_cost = 0.0
    loop._judge_cost_usd = 0.0
    loop._judge_prompt_tokens = 0
    loop._judge_completion_tokens = 0
    loop._judge_total_tokens = 0

    usage = types.SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    loop._record_judge_api_usage("openai", "openai/gpt-5-nano", usage)

    assert loop._judge_cost_usd == pytest.approx(0.45)
    assert loop._iter_cost == pytest.approx(0.45)


def test_judge_api_usage_maps_versioned_nano_to_nano_pricing():
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig(judge_log_details=False)
    loop._iter_cost = 0.0
    loop._judge_cost_usd = 0.0
    loop._judge_prompt_tokens = 0
    loop._judge_completion_tokens = 0
    loop._judge_total_tokens = 0

    usage = types.SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    loop._record_judge_api_usage("openai", "openai/gpt-5.4-nano", usage)

    assert loop._judge_cost_usd == pytest.approx(0.45)
