"""Tests for the information-isolated skill verifier (CoEvoSkills)."""

import asyncio
import json
from pathlib import Path

from src.loop.runner import SelfImprovingLoop
from src.loop.config import LoopConfig
from src.schemas import SkillVerifierResponse
from src.schemas.tool_generator import SkillTest


class _FakeTrace:
    def __init__(self, output):
        self.output = output
        self.total_cost_usd = 0.0


class _FakeVerifier:
    def __init__(self, response):
        self._response = response
        self.calls = 0

    async def run(self, query):
        self.calls += 1
        self._last_query = query
        return _FakeTrace(self._response)


def _write_skill(root: Path, name: str, body: str):
    d = root / ".claude" / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body)
    # generator-written (gameable) tests that the verifier should overwrite
    (d / "SKILL_TESTS.json").write_text('[{"name": "author_test"}]')
    return d


class _FakeManager:
    def __init__(self):
        self.commits = 0

    def commit(self, msg):
        self.commits += 1


def _loop(tmp_path, *, verifier, use=True):
    loop = object.__new__(SelfImprovingLoop)
    loop.config = LoopConfig(use_skill_verifier=use, skill_verifier_max_tests=4)
    loop._project_root = tmp_path
    loop._meta_path = tmp_path / ".evoskill" / "meta.md"
    loop.task_constraints = "treasury QA"
    loop._add_iteration_cost = lambda *a, **k: None
    loop._verifier_tests_cache = {}
    loop.manager = _FakeManager()

    class _Agents:
        skill_verifier = verifier

    loop.agents = _Agents()
    return loop


def test_verifier_overwrites_author_tests(tmp_path):
    d = _write_skill(tmp_path, "skA", "# A\nWhen To Use: ...")
    resp = SkillVerifierResponse(
        tests=[
            SkillTest(name="pos", scenario="s", should_activate=True, expected_behavior="acts"),
            SkillTest(name="neg", scenario="s2", should_activate=False, expected_behavior="must not over-trigger"),
        ],
        probe_reasoning="probed over-trigger",
    )
    verifier = _FakeVerifier(resp)
    loop = _loop(tmp_path, verifier=verifier)

    total = asyncio.run(loop._author_verifier_tests(["skA"]))
    assert total == 2
    assert verifier.calls == 1
    written = json.loads((d / "SKILL_TESTS.json").read_text())
    assert {t["name"] for t in written} == {"pos", "neg"}  # author tests replaced
    # the verifier query carries the skill + domain but no ground-truth answers
    assert "treasury QA" in verifier._last_query


def test_verifier_disabled_is_noop(tmp_path):
    d = _write_skill(tmp_path, "skB", "# B\n")
    loop = _loop(tmp_path, verifier=_FakeVerifier(SkillVerifierResponse(tests=[])), use=False)
    total = asyncio.run(loop._author_verifier_tests(["skB"]))
    assert total == 0
    assert (d / "SKILL_TESTS.json").read_text() == '[{"name": "author_test"}]'  # untouched


def test_verifier_none_agent_is_noop(tmp_path):
    _write_skill(tmp_path, "skC", "# C\n")
    loop = _loop(tmp_path, verifier=None, use=True)
    assert asyncio.run(loop._author_verifier_tests(["skC"])) == 0


def test_verifier_caches_by_skill_content(tmp_path):
    _write_skill(tmp_path, "skE", "# E\nstable content")
    verifier = _FakeVerifier(
        SkillVerifierResponse(tests=[SkillTest(name="x", scenario="s", should_activate=True, expected_behavior="b")])
    )
    loop = _loop(tmp_path, verifier=verifier)
    asyncio.run(loop._author_verifier_tests(["skE"]))
    asyncio.run(loop._author_verifier_tests(["skE"]))  # identical content
    assert verifier.calls == 1  # second served from cache, no new LLM call


def test_verifier_commits_written_tests(tmp_path):
    _write_skill(tmp_path, "skF", "# F\n")
    loop = _loop(
        tmp_path,
        verifier=_FakeVerifier(
            SkillVerifierResponse(tests=[SkillTest(name="x", scenario="s", should_activate=False, expected_behavior="b")])
        ),
    )
    asyncio.run(loop._author_verifier_tests(["skF"]))
    assert loop.manager.commits == 1  # persisted to the child branch


def test_verifier_caps_tests(tmp_path):
    d = _write_skill(tmp_path, "skD", "# D\n")
    many = [SkillTest(name=f"t{i}", scenario="s", should_activate=bool(i % 2), expected_behavior="b") for i in range(10)]
    loop = _loop(tmp_path, verifier=_FakeVerifier(SkillVerifierResponse(tests=many)))
    total = asyncio.run(loop._author_verifier_tests(["skD"]))
    assert total == 4  # skill_verifier_max_tests
    assert len(json.loads((d / "SKILL_TESTS.json").read_text())) == 4
