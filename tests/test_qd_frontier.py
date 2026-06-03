"""Tests for the quality-diversity frontier (Phase 2)."""

from __future__ import annotations

from src.registry.manager import ProgramManager
from src.registry.models import ProgramConfig


def _make_manager(monkeypatch, tmp_path):
    """A ProgramManager with git/config side effects stubbed and an in-memory
    frontier store keyed by program name -> (score, vector)."""
    mgr = ProgramManager(cwd=tmp_path)
    store: dict[str, tuple[float, dict]] = {}
    frontier: set[str] = set()

    monkeypatch.setattr(mgr, "_git_current_branch", lambda: "program/base")
    monkeypatch.setattr(mgr, "_git_checkout", lambda branch: None)
    monkeypatch.setattr(
        mgr, "_read_config", lambda: ProgramConfig(name="x", system_prompt={})
    )
    monkeypatch.setattr(mgr, "_write_config", lambda config: None)
    monkeypatch.setattr(mgr, "_git_add", lambda path: None)
    monkeypatch.setattr(mgr, "_git_commit", lambda msg: None)

    monkeypatch.setattr(mgr, "get_frontier", lambda: list(frontier))
    monkeypatch.setattr(
        mgr,
        "get_frontier_with_vectors",
        lambda: [(n, store[n][0], store[n][1]) for n in frontier],
    )
    monkeypatch.setattr(mgr, "mark_frontier", lambda n: frontier.add(n))
    monkeypatch.setattr(mgr, "unmark_frontier", lambda n: frontier.discard(n))

    # Intercept the persisted score+vector so the elite recompute can see it.
    orig = mgr.update_frontier_qd

    def patched(name, score, vector):
        store[name] = (score, dict(vector))
        return orig(name, score, vector)

    monkeypatch.setattr(mgr, "update_frontier_qd", patched)
    return mgr, frontier, store


def test_qd_keeps_category_specialist(monkeypatch, tmp_path):
    mgr, frontier, _ = _make_manager(monkeypatch, tmp_path)

    # P1: strong on 'tax', weak on 'unit'. Higher average.
    assert mgr.update_frontier_qd("p1", 0.7, {"tax": 0.9, "unit": 0.5})
    # P2: lower average but the best on 'unit' -> must survive as a niche elite.
    assert mgr.update_frontier_qd("p2", 0.6, {"tax": 0.4, "unit": 0.95})

    assert frontier == {"p1", "p2"}, "category specialist should not be crowded out"


def test_qd_evicts_dominated_program(monkeypatch, tmp_path):
    mgr, frontier, _ = _make_manager(monkeypatch, tmp_path)
    mgr.update_frontier_qd("p1", 0.5, {"tax": 0.5, "unit": 0.5})
    # p2 dominates p1 on every niche and on average -> p1 is nobody's elite.
    added = mgr.update_frontier_qd("p2", 0.8, {"tax": 0.9, "unit": 0.9})
    assert added is True
    assert frontier == {"p2"}


def test_qd_rejected_when_not_elite(monkeypatch, tmp_path):
    mgr, frontier, _ = _make_manager(monkeypatch, tmp_path)
    mgr.update_frontier_qd("p1", 0.9, {"tax": 0.9, "unit": 0.9})
    # p2 is worse everywhere -> not added.
    added = mgr.update_frontier_qd("p2", 0.3, {"tax": 0.2, "unit": 0.2})
    assert added is False
    assert frontier == {"p1"}


def test_niche_selection_rotates_over_elites(monkeypatch, tmp_path):
    mgr, frontier, store = _make_manager(monkeypatch, tmp_path)
    store["p1"] = (0.7, {"tax": 0.9, "unit": 0.4})
    store["p2"] = (0.6, {"tax": 0.4, "unit": 0.95})
    frontier.update({"p1", "p2"})
    # Scalar scored frontier needed by the fallback path.
    monkeypatch.setattr(
        mgr, "get_frontier_with_scores", lambda: [("p1", 0.7), ("p2", 0.6)]
    )
    picks = {mgr.select_from_frontier("niche", iteration=i) for i in range(4)}
    assert picks == {"p1", "p2"}  # both niche elites are reachable
