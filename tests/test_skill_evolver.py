import json


def test_skill_grouper_groups_by_skill_and_errors(tmp_path):
    from src.skill_evolver.grouper import SkillGrouper

    path = tmp_path / "trajectories.jsonl"
    records = [
        {
            "task_id": "t1",
            "skills_used": ["numeric"],
            "final_success": False,
            "error_log": ["TYPE_MISMATCH: bad number"],
            "steps": [{"success": False, "failure_reason": "TYPE_MISMATCH: bad number"}],
        },
        {
            "task_id": "t2",
            "skills_used": ["numeric"],
            "final_success": True,
            "steps": [{"success": True, "action": "ok"}],
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    groups = SkillGrouper(path).group_by_skill()

    assert len(groups["numeric"]["failures"]) == 1
    assert len(groups["numeric"]["successes"]) == 1
    assert groups["numeric"]["error_patterns"]["TYPE_MISMATCH"] == 1


def test_pareto_gate_rejects_invalid_and_accepts_valid_content():
    from src.skill_evolver.evolver import SkillProposal
    from src.skill_evolver.pareto_gate import ParetoGate

    gate = ParetoGate(baseline_score=0.5, min_score_delta=0.05)
    invalid = SkillProposal("refine", "numeric", "too short", "reason", 0.5)
    assert not gate.evaluate(invalid, 0.8).accepted

    valid_content = """---
name: numeric
trigger_conditions:
  - "numeric extraction"
---

## Procedure
1. Read the source carefully.
2. Preserve units and signs.

## Failure Patterns
- Retry when values are missing.
"""
    valid = SkillProposal("refine", "numeric", valid_content, "reason", 0.8)
    assert gate.evaluate(valid, 0.8).accepted


def test_evaluator_static_quality_scores_skill_content(tmp_path):
    from src.evaluation.evaluator import SkillLearnBenchEvaluator

    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "tasks.json").write_text("[]")
    evaluator = SkillLearnBenchEvaluator(bench)

    content = """---
name: numeric
trigger_conditions:
  - "office finance numeric task"
---

## Procedure
1. Extract the requested value.

## Examples
Input: office finance question
Output: value

## Failure Patterns
- Missing unit.
"""
    score = evaluator.evaluate_skill_quality(content, {"domain": "office finance"})

    assert score == 1.0
