#!/usr/bin/env bash
# Run EvoSkill evolver on SciBench chemistry trajectories.
# Uses pre-collected trajectories + LLM judge mode (no agent re-inference).
#
# Usage:
#   nohup bash scripts/run_scibench_evolver.sh > examples/scibench/.evoskill/evolver.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

set -a
source examples/officeqa/.evoskill/.env.volcengine.local
set +a

# Use deepseek-v4-pro for skill evolution, flash for subagents
export ANTHROPIC_MODEL="deepseek-v4-pro"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-pro"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-pro"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"

export EVOSKILL_NO_GIT=1
export PATH="$(pwd)/.venv/bin:$PATH"
unset CLAUDECODE

exec .venv/bin/python3 scripts/run_loop.py \
    --sdk claude \
    --model deepseek-v4-pro \
    --project_root examples/scibench \
    --trajectories_dir examples/scibench/.evoskill/trajectories/baseline-deepseek-flash \
    --use_llm_judge true \
    --judge_model deepseek-v4-pro \
    --judge_scoring bradley_terry \
    --judge_holdout_ratio 0.2 \
    --judge_distinct_from_generator false \
    --mode skill_only \
    --max_iterations 20 \
    --frontier_size 5 \
    --no_improvement_limit 5 \
    --concurrency 1 \
    --failure_samples 4 \
    --scibench_scorer true \
    --enable_multi_skill true \
    --continue_loop false
