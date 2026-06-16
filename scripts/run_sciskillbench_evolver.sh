#!/usr/bin/env bash
# Run EvoSkill evolver on SciSkillBench level-1 trajectories.
# Uses pre-collected trajectories + LLM judge mode (no agent re-inference).
#
# Usage:
#   nohup bash scripts/run_sciskillbench_evolver.sh > examples/sciskillbench/.evoskill/evolver.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

set -a
source examples/officeqa/.evoskill/.env.deepseek.local
set +a

# Override models: use deepseek-v4-pro for skill evolution
export ANTHROPIC_MODEL="deepseek-v4-pro"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-pro"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-pro"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"

export EVOSKILL_NO_GIT=1
export PATH="$(pwd)/.venv/bin:/home/admin/opt/lammps/lammps-static/bin:/home/admin/opt/xtb/xtb-dist/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib64/mpich/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XTBHOME="/home/admin/opt/xtb/xtb-dist"
export MP_API_KEY=VrB0VY1pi219m00EeBV3Puv1biEPX0j1
unset CLAUDECODE

exec .venv/bin/python3 scripts/run_loop.py \
    --sdk claude \
    --model deepseek-v4-pro \
    --project_root examples/sciskillbench \
    --trajectories_dir examples/sciskillbench/.evoskill/trajectories/baseline-deepseek-flash-level1-nomp \
    --use_llm_judge true \
    --judge_model deepseek-v4-pro \
    --judge_scoring bradley_terry \
    --judge_holdout_ratio 0.2 \
    --judge_distinct_from_generator false \
    --mode skill_only \
    --max_iterations 10 \
    --frontier_size 3 \
    --no_improvement_limit 3 \
    --concurrency 1 \
    --failure_samples 4 \
    --tolerance_csv examples/sciskillbench/.evoskill/tasks_no_orca.csv \
    --continue_loop false
