#!/usr/bin/env bash
# Run SciSkillBench baseline trajectory collection.
# Must be run from a screen/tmux session (NOT inside Claude Code).
#
# Usage:
#   screen -S scisci
#   bash scripts/run_sciskillbench_baseline.sh
#
# Or with nohup:
#   nohup bash scripts/run_sciskillbench_baseline.sh > examples/sciskillbench/.evoskill/collect.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

set -a
source examples/officeqa/.evoskill/.env.deepseek.local
set +a

export EVOSKILL_NO_GIT=1
export PATH="$(pwd)/.venv/bin:/home/admin/opt/lammps/lammps-static/bin:/home/admin/opt/xtb/xtb-dist/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib64/mpich/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XTBHOME="/home/admin/opt/xtb/xtb-dist"
export MP_API_KEY=VrB0VY1pi219m00EeBV3Puv1biEPX0j1
unset CLAUDECODE  # Allow nested Claude Code if needed

exec .venv/bin/python3 scripts/collect_sciskillbench_baseline.py \
    --output_dir examples/sciskillbench/.evoskill/trajectories/baseline-deepseek-flash \
    --level 1 \
    --concurrency 1
