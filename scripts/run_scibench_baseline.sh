#!/usr/bin/env bash
# Run SciBench chemistry baseline trajectory collection.
# Usage:
#   nohup bash scripts/run_scibench_baseline.sh > examples/scibench/.evoskill/collect.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

set -a
source examples/officeqa/.evoskill/.env.volcengine.local
set +a

export EVOSKILL_NO_GIT=1
export PATH="$(pwd)/.venv/bin:$PATH"
unset CLAUDECODE

exec .venv/bin/python3 scripts/collect_scibench_baseline.py \
    --output_dir examples/scibench/.evoskill/trajectories/baseline-deepseek-flash \
    --concurrency 4 \
    --model deepseek-v4-flash
