#!/usr/bin/env bash
set -euo pipefail

: "${OFFICEQA_OUT:?OFFICEQA_OUT is required}"
: "${OFFICEQA_LOG:?OFFICEQA_LOG is required}"

export EVOSKILL_NO_GIT="${EVOSKILL_NO_GIT:-1}"
export OPENHANDS_SUPPRESS_BANNER="${OPENHANDS_SUPPRESS_BANNER:-1}"
export OFFICEQA_MODEL="${OFFICEQA_MODEL:-openai/gpt-5.4-nano}"
export OFFICEQA_CONCURRENCY="${OFFICEQA_CONCURRENCY:-2}"
export OFFICEQA_AGENT_MAX_RETRIES="${OFFICEQA_AGENT_MAX_RETRIES:-2}"
export OFFICEQA_AGENT_TIMEOUT_SECONDS="${OFFICEQA_AGENT_TIMEOUT_SECONDS:-900}"
export OFFICEQA_TASK_TIMEOUT_SECONDS="${OFFICEQA_TASK_TIMEOUT_SECONDS:-1200}"
export OFFICEQA_PROJECT_ROOT="${OFFICEQA_PROJECT_ROOT:-examples/officeqa}"

mkdir -p "$(dirname "$OFFICEQA_LOG")" "$OFFICEQA_OUT"

exec .venv/bin/python -u scripts/collect_trajectories.py \
  --sdk openhands \
  --model "$OFFICEQA_MODEL" \
  --dataset examples/officeqa/data/officeqa_full/officeqa_full_with_source_prompts.csv \
  --question_column question \
  --ground_truth_column answer \
  --category_column difficulty \
  --uid_column uid \
  --output_dir "$OFFICEQA_OUT" \
  --concurrency "$OFFICEQA_CONCURRENCY" \
  --project_root "$OFFICEQA_PROJECT_ROOT" \
  --data_dir examples/officeqa/data/officeqa_full/treasury_bulletins \
  --task_prompt_file examples/officeqa/.evoskill/task.md \
  --resume false \
  --agent_timeout_seconds "$OFFICEQA_AGENT_TIMEOUT_SECONDS" \
  --agent_max_retries "$OFFICEQA_AGENT_MAX_RETRIES" \
  --task_timeout_seconds "$OFFICEQA_TASK_TIMEOUT_SECONDS" \
  > "$OFFICEQA_LOG" 2>&1
