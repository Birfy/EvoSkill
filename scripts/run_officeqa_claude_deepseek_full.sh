#!/usr/bin/env bash
set -euo pipefail

: "${OFFICEQA_OUT:?OFFICEQA_OUT is required}"
: "${OFFICEQA_LOG:?OFFICEQA_LOG is required}"

export ANTHROPIC_API_KEY="${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_API_KEY:-}}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.deepseek.com/anthropic}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-deepseek-v4-pro[1m]}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${ANTHROPIC_DEFAULT_OPUS_MODEL:-deepseek-v4-pro[1m]}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${ANTHROPIC_DEFAULT_SONNET_MODEL:-deepseek-v4-pro[1m]}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${ANTHROPIC_DEFAULT_HAIKU_MODEL:-deepseek-v4-flash}"
export CLAUDE_CODE_SUBAGENT_MODEL="${CLAUDE_CODE_SUBAGENT_MODEL:-deepseek-v4-flash}"
export CLAUDE_CODE_EFFORT_LEVEL="${CLAUDE_CODE_EFFORT_LEVEL:-max}"
export EVOSKILL_NO_GIT="${EVOSKILL_NO_GIT:-1}"
export OFFICEQA_CONCURRENCY="${OFFICEQA_CONCURRENCY:-2}"
export OFFICEQA_AGENT_MAX_RETRIES="${OFFICEQA_AGENT_MAX_RETRIES:-1}"

mkdir -p "$(dirname "$OFFICEQA_LOG")" "$OFFICEQA_OUT"

exec .venv/bin/python -u scripts/collect_trajectories.py \
  --sdk claude \
  --model "deepseek-v4-pro[1m]" \
  --dataset examples/officeqa/data/officeqa_full/officeqa_full_with_source_prompts.csv \
  --question_column question \
  --ground_truth_column answer \
  --category_column difficulty \
  --uid_column uid \
  --output_dir "$OFFICEQA_OUT" \
  --concurrency "$OFFICEQA_CONCURRENCY" \
  --project_root examples/officeqa \
  --data_dir examples/officeqa/data/officeqa_full/treasury_bulletins \
  --task_prompt_file examples/officeqa/.evoskill/task.md \
  --resume false \
  --agent_timeout_seconds 900 \
  --agent_max_retries "$OFFICEQA_AGENT_MAX_RETRIES" \
  --task_timeout_seconds 1200 \
  > "$OFFICEQA_LOG" 2>&1
