#!/usr/bin/env bash
# Run the gold suite for a given provider/mode and output dir.
# Usage: scripts/run_mode_suite.sh provider mode outdir model base
set -euo pipefail
PROV=${1:-lmstudio}
MODE=${2:-structured}
OUT=${3:-out/tmp_run}
MODEL=${4:-qwen/qwen3-4b-thinking-2507}
BASE=${5:-http://10.0.0.63:1234/v1}

export AOFW_PROVIDER=$PROV
export DECODE_MODE=$MODE
export OPENAI_API_BASE=$BASE
export OPENAI_API_KEY=lm-studio
export LMSTUDIO_MODEL=$MODEL
export MAX_THOUGHT_TOKENS=${MAX_THOUGHT_TOKENS:-196}
export WANDB_PROJECT=${WANDB_PROJECT:-agent-stable-slo}
export WANDB_ENTITY=${WANDB_ENTITY:-mike007}
export WANDB_MODE=${WANDB_MODE:-online}

python3 -m agent_stable_slo.train.grpo_trl --base-model "$MODEL" --tasks tasks/robust_eval_gold.jsonl --out "$OUT" --steps 540 --max-new-tokens 196
