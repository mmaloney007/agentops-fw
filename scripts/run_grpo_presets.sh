#!/usr/bin/env bash
# Preset GRPO/LoRA training commands for 4090 (Qwen3, GPT-OSS-20B) and Mac (MPS).
# All commands assume models live under ./models (not tracked; see .gitignore).
# Usage:
#   bash scripts/run_grpo_presets.sh 4090-qwen3
#   bash scripts/run_grpo_presets.sh 4090-gptoss
#   bash scripts/run_grpo_presets.sh mac-qwen3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Presets:
  4090-qwen3   Train Qwen3-4B-Thinking on 4090 with bf16 LoRA.
  4090-gptoss  Train GPT-OSS-20B (HF weights required) on 4090 with 4-bit LoRA.
  mac-qwen3    Train Qwen3-4B-Thinking on Mac MPS with small fp16 LoRA.

Env:
  - Assumes base mamba env: source activate_mamba.sh base
  - Models live under ./models (not committed). Place checkpoints there before running.
  - Override MODEL_DIR / OUT / STEPS / LOAD_IN_4BIT / MAX_NEW_TOKENS / GRAD_ACC via env.
  - Increase STEPS/GRAD_ACC/MAX_NEW_TOKENS to push quality; reduce for faster smoke tests.
EOF
}

if [[ "${1:-}" =~ ^(-h|--help)$ || $# -eq 0 ]]; then
  usage
  exit 0
fi

# Activate base mamba env (stays local to this script)
source "${ROOT}/activate_mamba.sh" base

MODE="${1}"

case "$MODE" in
  4090-qwen3)
    # Tune knobs here if you want more/less compute:
    # - STEPS: more steps -> better (start 2.5k, go 3-5k if time permits)
    # - MAX_NEW_TOKENS: raise to 192-256 for richer answers; lower if OOM
    # - GRAD_ACC: increase to simulate larger batch on 4090
    # - LOAD_IN_4BIT: set true if you want to try bigger bases (e.g., 7B/14B)
    MODEL_DIR=${MODEL_DIR:-./models/qwen3-4b-thinking-2507}
    OUT=${OUT:-out/grpo_qwen3_4090}
    STEPS=${STEPS:-2500}
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-192}
    GRAD_ACC=${GRAD_ACC:-2}
    LOAD_IN_4BIT=${LOAD_IN_4BIT:-false}
    python -m agent_stable_slo.train.grpo_train_loop \
      --base-model "$MODEL_DIR" \
      --tasks tasks/robust_eval_gold.jsonl \
      --out "$OUT" \
      --steps "$STEPS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --gradient-accumulation "$GRAD_ACC" \
      --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
      --temperature 0.7 --top-p 0.95 \
      --load-in-4bit "$LOAD_IN_4BIT"
    ;;
  4090-gptoss)
    # GPT-OSS-20B requires local HF weights at ./models/gpt-oss-20b (not tracked).
    # Keep 4-bit on; lower MAX_NEW_TOKENS if OOM; increase STEPS for better convergence.
    MODEL_DIR=${MODEL_DIR:-./models/gpt-oss-20b}
    OUT=${OUT:-out/grpo_gptoss20b_4090}
    STEPS=${STEPS:-1500}
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-160}
    GRAD_ACC=${GRAD_ACC:-2}
    LOAD_IN_4BIT=${LOAD_IN_4BIT:-true}
    python -m agent_stable_slo.train.grpo_train_loop \
      --base-model "$MODEL_DIR" \
      --tasks tasks/robust_eval_gold.jsonl \
      --out "$OUT" \
      --steps "$STEPS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --gradient-accumulation "$GRAD_ACC" \
      --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
      --temperature 0.7 --top-p 0.9 \
      --load-in-4bit "$LOAD_IN_4BIT"
    ;;
  mac-qwen3)
    # Mac/MPS constraints: no bitsandbytes; keep batch=1; short sequences to avoid OOM.
    # You can raise MAX_NEW_TOKENS modestly (e.g., 112-128) if memory allows.
    MODEL_DIR=${MODEL_DIR:-Qwen/Qwen3-4B-Thinking-2507}
    OUT=${OUT:-out/grpo_qwen3_mac}
    STEPS=${STEPS:-1200}
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
    python -m agent_stable_slo.train.grpo_train_loop \
      --base-model "$MODEL_DIR" \
      --tasks tasks/robust_eval_gold.jsonl \
      --out "$OUT" \
      --steps "$STEPS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --gradient-accumulation 1 \
      --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
      --temperature 0.7 --top-p 0.95 \
      --load-in-4bit false \
      --torch-dtype float16
    ;;
  *)
    usage
    exit 1
    ;;
esac
