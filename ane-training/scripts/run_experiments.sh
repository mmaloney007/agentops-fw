#!/bin/bash
# run_experiments.sh — Run all 4-path × 2-model experiments
# Default: 500 steps, group=2
set -e

cd "$(dirname "$0")/.."

STEPS=${STEPS:-500}
GROUP=${GROUP:-2}
LR=1e-5
TEMP=0.7
MAX_TOK=64
TASKS=scripts/hard_tasks.jsonl
PYTHON=/Users/maloney/.local/share/mamba/bin/python

# Qwen paths
QWEN_WEIGHTS=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors
QWEN_TOKENIZER=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json

mkdir -p results/experiments

echo "=== Paper 9 Experiments: 4 paths x 2 models x ${STEPS} steps ==="
echo "Group size: ${GROUP}, LR: ${LR}, Temperature: ${TEMP}, Max tokens: ${MAX_TOK}"
echo ""

# --- Qwen2.5-0.5B ---

echo "[1/8] Qwen — public (CPU-only)"
rm -f results/experiments/qwen_public/grpo_log.jsonl
mkdir -p results/experiments/qwen_public
./grpo_public --model qwen2.5-0.5b \
  --weights "$QWEN_WEIGHTS" \
  --tokenizer "$QWEN_TOKENIZER" \
  --tasks $TASKS --steps $STEPS --temperature $TEMP \
  --group-size $GROUP --lr $LR --max-tokens $MAX_TOK \
  --out-dir results/experiments/qwen_public 2>&1 | tail -5
echo ""

echo "[2/8] Qwen — private (ANE forward)"
rm -f results/experiments/qwen_private/grpo_log.jsonl
mkdir -p results/experiments/qwen_private
./grpo_private --model "$QWEN_WEIGHTS" \
  --tokenizer "$QWEN_TOKENIZER" \
  --tasks $TASKS --config qwen05b \
  --coreml-dir models/qwen05b_coreml/ \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens $MAX_TOK --out results/experiments/qwen_private/grpo_log.jsonl 2>&1 | tail -5
echo ""

echo "[3/8] Qwen — private-full (ANE forward + backward dx)"
rm -f results/experiments/qwen_private_full/grpo_log.jsonl
mkdir -p results/experiments/qwen_private_full
./grpo_private --model "$QWEN_WEIGHTS" \
  --tokenizer "$QWEN_TOKENIZER" \
  --tasks $TASKS --config qwen05b \
  --coreml-dir models/qwen05b_coreml/ --backward-ane \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens $MAX_TOK --out results/experiments/qwen_private_full/grpo_log.jsonl 2>&1 | tail -5
echo ""

echo "[4/8] Qwen — MLX (Metal GPU)"
rm -f results/experiments/qwen_mlx/grpo_log.jsonl
mkdir -p results/experiments/qwen_mlx
$PYTHON scripts/run_mlx_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --tasks $TASKS --steps $STEPS --group-size $GROUP \
  --lr $LR --temperature $TEMP --max-tokens $MAX_TOK \
  --out results/experiments/qwen_mlx/grpo_log.jsonl 2>&1 | tail -5
echo ""

# --- SmolLM2-360M ---

echo "[5/8] SmolLM2 — public (CPU-only)"
rm -f results/experiments/smollm2_public/grpo_log.jsonl
mkdir -p results/experiments/smollm2_public
./grpo_public --model smollm2-360m \
  --weights weights/smollm2-360m/model.safetensors \
  --tokenizer weights/smollm2-360m/tokenizer.json \
  --tasks $TASKS --steps $STEPS --temperature $TEMP \
  --group-size $GROUP --lr $LR --max-tokens $MAX_TOK \
  --out-dir results/experiments/smollm2_public 2>&1 | tail -5
echo ""

echo "[6/8] SmolLM2 — private (ANE forward)"
rm -f results/experiments/smollm2_private/grpo_log.jsonl
mkdir -p results/experiments/smollm2_private
./grpo_private --model weights/smollm2-360m/model.safetensors \
  --tokenizer weights/smollm2-360m/tokenizer.json \
  --tasks $TASKS --config smollm2 \
  --coreml-dir models/smollm2_coreml/ \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens $MAX_TOK --out results/experiments/smollm2_private/grpo_log.jsonl 2>&1 | tail -5
echo ""

echo "[7/8] SmolLM2 — private-full (ANE forward + backward dx)"
rm -f results/experiments/smollm2_private_full/grpo_log.jsonl
mkdir -p results/experiments/smollm2_private_full
./grpo_private --model weights/smollm2-360m/model.safetensors \
  --tokenizer weights/smollm2-360m/tokenizer.json \
  --tasks $TASKS --config smollm2 \
  --coreml-dir models/smollm2_coreml/ --backward-ane \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens $MAX_TOK --out results/experiments/smollm2_private_full/grpo_log.jsonl 2>&1 | tail -5
echo ""

echo "[8/8] SmolLM2 — MLX (Metal GPU)"
rm -f results/experiments/smollm2_mlx/grpo_log.jsonl
mkdir -p results/experiments/smollm2_mlx
$PYTHON scripts/run_mlx_grpo.py \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --tasks $TASKS --steps $STEPS --group-size $GROUP \
  --lr $LR --temperature $TEMP --max-tokens $MAX_TOK \
  --out results/experiments/smollm2_mlx/grpo_log.jsonl 2>&1 | tail -5
echo ""

echo "=== ALL 8 EXPERIMENTS COMPLETE ==="
echo "Results in: results/experiments/"
ls -la results/experiments/*/grpo_log.jsonl 2>/dev/null
