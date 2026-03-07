#!/bin/bash
# run_remaining.sh — Run the 7 experiments that need the transpose fix
set -e

cd "$(dirname "$0")/.."

STEPS=${STEPS:-100}
GROUP=${GROUP:-2}
LR=1e-5
TEMP=0.7
MAX_TOK=64
TASKS=scripts/hard_tasks.jsonl
PYTHON=/Users/maloney/.local/share/mamba/bin/python

echo "=== Rerunning 7 experiments with transpose fix ==="

# Clear any stale results
rm -f results/experiments/qwen_private/grpo_log.jsonl
rm -f results/experiments/qwen_private_full/grpo_log.jsonl
rm -f results/experiments/qwen_mlx/grpo_log.jsonl
rm -f results/experiments/smollm2_public/grpo_log.jsonl
rm -f results/experiments/smollm2_private/grpo_log.jsonl
rm -f results/experiments/smollm2_private_full/grpo_log.jsonl
rm -f results/experiments/smollm2_mlx/grpo_log.jsonl

echo "[2/8] Qwen — private (ANE forward)"
mkdir -p results/experiments/qwen_private
./grpo_private --model weights/qwen2.5-0.5b/model.safetensors \
  --tokenizer weights/qwen2.5-0.5b/tokenizer.json \
  --tasks $TASKS --config qwen05b \
  --coreml-dir models/qwen05b_coreml/ \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens $MAX_TOK --out results/experiments/qwen_private/grpo_log.jsonl 2>&1 | tail -3
echo ""

echo "[3/8] Qwen — private-full (ANE forward + backward dx)"
mkdir -p results/experiments/qwen_private_full
./grpo_private --model weights/qwen2.5-0.5b/model.safetensors \
  --tokenizer weights/qwen2.5-0.5b/tokenizer.json \
  --tasks $TASKS --config qwen05b \
  --coreml-dir models/qwen05b_coreml/ --backward-ane \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens $MAX_TOK --out results/experiments/qwen_private_full/grpo_log.jsonl 2>&1 | tail -3
echo ""

echo "[4/8] Qwen — MLX (Metal GPU)"
mkdir -p results/experiments/qwen_mlx
$PYTHON scripts/run_mlx_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --tasks $TASKS --steps $STEPS --group-size $GROUP \
  --lr $LR --temperature $TEMP --max-tokens $MAX_TOK \
  --out results/experiments/qwen_mlx/grpo_log.jsonl 2>&1 | tail -3
echo ""

echo "[5/8] SmolLM2 — public (CPU-only)"
mkdir -p results/experiments/smollm2_public
./grpo_public --model smollm2-360m \
  --weights weights/smollm2-360m/model.safetensors \
  --tokenizer weights/smollm2-360m/tokenizer.json \
  --tasks $TASKS --steps $STEPS --temperature $TEMP \
  --out-dir results/experiments/smollm2_public 2>&1 | tail -3
echo ""

echo "[6/8] SmolLM2 — private (ANE forward)"
mkdir -p results/experiments/smollm2_private
./grpo_private --model weights/smollm2-360m/model.safetensors \
  --tokenizer weights/smollm2-360m/tokenizer.json \
  --tasks $TASKS --config smollm2 \
  --coreml-dir models/smollm2_coreml/ \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens $MAX_TOK --out results/experiments/smollm2_private/grpo_log.jsonl 2>&1 | tail -3
echo ""

echo "[7/8] SmolLM2 — private-full (ANE forward + backward dx)"
mkdir -p results/experiments/smollm2_private_full
./grpo_private --model weights/smollm2-360m/model.safetensors \
  --tokenizer weights/smollm2-360m/tokenizer.json \
  --tasks $TASKS --config smollm2 \
  --coreml-dir models/smollm2_coreml/ --backward-ane \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens $MAX_TOK --out results/experiments/smollm2_private_full/grpo_log.jsonl 2>&1 | tail -3
echo ""

echo "[8/8] SmolLM2 — MLX (Metal GPU)"
mkdir -p results/experiments/smollm2_mlx
$PYTHON scripts/run_mlx_grpo.py \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --tasks $TASKS --steps $STEPS --group-size $GROUP \
  --lr $LR --temperature $TEMP --max-tokens $MAX_TOK \
  --out results/experiments/smollm2_mlx/grpo_log.jsonl 2>&1 | tail -3
echo ""

echo "=== ALL 7 REMAINING EXPERIMENTS COMPLETE ==="
ls -la results/experiments/*/grpo_log.jsonl 2>/dev/null
