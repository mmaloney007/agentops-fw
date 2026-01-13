#!/usr/bin/env bash
# P2 Training: All 4 models at 250 and 500 steps
# Author: Mike Maloney <mike.maloney@unh.edu>
#
# This script runs sequential training for Paper 2 experiments:
# - Qwen3-4B
# - Ministral-3B
# - Gemma-3-12B
# - GPT-OSS-20B
#
# Each model is trained at both 250 and 500 steps to study
# the effect of training duration on SLO-aware reward optimization.
#
# Usage: bash scripts/run_p2_all_models.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="out/p2_training_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================="
echo "P2 Training Suite - 4 Models x 2 Step Counts"
echo "Started: $(date)"
echo "Output: $LOG_DIR"
echo "================================================="

# Common training parameters
TASK_FILE="tasks/clinc_en.jsonl"
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
TEMPERATURE=0.7
TOP_P=0.95
MAX_NEW_TOKENS=64
GRAD_ACC=2

run_training() {
    local MODEL_PATH="$1"
    local MODEL_NAME="$2"
    local STEPS="$3"
    local EXTRA_ARGS="${4:-}"

    local OUT_DIR="out/p2_${MODEL_NAME}_${STEPS}"
    local LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${STEPS}.log"

    echo ""
    echo "--- Training: $MODEL_NAME @ $STEPS steps ---"
    echo "Model: $MODEL_PATH"
    echo "Output: $OUT_DIR"
    echo "Log: $LOG_FILE"
    echo "Started: $(date)"

    python -m agent_stable_slo.train.grpo_train_loop \
        --base-model "$MODEL_PATH" \
        --tasks "$TASK_FILE" \
        --out "$OUT_DIR" \
        --steps "$STEPS" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --gradient-accumulation "$GRAD_ACC" \
        --lora-rank "$LORA_RANK" \
        --lora-alpha "$LORA_ALPHA" \
        --lora-dropout "$LORA_DROPOUT" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --checkpoint-every 50 \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"

    echo "Completed: $(date)"
    echo "Checkpoint saved to: $OUT_DIR"
}

# Model paths
QWEN3_PATH="./models/qwen3-4b"
MINISTRAL_PATH="./models/ministral-3b-instruct"
GEMMA3_PATH="./models/gemma-3-12b-it"
GPTOSS_PATH="./models/gpt-oss-20b"

echo ""
echo "=== Phase 1: Qwen3-4B ==="
run_training "$QWEN3_PATH" "qwen3_4b" 250
run_training "$QWEN3_PATH" "qwen3_4b" 500

echo ""
echo "=== Phase 2: Ministral-3B ==="
run_training "$MINISTRAL_PATH" "ministral_3b" 250
run_training "$MINISTRAL_PATH" "ministral_3b" 500

echo ""
echo "=== Phase 3: Gemma-3-12B ==="
run_training "$GEMMA3_PATH" "gemma3_12b" 250 "--load-in-4bit true"
run_training "$GEMMA3_PATH" "gemma3_12b" 500 "--load-in-4bit true"

echo ""
echo "=== Phase 4: GPT-OSS-20B ==="
run_training "$GPTOSS_PATH" "gptoss_20b" 250 "--load-in-4bit true"
run_training "$GPTOSS_PATH" "gptoss_20b" 500 "--load-in-4bit true"

echo ""
echo "================================================="
echo "P2 Training Suite COMPLETE"
echo "Finished: $(date)"
echo "All outputs in: $LOG_DIR"
echo "================================================="

# Generate summary
cat > "${LOG_DIR}/summary.md" << EOF
# P2 Training Summary

**Date:** $(date)
**Author:** Mike Maloney <mike.maloney@unh.edu>

## Models Trained

| Model | 250 Steps | 500 Steps |
|-------|-----------|-----------|
| Qwen3-4B | out/p2_qwen3_4b_250 | out/p2_qwen3_4b_500 |
| Ministral-3B | out/p2_ministral_3b_250 | out/p2_ministral_3b_500 |
| Gemma-3-12B | out/p2_gemma3_12b_250 | out/p2_gemma3_12b_500 |
| GPT-OSS-20B | out/p2_gptoss_20b_250 | out/p2_gptoss_20b_500 |

## Configuration

- Task: CLINC150 intent classification
- LoRA rank: $LORA_RANK, alpha: $LORA_ALPHA
- Temperature: $TEMPERATURE, top-p: $TOP_P
- Max tokens: $MAX_NEW_TOKENS
- Gradient accumulation: $GRAD_ACC

## Next Steps

1. Run evaluation on each checkpoint
2. Generate comparison tables
3. Update Paper 2 with results
EOF

echo "Summary written to: ${LOG_DIR}/summary.md"
