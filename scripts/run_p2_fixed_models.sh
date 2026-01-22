#!/bin/bash
# Run the 4 fixed models for P2 training
# Created: 2026-01-21 - Fixes for Yi-1.5-6B, GPT-OSS-20B, Phi-3-mini, Falcon-Mamba-7B

set -e

OUT_DIR="out/p2_training_fixed"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/full_run.log"

echo "=== P2 Fixed Models Training ===" | tee "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

# Models and their configs
declare -A MODELS=(
    ["yi-1.5-6b"]="p2_yi_6b"
    ["gpt-oss-20b"]="p2_gpt_oss"
    ["phi-3-mini"]="p2_phi3_mini"
    ["falcon-mamba-7b"]="p2_falcon_mamba"
)

SEEDS=(42 123 456)
STEP_COUNTS=(250 500)

for model in "${!MODELS[@]}"; do
    config="${MODELS[$model]}"

    echo "" | tee -a "$LOG_FILE"
    echo "==============================================" | tee -a "$LOG_FILE"
    echo "MODEL: $model (config: $config)" | tee -a "$LOG_FILE"
    echo "==============================================" | tee -a "$LOG_FILE"

    for seed in "${SEEDS[@]}"; do
        for steps in "${STEP_COUNTS[@]}"; do
            run_name="${model}_seed${seed}_${steps}steps"
            run_out="$OUT_DIR/$run_name"

            echo "RUNNING: $run_name" | tee -a "$LOG_FILE"

            python -m agent_stable_slo.train.grpo_train_loop \
                --config-preset "$config" \
                --seed "$seed" \
                --steps "$steps" \
                --out "$run_out" \
                --checkpoint-every "$steps" \
                2>&1 | tee -a "$LOG_FILE" || {
                    echo "FAILED: $run_name" | tee -a "$LOG_FILE"
                    continue
                }

            # Extract stats from log if exists
            if [ -f "$run_out/train_log.jsonl" ]; then
                lines=$(wc -l < "$run_out/train_log.jsonl" | tr -d ' ')
                last50_valid=$(tail -50 "$run_out/train_log.jsonl" | grep -c '"json_valid": 1' 2>/dev/null || echo "0")
                last50_valid=${last50_valid:-0}
                last50_pct=$((last50_valid * 2))
                echo "  Steps: $lines, Last-50: ${last50_pct}%" | tee -a "$LOG_FILE"
            fi

            echo "DONE: $run_name" | tee -a "$LOG_FILE"
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== ALL FIXED MODELS COMPLETE ===" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
