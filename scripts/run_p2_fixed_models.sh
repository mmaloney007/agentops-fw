#!/bin/bash
# Run the 4 fixed models for P2 training
# Created: 2026-01-21 - Fixes for Yi-1.5-6B, GPT-OSS-20B, Phi-3-mini, Falcon-Mamba-7B

OUT_DIR="out/p2_training_fixed"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/full_run.log"

echo "=== P2 Fixed Models Training ===" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

# Models and their configs
MODELS="yi-1.5-6b:p2_yi_6b gpt-oss-20b:p2_gpt_oss phi-3-mini:p2_phi3_mini falcon-mamba-7b:p2_falcon_mamba"
SEEDS="42 123 456"
STEP_COUNTS="250 500"

for model_cfg in $MODELS; do
    model="${model_cfg%%:*}"
    config="${model_cfg##*:}"

    echo "" | tee -a "$LOG_FILE"
    echo "==============================================" | tee -a "$LOG_FILE"
    echo "MODEL: $model (config: $config)" | tee -a "$LOG_FILE"
    echo "==============================================" | tee -a "$LOG_FILE"

    for seed in $SEEDS; do
        for steps in $STEP_COUNTS; do
            run_name="${model}_seed${seed}_${steps}steps"
            run_out="$OUT_DIR/$run_name"

            # Skip if already completed
            if [ -f "$run_out/train_log.jsonl" ]; then
                existing_steps=$(wc -l < "$run_out/train_log.jsonl" | tr -d ' ')
                if [ "$existing_steps" -ge "$steps" ]; then
                    echo "SKIP (already done): $run_name" | tee -a "$LOG_FILE"
                    continue
                fi
            fi

            echo "RUNNING: $run_name" | tee -a "$LOG_FILE"

            python -m agent_stable_slo.train.grpo_train_loop \
                --config-preset "$config" \
                --seed "$seed" \
                --steps "$steps" \
                --out "$run_out" \
                --checkpoint-every "$steps" \
                2>&1 | tee -a "$LOG_FILE"

            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "FAILED (exit $exit_code): $run_name" | tee -a "$LOG_FILE"
                continue
            fi

            # Extract stats from log if exists
            if [ -f "$run_out/train_log.jsonl" ]; then
                lines=$(wc -l < "$run_out/train_log.jsonl" | tr -d ' ')
                last50_valid=$(tail -50 "$run_out/train_log.jsonl" | grep -c '"json_valid": 1' || true)
                last50_valid=${last50_valid:-0}
                if [ -z "$last50_valid" ] || [ "$last50_valid" = "" ]; then
                    last50_valid=0
                fi
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
