#!/bin/bash
# P1 Full Evaluation Suite - 4 models × 6 modes = 24 runs
# Author: Mike Maloney (mike.maloney@unh.edu)
# Using 50 examples per task for faster evaluation
# Estimated runtime: ~1.5 hours per run = ~36 hours total

set -e

PYTHON="/Users/maloney/.local/share/mamba/bin/python"
CRITERIA="configs/criteria/p1_core_public_v2.yaml"
ENDPOINT="http://localhost:1234/v1"
OUT_DIR="out/p1_full_eval"

# Skip artifact uploads for faster runs
export WANDB_SKIP_ARTIFACTS=1

MODELS=(
    "qwen/qwen3-vl-4b"
    "openai/gpt-oss-20b"
    "google/gemma-3-12b"
    "mistralai/ministral-3-3b"
)

MODES=(
    "UNCONSTRAINED"
    "PROVIDER_STRUCTURED"
    "PROVIDER_STRUCTURED_PLUS_VALIDATE"
    "SPEC_DRIVEN"
    "SPEC_DRIVEN_PLUS_REPAIR"
    "SPEC_DRIVEN_PLUS_SELFCONSISTENCY"
)

echo "=========================================="
echo "P1 Full Evaluation Suite"
echo "Start time: $(date)"
echo "Models: ${#MODELS[@]}"
echo "Modes: ${#MODES[@]}"
echo "Total runs: $((${#MODELS[@]} * ${#MODES[@]}))"
echo "=========================================="

run_count=0
total_runs=$((${#MODELS[@]} * ${#MODES[@]}))

for model in "${MODELS[@]}"; do
    for mode in "${MODES[@]}"; do
        run_count=$((run_count + 1))

        model_safe=$(echo "$model" | tr '/' '_')
        out_path="$OUT_DIR/p1_core_public_v2/${model_safe}/${mode}"

        # Skip if already completed
        if [ -f "$out_path/summary.json" ]; then
            echo "[$run_count/$total_runs] SKIP: $model $mode (already complete)"
            continue
        fi

        echo ""
        echo "=========================================="
        echo "[$run_count/$total_runs] Running: $model $mode"
        echo "Time: $(date)"
        echo "=========================================="

        $PYTHON -u -m agent_stable_slo.cli eval \
            --criteria "$CRITERIA" \
            --suite p1_core \
            --endpoint "$ENDPOINT" \
            --model "$model" \
            --mode "$mode" \
            --out-dir "$OUT_DIR" \
            --temperature 0.0 \
            --stability-k 1 \
            --max-examples 50 \
            --disable-judge

        echo "[$run_count/$total_runs] Complete: $model $mode"

        # Brief pause between runs
        sleep 5
    done
done

echo ""
echo "=========================================="
echo "P1 Full Evaluation Suite Complete"
echo "End time: $(date)"
echo "=========================================="

# Summary
echo ""
echo "Results summary:"
find "$OUT_DIR" -name "summary.json" -exec echo "  - {}" \;
