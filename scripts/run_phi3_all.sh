#!/bin/bash
# Run all 6 Phi-3-mini training configurations
# Uses the patched wrapper to fix DynamicCache compatibility

set -e

OUT_DIR="out/p2_training_fixed"
SCRIPT="scripts/run_phi3_patched.py"

echo "=== Starting Phi-3-mini P2 Training ==="
echo "Output: $OUT_DIR"
echo "Started: $(date)"

for seed in 42 123 456; do
    for steps in 250 500; do
        run_name="phi-3-mini_seed${seed}_${steps}steps"
        out_path="$OUT_DIR/$run_name"

        if [ -f "$out_path/train_log.jsonl" ] && [ $(wc -l < "$out_path/train_log.jsonl") -ge $steps ]; then
            echo "SKIP: $run_name (already complete)"
            continue
        fi

        echo ""
        echo "=== Running: $run_name ==="
        echo "Started: $(date)"

        python $SCRIPT \
            --config-preset p2_phi3_mini \
            --steps $steps \
            --seed $seed \
            --out "$out_path" \
            --checkpoint-every $steps

        # Quick stats
        total=$(wc -l < "$out_path/train_log.jsonl")
        valid=$(grep -c '"json_valid": 1' "$out_path/train_log.jsonl" || echo 0)
        last50=$(tail -50 "$out_path/train_log.jsonl" | grep -c '"json_valid": 1' || echo 0)
        echo "DONE: $run_name - Overall: $valid/$total, Last-50: $last50/50"
    done
done

echo ""
echo "=== All Phi-3-mini runs complete ==="
echo "Finished: $(date)"
