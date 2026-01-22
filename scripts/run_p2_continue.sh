#!/bin/bash
# P2 Training Continuation Script - Skip Phi-3-mini due to compatibility issue
# Continue from Qwen3-4B

OUT_DIR="out/p2_training"
SEEDS=(42 123 456)
STEPS=(250 500)

# Skip models with issues
SKIP_MODELS=("phi-3-mini")

# Models in size order - starting from where we left off
declare -A MODELS
MODELS=(
    ["qwen3-4b"]="p2_qwen3_4b"
    ["yi-1.5-6b"]="p2_yi_6b"
    ["mistral-7b"]="p2_mistral_7b"
    ["falcon-mamba-7b"]="p2_falcon_mamba"
    ["ministral-8b"]="p2_ministral_8b"
    ["llama-3.1-8b"]="p2_llama_8b"
    ["gemma-2-9b"]="p2_gemma_9b"
    ["gpt-oss-20b"]="p2_gpt_oss"
    ["gemma-3-12b"]="p2_gemma_12b"
)

MODEL_ORDER=(
    "qwen3-4b"
    "yi-1.5-6b"
    "mistral-7b"
    "falcon-mamba-7b"
    "ministral-8b"
    "llama-3.1-8b"
    "gemma-2-9b"
    "gpt-oss-20b"
    "gemma-3-12b"
)

mkdir -p "$OUT_DIR"

for model in "${MODEL_ORDER[@]}"; do
    config="${MODELS[$model]}"

    # Skip problematic models
    if [[ " ${SKIP_MODELS[*]} " =~ " ${model} " ]]; then
        echo "SKIP: $model (known compatibility issue)"
        continue
    fi

    echo "=============================================="
    echo "MODEL: $model (config: $config)"
    echo "=============================================="

    for seed in "${SEEDS[@]}"; do
        for steps in "${STEPS[@]}"; do
            run_name="${model}_seed${seed}_${steps}steps"
            run_dir="$OUT_DIR/$run_name"

            # Skip if already completed
            if [ -f "$run_dir/adapter/adapter_config.json" ]; then
                echo "SKIP: $run_name (already complete)"
                continue
            fi

            echo "RUNNING: $run_name"

            # Use || true to continue on errors
            python -m agent_stable_slo.train.grpo_train_loop \
                --config-preset "$config" \
                --steps "$steps" \
                --seed "$seed" \
                --out "$run_dir" \
                --checkpoint-every 250 \
                2>&1 | tee "$run_dir.log" || {
                    echo "ERROR: $run_name failed, continuing..."
                    continue
                }

            # Quick stats (with error handling)
            python3 -c "
import json
try:
    with open('$run_dir/train_log.jsonl') as f:
        records = [json.loads(l) for l in f]
    if records:
        valid = sum(r['json_valid'] for r in records)
        last50_valid = sum(r['json_valid'] for r in records[-50:])
        avg_reward = sum(r['reward'] for r in records) / len(records)
        print(f'  Steps: {len(records)}, JSON Valid: {100*valid/len(records):.1f}%, Last-50: {100*last50_valid/50:.1f}%, Avg Reward: {avg_reward:.3f}')
except Exception as e:
    print(f'  Stats error: {e}')
" || true
            echo "DONE: $run_name"
            echo ""
        done
    done
done

echo "=============================================="
echo "P2 TRAINING CONTINUATION COMPLETE"
echo "=============================================="
