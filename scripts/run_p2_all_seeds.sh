#!/bin/bash
# P2 Training Runner - All 13 models x 3 seeds x 2 step counts
# Resume from where we left off if OOM/crash

set -e

OUT_DIR="out/p2_training"
SEEDS=(42 123 456)
STEPS=(250 500)

# Models in size order (smallest to largest)
declare -A MODELS
MODELS=(
    ["llama-3.2-1b"]="p2_llama_1b"
    ["llama-3.2-3b"]="p2_llama_3b"
    ["qwen2.5-3b"]="p2_qwen25_3b"
    ["phi-3-mini"]="p2_phi3_mini"
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

# Order matters - smallest to largest
MODEL_ORDER=(
    "llama-3.2-1b"
    "llama-3.2-3b"
    "qwen2.5-3b"
    "phi-3-mini"
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

            python -m agent_stable_slo.train.grpo_train_loop \
                --config-preset "$config" \
                --steps "$steps" \
                --seed "$seed" \
                --out "$run_dir" \
                --checkpoint-every 250 \
                2>&1 | tee "$run_dir.log"

            # Quick stats
            python3 -c "
import json
with open('$run_dir/train_log.jsonl') as f:
    records = [json.loads(l) for l in f]
valid = sum(r['json_valid'] for r in records)
last50_valid = sum(r['json_valid'] for r in records[-50:])
avg_reward = sum(r['reward'] for r in records) / len(records)
print(f'  Steps: {len(records)}, JSON Valid: {100*valid/len(records):.1f}%, Last-50: {100*last50_valid/50:.1f}%, Avg Reward: {avg_reward:.3f}')
"
            echo "DONE: $run_name"
            echo ""
        done
    done
done

echo "=============================================="
echo "ALL P2 TRAINING COMPLETE"
echo "=============================================="
