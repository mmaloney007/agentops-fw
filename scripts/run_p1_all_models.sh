#!/bin/bash
# P1 Evaluation Script - All 13 Lucky Models
# Uses LM Studio at http://10.11.196.166:1234

set -e

export OPENAI_API_BASE="http://10.11.196.166:1234/v1"
export AOFW_PROVIDER="lmstudio"

OUT_DIR="out/p1_eval_20260118"
TASKS="tasks/clinc_en.jsonl tasks/hotpot_dev.jsonl tasks/t3_tools.jsonl"
MAX_RECORDS=150  # 50 per task type as per paper methodology

# Model IDs from LM Studio (verified working)
declare -A MODELS=(
    ["llama-3.2-1b"]="llama-3.2-1b-instruct"
    ["llama-3.2-3b"]="meta-llama_-_llama-3.2-3b-instruct"
    ["qwen2.5-3b"]="qwen2.5-3b-instruct"
    ["phi-3-mini"]="phi-3-mini-4k-instruct"
    ["qwen3-4b"]="qwen3-4b"
    ["yi-1.5-6b"]="01-ai_-_yi-1.5-6b-chat"
    ["mistral-7b-v0.3"]="mistralai_-_mistral-7b-instruct-v0.3"
    ["falcon-mamba-7b"]="falcon-mamba-7b-instruct"
    ["gpt-oss-20b"]="openai/gpt-oss-20b"
    ["ministral-8b"]="mistralai.ministral-8b-instruct-2410"
    ["llama-3.1-8b"]="meta-llama-llama-3.1-8b-instruct"
    ["gemma-2-9b"]="google/gemma-2-9b"
    ["gemma-3-12b"]="google/gemma-3-12b"
)

echo "========================================"
echo "P1 Evaluation - 13 Lucky Models"
echo "Started: $(date)"
echo "Output: $OUT_DIR"
echo "========================================"

for name in "${!MODELS[@]}"; do
    model_id="${MODELS[$name]}"
    echo ""
    echo ">>> Evaluating: $name ($model_id)"
    echo ">>> Started: $(date)"

    python scripts/eval_t_suite.py \
        --models "lmstudio:$model_id" \
        --tasks $TASKS \
        --out-dir "$OUT_DIR" \
        --run-name "p1_${name}" \
        --max-records $MAX_RECORDS

    echo ">>> Completed: $name at $(date)"
done

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Finished: $(date)"
echo "========================================"
