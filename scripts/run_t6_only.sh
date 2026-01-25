#!/bin/bash
# Run T6 (GSM8K math) evaluation on all 13 models

set -e

export PYTHONPATH=/Users/maloney/Documents/GitHub/agentops-fw:$PYTHONPATH
export OPENAI_API_BASE=http://localhost:1234/v1
PYTHON=/Users/maloney/.local/share/mamba/bin/python

MODELS=(
    "llama-3.2-1b-instruct"
    "meta-llama_-_llama-3.2-3b-instruct"
    "qwen2.5-3b-instruct"
    "phi-3-mini-4k-instruct"
    "qwen3-4b"
    "01-ai_-_yi-1.5-6b-chat"
    "mistralai_-_mistral-7b-instruct-v0.3"
    "falcon-mamba-7b-instruct"
    "ministral-8b-instruct-2410"
    "meta-llama-llama-3.1-8b-instruct"
    "google/gemma-2-9b"
    "google/gemma-3-12b"
    "openai/gpt-oss-20b"
)

TASKS="tasks/public_gsm8k.jsonl"
OUT_DIR="out/p1_t6_math"
PARALLEL=4
RUN_NAME="t6_gsm8k_$(date +%Y%m%d_%H%M%S)"

echo "=== T6 (GSM8K Math) Evaluation: All 13 Models ==="
echo "Start time: $(date)"
echo "Output dir: $OUT_DIR/$RUN_NAME"
echo "Tasks: 200 GSM8K math problems"
echo ""

mkdir -p "$OUT_DIR/$RUN_NAME"

completed=0
total=${#MODELS[@]}

for model in "${MODELS[@]}"; do
    ((completed++))
    echo ""
    echo "Model $completed/$total: $model"
    echo "Started: $(date)"

    $PYTHON scripts/eval_t_suite.py \
        --models "lmstudio:$model" \
        --tasks $TASKS \
        --capture-detailed \
        --parallel $PARALLEL \
        --out-dir "$OUT_DIR" \
        --run-name "$RUN_NAME"

    # Show results
    summary_file="$OUT_DIR/$RUN_NAME/lmstudio_${model//\//_}/summary.json"
    if [ -f "$summary_file" ]; then
        cat "$summary_file" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  t6_success: {d.get('t6_success', 'N/A')}, t6_numeric: {d.get('t6_numeric_match', 'N/A')}, latency: {d.get('avg_latency_ms', 0):.0f}ms\")
"
    fi
done

echo ""
echo "=== T6 Complete ==="
echo "End time: $(date)"

echo ""
echo "Summary:"
for f in "$OUT_DIR/$RUN_NAME"/lmstudio_*/summary.json; do
    model=$(basename $(dirname $f) | sed 's/lmstudio_//')
    t6=$(cat "$f" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('t6_success', 0)*100:.1f}%\")" 2>/dev/null)
    echo "  $model: $t6"
done
