#!/bin/bash
# Run P1 detailed evaluation on all 13 models with ALL 5 task types + mixed

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

# ALL 6 task types + mixed balanced
TASKS="tasks/clinc_en.jsonl tasks/hotpot_dev.jsonl tasks/t3_tools.jsonl tasks/t4_bfcl.jsonl tasks/t5_swebench.jsonl tasks/public_gsm8k.jsonl tasks/t1t5_balanced.jsonl"

OUT_DIR="out/p1_full_eval"
PARALLEL=4
RUN_NAME="p1_13models_5tasks_$(date +%Y%m%d_%H%M%S)"

echo "=== P1 Full Evaluation: All 13 Models x All 6 Task Types ==="
echo "Start time: $(date)"
echo "Output dir: $OUT_DIR/$RUN_NAME"
echo "Parallel workers: $PARALLEL"
echo ""
echo "Tasks:"
echo "  - T1: clinc_en.jsonl (500 intent classification)"
echo "  - T2: hotpot_dev.jsonl (1000 QA)"
echo "  - T3: t3_tools.jsonl (500 tool calling)"
echo "  - T4: t4_bfcl.jsonl (500 BFCL function calling)"
echo "  - T5: t5_swebench.jsonl (300 SWE-bench patches)"
echo "  - T6: public_gsm8k.jsonl (200 math/GSM8K)"
echo "  - Mixed: t1t5_balanced.jsonl (500 balanced mix)"
echo ""
echo "Total: 3500 tasks per model x 13 models = 45,500 evaluations"
echo ""

mkdir -p "$OUT_DIR/$RUN_NAME"

completed=0
total=${#MODELS[@]}

for model in "${MODELS[@]}"; do
    ((completed++))
    echo ""
    echo "=========================================="
    echo "Model $completed/$total: $model"
    echo "Started: $(date)"
    echo "=========================================="

    $PYTHON scripts/eval_t_suite.py \
        --models "lmstudio:$model" \
        --tasks $TASKS \
        --capture-detailed \
        --parallel $PARALLEL \
        --out-dir "$OUT_DIR" \
        --run-name "$RUN_NAME"

    echo "Completed: $(date)"

    # Show quick summary
    summary_file="$OUT_DIR/$RUN_NAME/lmstudio_${model//\//_}/summary.json"
    if [ -f "$summary_file" ]; then
        echo "Results:"
        cat "$summary_file" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  json_valid: {d.get('json_valid', 'N/A')}\")
print(f\"  t3_success: {d.get('t3_success', 'N/A')}\")
print(f\"  t4_success: {d.get('t4_success', 'N/A')}\")
print(f\"  t5_has_patch: {d.get('t5_has_patch', 'N/A')}\")
print(f\"  t6_success: {d.get('t6_success', 'N/A')}\")
print(f\"  avg_latency: {d.get('avg_latency_ms', 'N/A')}ms\")
"
    fi
    echo ""
done

echo ""
echo "=========================================="
echo "=== All 13 models complete ==="
echo "=========================================="
echo "End time: $(date)"
echo "Results in: $OUT_DIR/$RUN_NAME"
echo ""
echo "Summary of all models:"
for f in "$OUT_DIR/$RUN_NAME"/lmstudio_*/summary.json; do
    model=$(basename $(dirname $f) | sed 's/lmstudio_//')
    echo "  $model:"
    cat "$f" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"    json_valid={d.get('json_valid', 'N/A')}, t3={d.get('t3_success', 'N/A')}, t4={d.get('t4_success', 'N/A')}, t5_patch={d.get('t5_has_patch', 'N/A')}\")
" 2>/dev/null
done
