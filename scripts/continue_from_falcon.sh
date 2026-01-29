#!/bin/bash
# Continue P1 evaluation from falcon-mamba-7b (model 8/13)
# Models 1-7 already completed

set -e

export PYTHONPATH=/Users/maloney/Documents/GitHub/agentops-fw:$PYTHONPATH
export OPENAI_API_BASE=http://localhost:1234/v1
export PYTHONUNBUFFERED=1
PYTHON=/Users/maloney/.local/share/mamba/bin/python

# Remaining 6 models (8-13)
MODELS=(
    "falcon-mamba-7b-instruct"
    "ministral-8b-instruct-2410"
    "meta-llama-llama-3.1-8b-instruct"
    "google/gemma-2-9b"
    "google/gemma-3-12b"
    "openai/gpt-oss-20b"
)

# All 6 task types + mixed balanced
TASKS="tasks/clinc_en.jsonl tasks/hotpot_dev.jsonl tasks/t3_tools.jsonl tasks/t4_bfcl.jsonl tasks/t5_swebench.jsonl tasks/t1t5_balanced.jsonl"

OUT_DIR="out/p1_full_eval"
PARALLEL=4
RUN_NAME="p1_13models_5tasks_20260124_221550"

echo "=== Continuing P1 Evaluation: Models 8-13 ==="
echo "Start time: $(date)"
echo "Output dir: $OUT_DIR/$RUN_NAME"
echo ""
echo "Completed: llama-3.2-1b, llama-3.2-3b, qwen2.5-3b, phi-3-mini, qwen3-4b, yi-1.5-6b, mistral-7b"
echo "Remaining: ${#MODELS[@]} models"
echo ""

completed=7
total=13

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
print(f\"  avg_latency: {d.get('avg_latency_ms', 'N/A')}ms\")
"
    fi
    echo ""
done

echo ""
echo "=========================================="
echo "=== Main evaluation complete ==="
echo "=========================================="
echo "End time: $(date)"

# Now run T6
echo ""
echo "Starting T6 (GSM8K math) evaluation..."
/Users/maloney/Documents/GitHub/agentops-fw/scripts/run_t6_only.sh
