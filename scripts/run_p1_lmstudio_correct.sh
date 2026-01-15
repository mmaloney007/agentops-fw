#!/usr/bin/env bash
set -euo pipefail

# P1 Baseline Evaluations with correct LM Studio model IDs
# Uses remote server at 10.0.0.63 which has all 4 models loaded

export AOFW_PROVIDER=lmstudio
export OPENAI_API_KEY="lm-studio"
export MAX_THOUGHT_TOKENS=512
export WANDB_MODE=offline

ENDPOINT="http://10.0.0.63:1234/v1"
CRITERIA="criteria.yaml"
OUTDIR="out/p1_baseline_lmstudio_v2"
mkdir -p "$OUTDIR"

# Model mapping: our names -> LM Studio model IDs
declare -A MODELS
MODELS["qwen3-4b"]="qwen_qwen3-4b-instruct-2507"
MODELS["ministral-8b"]="ministral-3-8b-instruct-2512"
MODELS["llama-3.2-3b"]="llama-3.2-3b-instruct"
MODELS["gemma-3-12b"]="google/gemma-3-12b"

# Run evaluations sequentially (remote server handles one at a time best)
for name in "qwen3-4b" "ministral-8b" "llama-3.2-3b" "gemma-3-12b"; do
    model_id="${MODELS[$name]}"
    echo "=== Running P1 eval for $name (model: $model_id) ==="

    export LMSTUDIO_MODEL="$model_id"
    export OPENAI_API_BASE="$ENDPOINT"

    timestamp=$(date +%Y%m%d_%H%M%S)
    python -m agent_stable_slo.cli eval \
        --criteria "$CRITERIA" \
        --suite p1_core \
        --provider lmstudio \
        --endpoint "$ENDPOINT" \
        --model "$model_id" \
        --mode SPEC_DRIVEN \
        --out-dir "$OUTDIR/p1_${name}_${timestamp}" \
        --max-examples 50 \
        --disable-judge \
        2>&1 | tee "$OUTDIR/${name}_eval.log"

    echo "=== Completed $name ==="
    echo ""
done

echo "All P1 evaluations complete. Results in $OUTDIR"
