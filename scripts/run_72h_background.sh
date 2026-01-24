#!/bin/bash
# Background training script for MacBook Pro M2 Max
# Logs to out/p2_macbook_72h_YYYYMMDD/

set -e

# Setup
export MAMBA_EXE='/opt/homebrew/bin/micromamba'
export MAMBA_ROOT_PREFIX='/Users/maloney/.local/share/mamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate

cd /Users/maloney/Documents/GitHub/agentops-fw

OUT_DIR="out/p2_macbook_72h_$(date +%Y%m%d)"
mkdir -p "$OUT_DIR"
LOG_FILE="${OUT_DIR}/training.log"
STATUS_FILE="${OUT_DIR}/status.txt"

# Models in priority order
declare -a HF_MODELS=(
    "tiiuae/falcon-mamba-7b-instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-2-9b-it"
    "google/gemma-3-12b-it"
    "Qwen/Qwen2.5-3B-Instruct"
)

declare -a MODEL_NAMES=(
    "falcon-mamba-7b"
    "llama-3.1-8b"
    "gemma-2-9b"
    "gemma-3-12b"
    "qwen2.5-3b"
)

SEED=42
STEPS=500

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

update_status() {
    echo "$1" > "$STATUS_FILE"
    echo "Last updated: $(date)" >> "$STATUS_FILE"
}

# Start
log "=============================================="
log "72-HOUR TRAINING RUN STARTED"
log "=============================================="
log "Output: $OUT_DIR"
log "Models: ${#HF_MODELS[@]}"
log ""

update_status "RUNNING: Starting training..."

for i in "${!HF_MODELS[@]}"; do
    MODEL="${HF_MODELS[$i]}"
    NAME="${MODEL_NAMES[$i]}"
    MODEL_OUT="${OUT_DIR}/${NAME}_seed${SEED}_${STEPS}steps"

    log "=============================================="
    log "MODEL $((i+1))/${#HF_MODELS[@]}: $NAME"
    log "HuggingFace: $MODEL"
    log "Output: $MODEL_OUT"
    log "Started: $(date)"
    log "=============================================="

    update_status "RUNNING: Model $((i+1))/${#HF_MODELS[@]} - $NAME (started $(date '+%H:%M'))"

    # Run training
    if python -m agent_stable_slo.train.grpo_trl \
        --model "$MODEL" \
        --output-dir "$MODEL_OUT" \
        --seed "$SEED" \
        --steps "$STEPS" \
        --dtype float16 \
        --use-lora \
        --lora-r 16 \
        --lora-alpha 32 \
        --batch-size 2 \
        --gradient-accumulation-steps 2 \
        2>&1 | tee -a "$LOG_FILE"; then
        log "✅ $NAME COMPLETED SUCCESSFULLY"
    else
        log "❌ $NAME FAILED (exit code $?)"
    fi

    log "Finished: $(date)"
    log ""

    # Clear MPS cache
    python -c "import torch; torch.mps.empty_cache()" 2>/dev/null || true
done

log "=============================================="
log "ALL TRAINING COMPLETE"
log "Finished: $(date)"
log "=============================================="

update_status "COMPLETE: All 5 models finished at $(date)"
