#!/bin/bash
# =============================================================================
# 72-Hour P2 Training Run on MacBook Pro M2 Max (64GB)
# =============================================================================
#
# Runs 5 high-priority models sequentially:
#   1. Falcon-Mamba-7B  (~13.5h) - Fill gap, was blocked on 4090
#   2. Llama-3.1-8B     (~12.6h) - Non-Google near threshold
#   3. Gemma-2-9B       (~14.4h) - Confirm learning
#   4. Gemma-3-12B      (~20.2h) - Highest learner
#   5. Qwen2.5-3B       (~6.8h)  - Validate outlier
#
# Total estimated time: ~67.5 hours
#
# Usage:
#   ./scripts/run_72h_macbook.sh [--dry-run]
#
# =============================================================================

set -e

# Configuration
OUT_DIR="out/p2_macbook_72h_$(date +%Y%m%d)"
SEED=42
STEPS=500
LOG_FILE="${OUT_DIR}/run.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for dry run
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}DRY RUN MODE - No training will be executed${NC}"
fi

# Setup mamba environment
setup_env() {
    echo -e "${GREEN}Setting up mamba environment...${NC}"
    export MAMBA_EXE='/opt/homebrew/bin/micromamba'
    export MAMBA_ROOT_PREFIX='/Users/maloney/.local/share/mamba'
    eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
    micromamba activate

    # Verify
    python --version
    which python
}

# Create output directory
mkdir -p "$OUT_DIR"

# Models to train (in priority order)
declare -a MODELS=(
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

declare -a EST_HOURS=(
    "13.5"
    "12.6"
    "14.4"
    "20.2"
    "6.8"
)

# Print plan
echo "============================================================"
echo "  72-Hour P2 Training Plan - MacBook Pro M2 Max"
echo "============================================================"
echo ""
echo "Output directory: $OUT_DIR"
echo "Seed: $SEED"
echo "Steps: $STEPS"
echo ""
echo "Models to train:"
echo "------------------------------------------------------------"
total_hours=0
for i in "${!MODELS[@]}"; do
    printf "  %d. %-25s ~%5sh\n" $((i+1)) "${MODEL_NAMES[$i]}" "${EST_HOURS[$i]}"
    total_hours=$(echo "$total_hours + ${EST_HOURS[$i]}" | bc)
done
echo "------------------------------------------------------------"
printf "  TOTAL ESTIMATED TIME:        ~%5sh\n" "$total_hours"
echo ""

if $DRY_RUN; then
    echo -e "${YELLOW}Dry run complete. Run without --dry-run to start training.${NC}"
    exit 0
fi

# Confirm before starting
read -p "Start training? This will take ~68 hours. (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Setup environment
setup_env

# Log start
echo "Training started at $(date)" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Train each model
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NAME="${MODEL_NAMES[$i]}"
    EST="${EST_HOURS[$i]}"

    echo "============================================================" | tee -a "$LOG_FILE"
    echo "  Model $((i+1))/5: $NAME" | tee -a "$LOG_FILE"
    echo "  HuggingFace ID: $MODEL" | tee -a "$LOG_FILE"
    echo "  Estimated time: ~${EST}h" | tee -a "$LOG_FILE"
    echo "  Started at: $(date)" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    MODEL_OUT="${OUT_DIR}/${NAME}_seed${SEED}_${STEPS}steps"

    # Run training
    # Using float16 (no BitsAndBytes 4-bit on MPS)
    python -m agent_stable_slo.train.grpo_trl \
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
        2>&1 | tee -a "$LOG_FILE"

    # Check exit status
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $NAME completed successfully${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${RED}❌ $NAME failed${NC}" | tee -a "$LOG_FILE"
    fi

    echo "Finished at: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Clear MPS cache between models
    python -c "import torch; torch.mps.empty_cache()" 2>/dev/null || true
done

echo "============================================================" | tee -a "$LOG_FILE"
echo "  ALL TRAINING COMPLETE" | tee -a "$LOG_FILE"
echo "  Finished at: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Generate summary
echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $OUT_DIR" | tee -a "$LOG_FILE"
ls -la "$OUT_DIR" | tee -a "$LOG_FILE"
