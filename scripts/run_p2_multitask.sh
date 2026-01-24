#!/bin/bash
# =============================================================================
# P2 Multi-Task Training Script (MacBook M2 Max + RTX 4090 Compatible)
# =============================================================================
#
# Trains models on balanced T1-T5 dataset for Paper 2.
# Auto-detects hardware and adjusts settings accordingly.
#
# Usage:
#   ./scripts/run_p2_multitask.sh [OPTIONS]
#
# Options:
#   --model MODEL       Model to train (default: Qwen/Qwen2.5-3B-Instruct)
#   --steps STEPS       Training steps (default: 500)
#   --seed SEED         Random seed (default: 42)
#   --tasks FILE        Task file (default: tasks/t1t5_balanced.jsonl)
#   --out-dir DIR       Output directory (default: out/p2_multitask)
#   --dry-run           Show config without running
#   --all-seeds         Run with seeds 42, 123, 456
#   --create-dataset    Create balanced dataset first
#
# Examples:
#   # Quick test
#   ./scripts/run_p2_multitask.sh --model Qwen/Qwen2.5-3B-Instruct --steps 50
#
#   # Full run with all seeds
#   ./scripts/run_p2_multitask.sh --model google/gemma-2-9b-it --all-seeds
#
#   # Create dataset and train
#   ./scripts/run_p2_multitask.sh --create-dataset --model Qwen/Qwen2.5-3B-Instruct
#
# =============================================================================

set -e

# Defaults
MODEL="Qwen/Qwen2.5-3B-Instruct"
STEPS=500
SEED=42
TASKS="tasks/t1t5_balanced.jsonl"
OUT_DIR="out/p2_multitask"
DRY_RUN=false
ALL_SEEDS=false
CREATE_DATASET=false
TARGET_PER_TASK=100

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --all-seeds) ALL_SEEDS=true; shift ;;
        --create-dataset) CREATE_DATASET=true; shift ;;
        --target-per-task) TARGET_PER_TASK="$2"; shift 2 ;;
        -h|--help)
            head -35 "$0" | tail -30
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Detect hardware
detect_hardware() {
    if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        echo "mps"
    elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "cuda"
    else
        echo "cpu"
    fi
}

HARDWARE=$(detect_hardware)

# Set hardware-specific defaults
case $HARDWARE in
    mps)
        DTYPE="float16"
        BATCH_SIZE=2
        GRAD_ACCUM=2
        USE_4BIT="false"  # BitsAndBytes doesn't work on MPS
        ;;
    cuda)
        DTYPE="bfloat16"
        BATCH_SIZE=4
        GRAD_ACCUM=1
        USE_4BIT="true"   # 4-bit quantization on CUDA
        ;;
    cpu)
        DTYPE="float32"
        BATCH_SIZE=1
        GRAD_ACCUM=4
        USE_4BIT="false"
        ;;
esac

# Extract model name for output paths
MODEL_NAME=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -d '-')

# Print configuration
echo "============================================================"
echo -e "  ${BLUE}P2 Multi-Task Training Configuration${NC}"
echo "============================================================"
echo ""
echo "  Hardware:     $HARDWARE"
echo "  Model:        $MODEL"
echo "  Model name:   $MODEL_NAME"
echo "  Steps:        $STEPS"
echo "  Seed(s):      $(if $ALL_SEEDS; then echo '42, 123, 456'; else echo $SEED; fi)"
echo "  Tasks file:   $TASKS"
echo "  Output dir:   $OUT_DIR"
echo ""
echo "  dtype:        $DTYPE"
echo "  batch_size:   $BATCH_SIZE"
echo "  grad_accum:   $GRAD_ACCUM"
echo "  4-bit quant:  $USE_4BIT"
echo ""

# Check if dataset exists, create if requested
if [[ ! -f "$TASKS" ]] || $CREATE_DATASET; then
    echo -e "${YELLOW}Creating balanced dataset...${NC}"
    if $DRY_RUN; then
        echo "  [DRY RUN] Would run: python scripts/create_multitask_dataset.py -t $TARGET_PER_TASK -o $TASKS"
    else
        python scripts/create_multitask_dataset.py -t "$TARGET_PER_TASK" -o "$TASKS"
    fi
    echo ""
fi

# Verify dataset exists
if [[ ! -f "$TASKS" ]] && ! $DRY_RUN; then
    echo -e "${RED}ERROR: Task file not found: $TASKS${NC}"
    echo "Run with --create-dataset to create it first."
    exit 1
fi

# Print dataset info
if [[ -f "$TASKS" ]]; then
    TASK_COUNT=$(wc -l < "$TASKS" | tr -d ' ')
    echo "  Dataset:      $TASK_COUNT examples"
    echo "  Epochs/run:   $(echo "scale=1; $STEPS / $TASK_COUNT" | bc)"
    echo ""
fi

if $DRY_RUN; then
    echo -e "${YELLOW}DRY RUN - No training will be executed${NC}"
    exit 0
fi

# Create output directory
mkdir -p "$OUT_DIR"

# Determine seeds to use
if $ALL_SEEDS; then
    SEEDS=(42 123 456)
else
    SEEDS=($SEED)
fi

# Training function
run_training() {
    local seed=$1
    local run_name="${MODEL_NAME}_t1t5_seed${seed}_${STEPS}steps"
    local run_dir="${OUT_DIR}/${run_name}"

    echo "============================================================"
    echo -e "  ${GREEN}Starting Training${NC}"
    echo "  Run name: $run_name"
    echo "  Output:   $run_dir"
    echo "  Started:  $(date)"
    echo "============================================================"

    # Build command
    CMD="python -m agent_stable_slo.train.grpo_train_loop"
    CMD="$CMD --model $MODEL"
    CMD="$CMD --tasks $TASKS"
    CMD="$CMD --steps $STEPS"
    CMD="$CMD --seed $seed"
    CMD="$CMD --out $run_dir"
    CMD="$CMD --checkpoint-every 100"

    # Hardware-specific options
    if [[ "$USE_4BIT" == "true" ]]; then
        CMD="$CMD --load-in-4bit"
    fi

    echo ""
    echo "Command: $CMD"
    echo ""

    # Run training
    eval $CMD

    # Check result
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Training completed successfully${NC}"
    else
        echo -e "${RED}Training failed${NC}"
        return 1
    fi

    echo ""
    echo "Finished: $(date)"
    echo ""
}

# Main training loop
echo ""
echo "Starting training for ${#SEEDS[@]} seed(s)..."
echo ""

START_TIME=$(date +%s)

for seed in "${SEEDS[@]}"; do
    run_training $seed
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_HOURS=$(echo "scale=2; $TOTAL_TIME / 3600" | bc)

echo "============================================================"
echo -e "  ${GREEN}ALL TRAINING COMPLETE${NC}"
echo "============================================================"
echo ""
echo "  Total time: ${TOTAL_TIME}s (${TOTAL_HOURS}h)"
echo "  Results:    $OUT_DIR"
echo ""
ls -la "$OUT_DIR"
