#!/bin/bash
# =============================================================================
# Full P2 Comparison Study: Single-Task vs Multi-Task Training
# =============================================================================
#
# Runs complete comparison study:
#   - 6 conditions: Single-T1, T2, T3, T4, T5 + Multi-T1T5
#   - 5 models: Qwen2.5-3B, Qwen3-4B, Yi-1.5-6B, Gemma-2-9B, Gemma-3-12B
#   - 3 seeds: 42, 123, 456
#   - 2 step configs: 500 (baseline) + 1500 (extended)
#
# Total: 180 runs
# Estimated time on M2 Max: ~4 days (baseline) or ~15 days (full)
#
# Usage:
#   ./scripts/run_full_p2_study.sh [OPTIONS]
#
# Options:
#   --phase PHASE       baseline|extended|all (default: baseline)
#   --models MODELS     small|medium|large|all (default: all)
#   --conditions CONDS  t1,t2,t3,t4,t5,multi or all (default: all)
#   --seeds SEEDS       Comma-separated seeds (default: 42,123,456)
#   --dry-run           Show plan without executing
#   --resume            Skip completed runs
#
# Examples:
#   # Dry run to see plan
#   ./scripts/run_full_p2_study.sh --dry-run
#
#   # Quick test (1 model, 1 condition, 1 seed)
#   ./scripts/run_full_p2_study.sh --models small --conditions t3 --seeds 42
#
#   # Baseline only (500 steps, all models)
#   ./scripts/run_full_p2_study.sh --phase baseline
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

OUT_BASE="out/p2_full_study_$(date +%Y%m%d)"

# Models
MODEL_SMALL="Qwen/Qwen2.5-3B-Instruct"
MODEL_MED1="Qwen/Qwen3-4B"
MODEL_MED2="01-ai/Yi-1.5-6B-Chat"
MODEL_LARGE1="google/gemma-2-9b-it"
MODEL_LARGE2="google/gemma-3-12b-it"

# Task files
TASK_T1="tasks/t1_structured.jsonl"
TASK_T2="tasks/t2_expanded.jsonl"
TASK_T3="tasks/t3_tools.jsonl"
TASK_T4="tasks/t4_bfcl.jsonl"
TASK_T5="tasks/t5_swebench.jsonl"
TASK_MULTI="tasks/t1t5_balanced.jsonl"

# Defaults
PHASE="baseline"
MODELS_FILTER="all"
CONDITIONS_FILTER="all"
SEEDS="42,123,456"
DRY_RUN=false
RESUME=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Python path
PYTHON="/Users/maloney/.local/share/mamba/bin/python"

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        --models) MODELS_FILTER="$2"; shift 2 ;;
        --conditions) CONDITIONS_FILTER="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        --out-dir) OUT_BASE="$2"; shift 2 ;;
        -h|--help) head -35 "$0" | tail -30; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================

get_task_file() {
    local cond="$1"
    case $cond in
        t1) echo "$TASK_T1" ;;
        t2) echo "$TASK_T2" ;;
        t3) echo "$TASK_T3" ;;
        t4) echo "$TASK_T4" ;;
        t5) echo "$TASK_T5" ;;
        multi) echo "$TASK_MULTI" ;;
        *) echo "" ;;
    esac
}

get_models() {
    local filter="$1"
    case $filter in
        small) echo "$MODEL_SMALL" ;;
        medium) echo "$MODEL_MED1 $MODEL_MED2" ;;
        large) echo "$MODEL_LARGE1 $MODEL_LARGE2" ;;
        all) echo "$MODEL_SMALL $MODEL_MED1 $MODEL_MED2 $MODEL_LARGE1 $MODEL_LARGE2" ;;
        *) echo "$filter" ;;  # Custom model
    esac
}

get_conditions() {
    local filter="$1"
    case $filter in
        all) echo "t1 t2 t3 t4 t5 multi" ;;
        *) echo "$filter" | tr ',' ' ' ;;
    esac
}

get_steps() {
    local phase="$1"
    case $phase in
        baseline) echo "500" ;;
        extended) echo "1500" ;;
        all) echo "500 1500" ;;
        *) echo "$phase" ;;
    esac
}

model_short_name() {
    local model="$1"
    basename "$model" | tr '[:upper:]' '[:lower:]' | sed 's/-instruct//' | sed 's/-chat//' | sed 's/-it//'
}

# =============================================================================
# Run Training
# =============================================================================

run_training() {
    local model="$1"
    local condition="$2"
    local steps="$3"
    local seed="$4"

    local task_file=$(get_task_file "$condition")
    local model_name=$(model_short_name "$model")
    local run_name="${model_name}_${condition}_${steps}s_seed${seed}"
    local run_dir="${OUT_BASE}/${run_name}"

    # Check if completed
    if $RESUME && [[ -f "${run_dir}/train_log.jsonl" ]]; then
        echo -e "  ${YELLOW}SKIP${NC} $run_name (exists)"
        return 0
    fi

    if $DRY_RUN; then
        echo -e "  ${CYAN}PLAN${NC} $run_name"
        return 0
    fi

    echo -e "  ${GREEN}RUN${NC}  $run_name"

    mkdir -p "$run_dir"
    local start=$(date +%s)

    $PYTHON -m agent_stable_slo.train.grpo_train_loop \
        --base-model "$model" \
        --tasks "$task_file" \
        --steps "$steps" \
        --seed "$seed" \
        --out "$run_dir" \
        --checkpoint-every 100 \
        > "${run_dir}/stdout.log" 2>&1

    local exit_code=$?
    local end=$(date +%s)
    local dur=$((end - start))

    if [[ $exit_code -eq 0 ]]; then
        echo -e "        ${GREEN}OK${NC} (${dur}s)"
    else
        echo -e "        ${RED}FAIL${NC} (exit $exit_code)"
    fi

    # Clear cache
    $PYTHON -c "import torch; torch.mps.empty_cache() if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else None" 2>/dev/null || true

    return $exit_code
}

# =============================================================================
# Main
# =============================================================================

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  P2 FULL COMPARISON STUDY${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Check python
if [[ ! -f "$PYTHON" ]]; then
    echo -e "${RED}ERROR: Python not found at $PYTHON${NC}"
    exit 1
fi

# Build run list
MODELS=$(get_models "$MODELS_FILTER")
CONDITIONS=$(get_conditions "$CONDITIONS_FILTER")
STEPS_LIST=$(get_steps "$PHASE")
IFS=',' read -ra SEEDS_ARR <<< "$SEEDS"

# Count runs
total_runs=0
for m in $MODELS; do
    for c in $CONDITIONS; do
        for s in $STEPS_LIST; do
            for seed in "${SEEDS_ARR[@]}"; do
                ((total_runs++))
            done
        done
    done
done

# Estimate time (based on smoke test: ~2.5s/step average)
avg_step_time=2.5
if [[ "$PHASE" == "baseline" ]]; then
    est_secs=$(echo "$total_runs * 500 * $avg_step_time" | bc)
elif [[ "$PHASE" == "extended" ]]; then
    est_secs=$(echo "$total_runs * 1500 * $avg_step_time" | bc)
else
    est_secs=$(echo "$total_runs * 1000 * $avg_step_time" | bc)
fi
est_hours=$(echo "scale=1; $est_secs / 3600" | bc)
est_days=$(echo "scale=2; $est_hours / 24" | bc)

echo "Configuration:"
echo "  Phase:      $PHASE"
echo "  Models:     $MODELS_FILTER → $(echo $MODELS | wc -w | tr -d ' ') models"
echo "  Conditions: $CONDITIONS_FILTER → $(echo $CONDITIONS | wc -w | tr -d ' ') conditions"
echo "  Seeds:      ${SEEDS_ARR[*]}"
echo "  Steps:      $STEPS_LIST"
echo ""
echo -e "Total runs:   ${GREEN}$total_runs${NC}"
echo -e "Est. time:    ${YELLOW}~${est_hours}h (${est_days} days)${NC}"
echo ""

if $DRY_RUN; then
    echo -e "${YELLOW}DRY RUN - Plan:${NC}"
    echo ""
fi

# Confirm (unless dry-run)
if ! $DRY_RUN; then
    read -p "Start $total_runs runs? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    mkdir -p "$OUT_BASE"
fi

# Execute runs
completed=0
failed=0

for model in $MODELS; do
    model_name=$(model_short_name "$model")
    echo ""
    echo -e "${CYAN}Model: $model_name${NC}"

    for condition in $CONDITIONS; do
        for steps in $STEPS_LIST; do
            for seed in "${SEEDS_ARR[@]}"; do
                if run_training "$model" "$condition" "$steps" "$seed"; then
                    ((completed++))
                else
                    ((failed++))
                fi
            done
        done
    done
done

# Summary
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
if $DRY_RUN; then
    echo -e "${BLUE}  DRY RUN COMPLETE - $total_runs runs planned${NC}"
else
    echo -e "${BLUE}  STUDY COMPLETE${NC}"
    echo -e "  Completed: ${GREEN}$completed${NC} / Failed: ${RED}$failed${NC}"
    echo -e "  Results: $OUT_BASE"
fi
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
