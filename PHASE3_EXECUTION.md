# Phase 3 Execution Plan - Step-by-Step Commands

**Status**: Ready to execute
**Prerequisites**: LM Studio running with all 4 models loaded
**Working Directory**: `/Users/maloney/Documents/GitHub/agentops-fw`

---

## Quick Start (Copy-Paste Ready)

### Step 1: Verify Environment
```bash
cd /Users/maloney/Documents/GitHub/agentops-fw
/Users/maloney/.local/share/mamba/bin/python --version
curl -s http://localhost:1234/v1/models | grep '"id"'
```

Expected output: Python 3.12.11 and 4 model IDs

---

## Part A: Full P1 Evaluation Suite (24 Runs)

**Total Time**: ~8-16 hours
**Output Directory**: `out/p1_full_eval/`

### Model 1: Qwen3-VL-4B (6 modes)

```bash
# Mode 1: UNCONSTRAINED
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model qwen/qwen3-vl-4b \
  --mode UNCONSTRAINED \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 2: PROVIDER_STRUCTURED
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model qwen/qwen3-vl-4b \
  --mode PROVIDER_STRUCTURED \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 3: PROVIDER_STRUCTURED_PLUS_VALIDATE
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model qwen/qwen3-vl-4b \
  --mode PROVIDER_STRUCTURED_PLUS_VALIDATE \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 4: SPEC_DRIVEN
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model qwen/qwen3-vl-4b \
  --mode SPEC_DRIVEN \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 5: SPEC_DRIVEN_PLUS_REPAIR
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model qwen/qwen3-vl-4b \
  --mode SPEC_DRIVEN_PLUS_REPAIR \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 6: SPEC_DRIVEN_PLUS_SELFCONSISTENCY
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model qwen/qwen3-vl-4b \
  --mode SPEC_DRIVEN_PLUS_SELFCONSISTENCY \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge
```

**Progress Check**:
```bash
find out/p1_full_eval -name "summary.json" | grep qwen | wc -l
# Should show 6 after all modes complete
```

---

### Model 2: GPT-OSS-20B (6 modes)

```bash
# Mode 1: UNCONSTRAINED
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model openai/gpt-oss-20b \
  --mode UNCONSTRAINED \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 2: PROVIDER_STRUCTURED
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model openai/gpt-oss-20b \
  --mode PROVIDER_STRUCTURED \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 3: PROVIDER_STRUCTURED_PLUS_VALIDATE
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model openai/gpt-oss-20b \
  --mode PROVIDER_STRUCTURED_PLUS_VALIDATE \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 4: SPEC_DRIVEN
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model openai/gpt-oss-20b \
  --mode SPEC_DRIVEN \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 5: SPEC_DRIVEN_PLUS_REPAIR
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model openai/gpt-oss-20b \
  --mode SPEC_DRIVEN_PLUS_REPAIR \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 6: SPEC_DRIVEN_PLUS_SELFCONSISTENCY
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model openai/gpt-oss-20b \
  --mode SPEC_DRIVEN_PLUS_SELFCONSISTENCY \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge
```

**Progress Check**:
```bash
find out/p1_full_eval -name "summary.json" | grep gpt-oss | wc -l
# Should show 6 after all modes complete
```

---

### Model 3: Gemma-3-12B (6 modes)

```bash
# Mode 1: UNCONSTRAINED
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model google/gemma-3-12b \
  --mode UNCONSTRAINED \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 2: PROVIDER_STRUCTURED
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model google/gemma-3-12b \
  --mode PROVIDER_STRUCTURED \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 3: PROVIDER_STRUCTURED_PLUS_VALIDATE
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model google/gemma-3-12b \
  --mode PROVIDER_STRUCTURED_PLUS_VALIDATE \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 4: SPEC_DRIVEN
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model google/gemma-3-12b \
  --mode SPEC_DRIVEN \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 5: SPEC_DRIVEN_PLUS_REPAIR
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model google/gemma-3-12b \
  --mode SPEC_DRIVEN_PLUS_REPAIR \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 6: SPEC_DRIVEN_PLUS_SELFCONSISTENCY
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model google/gemma-3-12b \
  --mode SPEC_DRIVEN_PLUS_SELFCONSISTENCY \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge
```

**Progress Check**:
```bash
find out/p1_full_eval -name "summary.json" | grep gemma | wc -l
# Should show 6 after all modes complete
```

---

### Model 4: Ministral-3B (6 modes)

```bash
# Mode 1: UNCONSTRAINED
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model mistralai/ministral-3-3b \
  --mode UNCONSTRAINED \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 2: PROVIDER_STRUCTURED
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model mistralai/ministral-3-3b \
  --mode PROVIDER_STRUCTURED \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 3: PROVIDER_STRUCTURED_PLUS_VALIDATE
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model mistralai/ministral-3-3b \
  --mode PROVIDER_STRUCTURED_PLUS_VALIDATE \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 4: SPEC_DRIVEN
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model mistralai/ministral-3-3b \
  --mode SPEC_DRIVEN \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 5: SPEC_DRIVEN_PLUS_REPAIR
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model mistralai/ministral-3-3b \
  --mode SPEC_DRIVEN_PLUS_REPAIR \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge

# Mode 6: SPEC_DRIVEN_PLUS_SELFCONSISTENCY
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model mistralai/ministral-3-3b \
  --mode SPEC_DRIVEN_PLUS_SELFCONSISTENCY \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge
```

**Final Progress Check**:
```bash
find out/p1_full_eval -name "summary.json" | wc -l
# Should show 24 total (4 models × 6 modes)

find out/p1_full_eval -name "summary.json" -exec wc -l {} \; | awk '{sum+=$1} END {print sum " total lines"}'
# Verify all summary files were written
```

---

## Part B: P2 Training Experiments (12+ Runs)

**Total Time**: ~48-96 hours (can run in background)
**Output Directory**: `out/p2_training/`

### Quick Baseline (Ministral - Fastest)

```bash
# Baseline training with all 6 reward components
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.train.grpo_train_loop \
  --base-model mistralai/ministral-3-3b \
  --tasks tasks/hotpot_dev.jsonl \
  --out out/p2_training/ministral_baseline \
  --steps 500 \
  --eval-interval 50 \
  --enable-faithfulness-judge \
  --judge-base-url http://localhost:1234/v1 \
  --judge-model openai/gpt-oss-20b \
  --kappa-faithfulness 0.5 \
  --stability-samples 3 \
  --gamma-stability 0.2 \
  --lam-latency 0.01 \
  --mu-cost 0.001 \
  --lr 1e-5 \
  --lora-rank 16
```

**Monitor Progress**:
```bash
tail -f out/p2_training/ministral_baseline/train.log
# Or check W&B: https://wandb.ai/neuralift-ai/agent-stable-slo
```

---

### Ablation: Structure + Success Only (No SLO components)

```bash
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.train.grpo_train_loop \
  --base-model mistralai/ministral-3-3b \
  --tasks tasks/hotpot_dev.jsonl \
  --out out/p2_training/ministral_naive \
  --steps 500 \
  --eval-interval 50 \
  --kappa-faithfulness 0.0 \
  --gamma-stability 0.0 \
  --lam-latency 0.0 \
  --mu-cost 0.0 \
  --lr 1e-5 \
  --lora-rank 16
```

---

### Ablation: Vary Faithfulness Weight κ

```bash
# κ = 0.3
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.train.grpo_train_loop \
  --base-model mistralai/ministral-3-3b \
  --tasks tasks/hotpot_dev.jsonl \
  --out out/p2_training/ministral_kappa_0.3 \
  --steps 500 \
  --enable-faithfulness-judge \
  --judge-base-url http://localhost:1234/v1 \
  --judge-model openai/gpt-oss-20b \
  --kappa-faithfulness 0.3 \
  --stability-samples 3 \
  --gamma-stability 0.2 \
  --lam-latency 0.01 \
  --mu-cost 0.001

# κ = 0.7
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.train.grpo_train_loop \
  --base-model mistralai/ministral-3-3b \
  --tasks tasks/hotpot_dev.jsonl \
  --out out/p2_training/ministral_kappa_0.7 \
  --steps 500 \
  --enable-faithfulness-judge \
  --judge-base-url http://localhost:1234/v1 \
  --judge-model openai/gpt-oss-20b \
  --kappa-faithfulness 0.7 \
  --stability-samples 3 \
  --gamma-stability 0.2 \
  --lam-latency 0.01 \
  --mu-cost 0.001
```

---

## Part C: Generate Tables & Figures

**After all P1 evals complete**:

```bash
# Generate Paper 1 tables
/Users/maloney/.local/share/mamba/bin/python scripts/paper/p1_make_tables.py \
  --results-dir out/p1_full_eval \
  --output-dir papers/P1_stable_slo/arxiv/figs

# Generate Paper 1 figures
/Users/maloney/.local/share/mamba/bin/python scripts/paper/p1_make_figures.py \
  --results-dir out/p1_full_eval \
  --output-dir papers/P1_stable_slo/arxiv/figs
```

**After P2 training complete**:

```bash
# Export training data from W&B and generate tables manually
# (Scripts may need to be created for P2)
```

---

## Verification Commands

### Check All P1 Runs Completed
```bash
find out/p1_full_eval -name "summary.json" | while read f; do
  echo "=== $f ==="
  jq '.model, .decode_mode, .num_episodes, .json_valid_rate, .p95_latency_ms' "$f"
done
```

### Check W&B Runs
```bash
# List all W&B runs
wandb runs list neuralift-ai/specsloeval
```

### Aggregate Summary Statistics
```bash
# Create a CSV of all results
find out/p1_full_eval -name "summary.json" -exec jq -r '[.model, .decode_mode, .json_valid_rate, .schema_valid_rate, .p95_latency_ms, .tier_bronze, .tier_silver, .tier_gold] | @csv' {} \; > out/p1_full_results.csv

# View
column -t -s',' out/p1_full_results.csv | less
```

---

## Troubleshooting

### If a run fails:
1. Check the error message
2. Verify LM Studio is still running: `curl http://localhost:1234/v1/models`
3. Check disk space: `df -h`
4. Re-run the specific command

### If LM Studio crashes:
1. Restart LM Studio
2. Reload all 4 models
3. Resume from last successful run

### If W&B is slow:
1. Check network connection
2. Consider running overnight for large artifact uploads

---

## Time Estimates

**P1 Evaluation**:
- Per run: ~30-60 minutes (2000 examples)
- Per model (6 modes): ~3-6 hours
- All 4 models: ~12-24 hours

**P2 Training**:
- Per run (500 steps): ~4-8 hours
- Recommended: Start with Ministral (fastest) to validate

**Total Phase 3**: ~2-4 days of wall-clock time

---

**Next**: Once experiments complete, proceed to PHASE4_POLISH.md
