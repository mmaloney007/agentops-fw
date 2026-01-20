# CLAUDE.md - Execution Context for Claude Code

## Current Focus: P1 + P2 Papers with Lucky 13 Models

### Quick Links
- **Strategic Plan**: `plan.md` (thesis, model rationale, timeline)
- **Paper 1**: `papers/P1_stable_slo/arxiv/main.tex` (The Deployment Gap)
- **Paper 2**: `papers/P2_reward_stability/arxiv/main.tex` (Capacity Thresholds)
- **Progress Tracker**: `PROGRESS.md` (checkpoints, completed runs)

---

## The 13 Models (Actual LM Studio IDs - Updated 2025-01-18)

| # | Model | LM Studio Model ID | HuggingFace (Training) | Status |
|---|-------|--------------------|------------------------|--------|
| 1 | Llama-3.2-1B | `lmstudio-community/llama-3.2-1b-instruct` | `meta-llama/Llama-3.2-1B-Instruct` | ✅ ready |
| 2 | Llama-3.2-3B | `RichardErkhov/meta-llama-_-llama-3.2-3b-instruct` | `meta-llama/Llama-3.2-3B-Instruct` | ✅ ready |
| 3 | Qwen2.5-3B | `Qwen/qwen2.5-3b-instruct` | `Qwen/Qwen2.5-3B-Instruct` | ✅ ready |
| 4 | Phi-3-mini | `microsoft/phi-3-mini-4k-instruct` | `microsoft/Phi-3-mini-4k-instruct` | ✅ ready |
| 5 | Qwen3-4B | `Qwen/qwen3-4b` | `Qwen/Qwen3-4B` | ✅ ready |
| 6 | Yi-1.5-6B | `RichardErkhov/01-ai-_-yi-1.5-6b-chat` | `01-ai/Yi-1.5-6B-Chat` | ✅ ready |
| 7 | Mistral-7B-v0.3 | `RichardErkhov/mistralai-_-mistral-7b-instruct-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` | ✅ ready |
| 8 | Falcon-Mamba-7B | `tiiuae/falcon-mamba-7b-instruct@q4_k_m` | `tiiuae/falcon-mamba-7b-instruct` | ✅ ready |
| 9 | GPT-OSS-20B | `openai/gpt-oss-20b` | `openai/gpt-oss-20b` | ✅ ready |
| 10 | Ministral-8B | `DevQuasar/mistralai.ministral-8b-instruct-2410` | `mistralai/Ministral-8B-Instruct-2410` | ✅ ready |
| 11 | Llama-3.1-8B | `featherless-ai-quants/meta-llama-llama-3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | ✅ ready |
| 12 | Gemma-2-9B | `google/gemma-2-9b` | `google/gemma-2-9b-it` | ✅ ready |
| 13 | Gemma-3-12B | `google/gemma-3-12b` | `google/gemma-3-12b-it` | ✅ ready |

**13/13 models ready** ✅ ALL DOWNLOADED!

---

## Execution Commands

### P1 Evaluation (LM Studio required)
```bash
# Single model evaluation
python scripts/eval_t_suite.py \
  --models lmstudio:MODEL_NAME \
  --tasks tasks/clinc_en.jsonl tasks/hotpot_dev.jsonl tasks/t3_tools.jsonl \
  --out-dir out/p1_eval/MODEL_NAME \
  --run-name p1_MODEL_NAME_seed42

# Full 13-model run (after all downloaded)
python scripts/run_p1_p2_full.py --phase baseline --out-dir out/p1_13models
```

### P2 Training (HuggingFace models)
```bash
# Single model training
python -m agent_stable_slo.train.grpo_train_loop \
  --config-preset CONFIG_NAME \
  --steps 500 \
  --out out/p2_train/MODEL_NAME \
  --checkpoint-every 100

# Full training run
python scripts/run_p1_p2_full.py --phase train --out-dir out/p2_13models
```

---

## Recovery Protocol (OOM / Crash Handling)

### Before Starting Any Run
1. Check `PROGRESS.md` for last completed checkpoint
2. Note which model/seed was in progress
3. Resume from last good checkpoint

### After Each Model Completes
1. Update `PROGRESS.md` with:
   - Model name
   - Phase (eval/train)
   - Seed
   - Output path
   - Key metrics (if available)
2. Commit checkpoint: `git add PROGRESS.md && git commit -m "checkpoint: MODEL phase"`

### On OOM or Crash
1. Check `out/*/` for partial results
2. Look for `run_summary.json` or `checkpoint-*` dirs
3. Resume with `--resume` flag if supported, or restart that model only
4. DO NOT re-run completed models

---

## TBD Values to Fill

### Paper 1 (main.tex lines 688-701)
```
GPT-OSS-20B: Accuracy=TBD, Success@SLO=TBD, P95=TBD
Gemma-2-9B: Accuracy=TBD, Success@SLO=TBD, P95=TBD
Llama-3.1-8B: Accuracy=TBD, Success@SLO=TBD, P95=TBD
...
```

### Paper 2 (main.tex lines 773-786)
```
Same models: JSON Valid, Last-50 Valid, Avg Reward, Learning?
```

### How to Update
```bash
# After getting results, update LaTeX directly:
# Find: TBD\% & TBD & TBD\%
# Replace with actual values from out/p1_eval/MODEL/metrics.json
```

---

## What Blocks Progress

| Blocker | Owner | Status |
|---------|-------|--------|
| All 13 models downloaded | User (LM Studio) | ✅ DONE |
| LM Studio server running | User | Required for P1 evals |
| HuggingFace login | User | Required for gated models |
| RTX 4090 available | User | Required for P2 training |

---

## Session Handoff Notes

*Update this section at end of each session:*

**Session (2025-01-18 evening):**
- ✅ ALL 13 MODELS DOWNLOADED AND READY
- Cleaned up LM Studio - removed ~100GB of duplicates/unused models
- Updated CLAUDE.md, PROGRESS.md, run_p1_p2_full.py with actual LM Studio IDs
- Total disk: ~55GB for all 13 models

**Next steps:**
1. Start LM Studio server
2. Run P1 eval on Llama-3.2-1B (smallest, establishes floor)
3. Continue with remaining 8 new models
4. Update PROGRESS.md after each run
