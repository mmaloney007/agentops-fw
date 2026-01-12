# 🚀 Quick Restart Guide

**Use this file to resume work on Papers 1 & 2**

---

## 📁 Key Files (Read These First)

### 1. **Current Status** ⭐ START HERE
```
STATUS.md
```
- What's completed (Phases 1 & 2)
- Current progress (Phase 3 smoke test done)
- Exact next commands to run
- Known issues and workarounds

### 2. **Complete Master Plan**
```
COMPREHENSIVE_PLAN.md
```
- Full 14-day plan across 4 phases
- All code changes documented
- Paper updates detailed
- Success criteria listed

### 3. **Phase 3 Execution Commands** ⭐ RUN THESE NEXT
```
PHASE3_EXECUTION.md
```
- 24 copy-paste ready eval commands (4 models × 6 modes)
- Training commands for P2
- Progress check commands
- Troubleshooting tips

### 4. **Original Detailed Plan** (Claude's cache)
```
/Users/maloney/.claude/plans/melodic-beaming-cosmos.md
```
- Original analysis from exploration phase
- Detailed code locations with line numbers
- Implementation specifications

---

## ⚡ Quick Resume (Copy-Paste This to Claude)

```
I need to resume work on Papers 1 & 2 for arXiv submission.

Please read these files in order:
1. /Users/maloney/Documents/GitHub/agentops-fw/STATUS.md
2. /Users/maloney/Documents/GitHub/agentops-fw/PHASE3_EXECUTION.md

Then execute the immediate next action from STATUS.md to continue Phase 3 experiments.

Context: Phases 1 & 2 are complete (code fixes + paper text). Smoke test passed. Ready to run full evaluation suite across 4 models.
```

---

## 📊 Current State Summary

**Branch**: `paper_1_redirect`

**Completed** ✅:
- Phase 1: Code fixes (faithfulness, stability, tiers, warmup)
- Phase 2: Paper updates (PPO→policy gradient, Paper 2 expanded)
- Smoke test: Verified all features working

**In Progress** 🔄:
- Phase 3: Full evaluation suite (0 of 24 runs complete)

**Pending** ⏸️:
- Phase 3: P2 training experiments
- Phase 3: Generate tables and figures
- Phase 4: Insert data, proofread, compile

---

## 🎯 Immediate Next Action

**Command to run first**:
```bash
cd /Users/maloney/Documents/GitHub/agentops-fw

/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model qwen/qwen3-vl-4b \
  --mode UNCONSTRAINED \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge
```

Then continue with remaining 23 runs from PHASE3_EXECUTION.md

---

## 📋 File Locations Reference

### Documentation
- `RESTART_HERE.md` ← You are here
- `STATUS.md` ← Current status
- `COMPREHENSIVE_PLAN.md` ← Master plan
- `PHASE3_EXECUTION.md` ← Commands to run
- `plan.md` ← Original P1-only plan

### Code (Modified)
- `agent_stable_slo/rewards/composite.py`
- `agent_stable_slo/utils/config.py`
- `agent_stable_slo/train/grpo_train_loop.py`
- `agent_stable_slo/eval/p1_eval_harness.py`

### Papers
- `papers/P1_stable_slo/arxiv/main.tex`
- `papers/P2_reward_stability/arxiv/main.tex`

### Results
- `out/p1_smoke_test/` ← Smoke test results (complete)
- `out/p1_full_eval/` ← Full eval results (pending)
- `out/p2_training/` ← Training results (pending)

### W&B
- Project: https://wandb.ai/neuralift-ai/specsloeval
- Entity: neuralift-ai (mike007)

---

## 🔧 Prerequisites

### Verify Before Starting
```bash
# 1. Check Python environment
/Users/maloney/.local/share/mamba/bin/python --version
# Expected: Python 3.12.11

# 2. Check LM Studio running
curl -s http://localhost:1234/v1/models | jq -r '.data[].id'
# Expected: 4 model IDs
#   qwen/qwen3-vl-4b
#   openai/gpt-oss-20b
#   google/gemma-3-12b
#   mistralai/ministral-3-3b

# 3. Check W&B login
wandb whoami
# Expected: mike007 (neuralift-ai)

# 4. Check git status
git status
# Expected: On branch paper_1_redirect
```

---

## 📈 Progress Tracking

### Check P1 Evaluation Progress
```bash
find out/p1_full_eval -name "summary.json" | wc -l
# Target: 24 (4 models × 6 modes)
# Current: 0
```

### Check P2 Training Progress
```bash
find out/p2_training -name "*.jsonl" | wc -l
# Target: ~12+ training runs
# Current: 0
```

---

## 🆘 If Something Breaks

### LM Studio not responding?
```bash
# Check if running
curl http://localhost:1234/v1/models

# If down, restart LM Studio app
# Then reload all 4 models
```

### W&B upload failing?
```bash
# Check login
wandb login --relogin

# Check network
ping api.wandb.ai
```

### Disk space issues?
```bash
# Check free space
df -h

# Clean up if needed
rm -rf out/p1_smoke_test  # After verifying results copied
```

---

## 📞 Contact

**Author**: Michael Maloney
**Email**: mike.maloney@unh.edu
**Affiliation**: UNH / Neuralift

---

**Last Updated**: 2026-01-10
**Status**: Ready for Phase 3 execution
