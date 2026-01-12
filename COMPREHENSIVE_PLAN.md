# Comprehensive Plan: Papers 1 & 2 to arXiv Submission

**Author**: Michael Maloney (mike.maloney@unh.edu)
**Objective**: Complete Papers 1 and 2 to PhD-level quality for arXiv submission
**Timeline**: 14 days (4 phases)
**Branch**: `paper_1_redirect`

---

## Executive Summary

**Paper 1 (SpecSLOEval)**: Evaluation framework for SLO-aware structured output generation
**Paper 2 (Reward Stability)**: Policy gradient training with composite SLO-aware rewards

**Current Status**: Phases 1 & 2 complete (code fixes + paper text), Phase 3 in progress (experiments)

---

## Phase 1: Critical Code Fixes (✅ COMPLETE)

### 1.1 Faithfulness Integration in Training

**Problem**: Judge-based faithfulness exists in P1 eval but not used in P2 training

**Files Modified**:
- `agent_stable_slo/rewards/composite.py`
- `agent_stable_slo/utils/config.py`
- `agent_stable_slo/train/grpo_train_loop.py`

**Changes**:
```python
# composite.py - Added faithfulness parameter
def composite_reward(..., faithfulness=1.0, kappa_faithfulness=0.0):
    r += kappa_faithfulness * (faithfulness - 0.5)  # Centered at 0.5

# config.py - Added 6 new fields
enable_faithfulness_judge: bool
judge_base_url: str
judge_model: str
judge_temperature: float
kappa_faithfulness: float
stability_samples: int

# grpo_train_loop.py - Added judge integration
_parse_hotpot_prompt()  # Extract question/context
_canonical_json()  # For stability measurement
# Multi-sample generation loop (lines 404-451)
# Faithfulness judge call (lines 453-467)
```

**Verification**: Smoke test confirmed faithfulness logging in episode records

---

### 1.2 Stability Measurement During Training

**Problem**: `disagreement_rate` hardcoded to 0.0, gamma parameter unused

**Solution**:
- Multi-sample generation (k=1-10 configurable)
- Canonical JSON comparison for disagreement computation
- Real disagreement_rate passed to reward function

**Code Location**: `agent_stable_slo/train/grpo_train_loop.py` lines 404-451

**Verification**: Smoke test showed `disagreement_at_k: 0.0` (perfect agreement)

---

### 1.3 Tier Enforcement in Evaluation

**Problem**: Bronze/Silver/Gold tiers defined but not enforced

**Solution**: `_evaluate_tiers()` function with lexicographic gating

**Code Location**: `agent_stable_slo/eval/p1_eval_harness.py` after line 100

**Tier Logic**:
- **Bronze**: JSON valid ≥ threshold AND schema valid ≥ threshold AND p95 ≤ max
- **Silver**: Bronze + accuracy ≥ min AND faithfulness ≥ min AND disagreement ≤ max
- **Gold**: Silver + tool success ≥ min AND success@SLO ≥ min

**Verification**: Smoke test summary.json shows `tier_bronze: false, tier_silver: false, tier_gold: false`

---

### 1.4 Warmup Requests

**Problem**: Criteria specified 20 warmup requests but not implemented

**Solution**: Loop executing dummy requests before main evaluation

**Code Location**: `agent_stable_slo/eval/p1_eval_harness.py` lines 330-353

**Verification**: Smoke test output shows "[P1] Running 20 warmup requests... [P1] Warmup complete"

---

**Git Commit**: `f0d5b7c` - Phase 1 complete

---

## Phase 2: Paper Updates (✅ COMPLETE)

### 2.1 Paper 1: SpecSLOEval

**File**: `papers/P1_stable_slo/arxiv/main.tex`

#### Fix 1: PPO Terminology (CRITICAL)
**Problem**: Papers claim "PPO" but code implements REINFORCE with baseline

**Changes** (4 locations):
- Line 72: "PPO-style" → "policy gradient"
- Line 416: "TRL PPO" → "policy gradient methods (e.g., REINFORCE)"
- Line 536: "PPO" → "policy gradient"
- Line 1715: "PPO" → "policy gradient methods"

**Impact**: Prevents reviewer rejection for false algorithmic claims

---

#### Fix 2: Decoding Modes Update
**Problem**: Paper described 5 modes but code has 6 with different names

**Changes** (lines 1198-1232):
```latex
\item \textbf{U}: UNCONSTRAINED
\item \textbf{P}: PROVIDER_STRUCTURED
\item \textbf{P+V}: PROVIDER_STRUCTURED_PLUS_VALIDATE
\item \textbf{S}: SPEC_DRIVEN
\item \textbf{SR}: SPEC_DRIVEN_PLUS_REPAIR
\item \textbf{SSC}: SPEC_DRIVEN_PLUS_SELFCONSISTENCY
```

---

#### Fix 3: Task Descriptions Reconciliation
**Problem**: Section 5.3 had conflicting task descriptions

**Changes** (lines 1169-1205):
- T1: CLINC150 Intent Classification (tasks/clinc_en.jsonl, 500 examples)
- T2: HotpotQA Grounded Summaries (tasks/hotpot_dev.jsonl, 1000 examples)
- T3: Tool-Call Episodes (tasks/t3_tools.jsonl, 500 examples)

**Git Commit**: `0e3bd7f` - Paper 1 updates complete

---

### 2.2 Paper 2: Reward Stability

**File**: `papers/P2_reward_stability/arxiv/main.tex`

**Complete Rewrite**: 23 lines → 465 lines (15 pages)

**Structure**:

1. **Abstract** (1 page)
   - Problem: SLO-aware agent training
   - Method: REINFORCE with 6-component composite reward
   - Results: Placeholder data for Phase 3

2. **Introduction** (2 pages)
   - Motivation: Agents need SLO awareness
   - Connection to P1 metrics
   - 6-component reward formula: R = R_schema + R_success + κ·(f-0.5) - λL - μC - γD
   - Contributions: Single-GPU training, LoRA efficiency, SpecSLOEval integration

3. **Background** (3 pages)
   - REINFORCE with baseline
   - LoRA/QLoRA for memory efficiency
   - Constrained MDPs and Lagrangian methods
   - SpecSLOEval metric families

4. **Method** (3 pages)
   - Problem formulation (MDP with SLO constraints)
   - Algorithm 1: REINFORCE with Baseline pseudocode (lines 184-213)
   - Composite reward design (all 6 components detailed)
   - Faithfulness integration via LLM-as-judge
   - Stability measurement via multi-sample generation

5. **Implementation** (2 pages)
   - Single-GPU setup (RTX 4090 / M2 Max)
   - PEFT LoRA configuration
   - W&B logging and artifact tracking
   - SpecSLOEval integration for evaluation

6. **Experiments** (3 pages)
   - Experimental setup
   - Table 1: Schema validity rates (placeholder)
   - Table 2: Faithfulness and stability (placeholder)
   - Table 3: Latency and overhead (placeholder)
   - Ablation studies (vary κ, γ, λ, μ)
   - Pareto frontiers (latency vs quality trade-offs)

7. **Discussion** (1 page)
   - Limitations (single-GPU, simplified reward)
   - Future work (true PPO, multi-GPU, automatic tuning)

8. **Conclusion** (0.5 pages)

**Git Commit**: "Paper 2: Complete expansion to 15-page PhD-level paper"

---

## Phase 3: Verification & Experiments (🔄 IN PROGRESS)

### 3.1 Smoke Test (✅ COMPLETE)

**Configuration**:
- Model: qwen/qwen3-vl-4b
- Mode: SPEC_DRIVEN
- Examples: 10 per task
- Judge: Disabled

**Results**:
- JSON/Schema valid: 100%
- p95 latency: 10,996 ms
- Stability: 0% disagreement
- All tiers failed (p95 exceeds threshold)
- W&B artifacts uploaded successfully

**Output**: `out/p1_smoke_test/p1_core_public_v2/qwen_qwen3-vl-4b/SPEC_DRIVEN/summary.json`

**Verification**: ✅ All Phase 1 code features working correctly

---

### 3.2 Full P1 Evaluation Suite (⏸️ PENDING)

**Objective**: Generate comprehensive evaluation results for Paper 1

**Models** (4 total - all confirmed running in LM Studio):
1. qwen/qwen3-vl-4b
2. openai/gpt-oss-20b
3. google/gemma-3-12b
4. mistralai/ministral-3-3b

**Decoding Modes** (6 total):
1. UNCONSTRAINED
2. PROVIDER_STRUCTURED
3. PROVIDER_STRUCTURED_PLUS_VALIDATE
4. SPEC_DRIVEN
5. SPEC_DRIVEN_PLUS_REPAIR
6. SPEC_DRIVEN_PLUS_SELFCONSISTENCY

**Total Runs**: 4 models × 6 modes = **24 evaluation runs**

**Tasks per Run**:
- T1: CLINC150 (500 examples)
- T2: HotpotQA (1000 examples)
- T3: Tools (500 examples)
- Total: 2000 examples per run

**Command Template**:
```bash
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.cli eval \
  --criteria configs/criteria/p1_core_public_v2.yaml \
  --suite p1_core \
  --endpoint http://localhost:1234/v1 \
  --model <MODEL> \
  --mode <MODE> \
  --out-dir out/p1_full_eval \
  --temperature 0.0 \
  --disable-judge
```

**Estimated Time**: 2-4 hours per model (all 6 modes) = 8-16 hours total

**Deliverables**:
- 24 summary.json files with metrics
- Episode logs for each run
- W&B runs with full artifact lineage
- Data for Paper 1 tables/figures

---

### 3.3 P2 Training Experiments (⏸️ PENDING)

**Objective**: Train models with SLO-aware rewards for Paper 2

**Configurations** (3 types):
1. **Baseline**: Pretrained only, no RL (evaluation baseline)
2. **Naive RL**: Structure + success only (λ=μ=γ=κ=0)
3. **SLO-Aware RL**: All 6 reward components enabled

**Models to Train** (recommend order):
1. mistralai/ministral-3-3b (fastest)
2. qwen/qwen3-vl-4b
3. google/gemma-3-12b
4. openai/gpt-oss-20b (slowest)

**Training Configuration**:
```bash
/Users/maloney/.local/share/mamba/bin/python -m agent_stable_slo.train.grpo_train_loop \
  --base-model <MODEL> \
  --tasks tasks/hotpot_dev.jsonl \
  --schema tasks/schemas/hotpot_explainer_schema.json \
  --enable-faithfulness-judge \
  --judge-base-url http://localhost:1234/v1 \
  --judge-model openai/gpt-oss-20b \
  --kappa-faithfulness 0.5 \
  --stability-samples 3 \
  --gamma-stability 0.2 \
  --lam-latency 0.01 \
  --mu-cost 0.001 \
  --steps 500 \
  --eval-interval 50 \
  --out out/p2_training/<MODEL>_slo_aware \
  --wandb-project agent-stable-slo \
  --wandb-entity neuralift-ai
```

**Hyperparameter Sweeps**:
- Faithfulness weight: κ ∈ {0.0, 0.3, 0.5, 0.7}
- Stability weight: γ ∈ {0.0, 0.1, 0.2, 0.5}
- Latency penalty: λ ∈ {0.0, 0.01, 0.05, 0.1}
- Cost penalty: μ ∈ {0.0, 0.001, 0.01}

**Estimated Time**: 4-8 hours per training run × 3 configs × 4 models = 48-96 hours

**Deliverables**:
- LoRA adapter checkpoints
- Training curves (reward, loss, metrics)
- Before/after comparison on SpecSLOEval
- Data for Paper 2 tables/figures

---

### 3.4 Generate Tables & Figures (⏸️ PENDING)

**Scripts**:
- `scripts/paper/p1_make_tables.py`
- `scripts/paper/p1_make_figures.py`

**Paper 1 Outputs**:
1. **Table 1**: Schema validity rates across models/modes
2. **Table 2**: Accuracy and faithfulness by task
3. **Table 3**: Latency percentiles (p50/p95/p99)
4. **Table 4**: Tier compliance matrix
5. **Figure 1**: Latency CDFs by mode
6. **Figure 2**: Success@SLO bar charts
7. **Figure 3**: Pareto frontier (latency vs quality)

**Paper 2 Outputs**:
1. **Table 1**: Training hyperparameters
2. **Table 2**: Baseline vs RL comparison (replace placeholder)
3. **Table 3**: Ablation studies (replace placeholder)
4. **Figure 1**: Training curves (reward, metrics)
5. **Figure 2**: Pareto frontiers (vary λ, μ)
6. **Figure 3**: Stability analysis (disagreement@k)

**Output Directory**: `papers/P1_stable_slo/arxiv/figs/` and `papers/P2_reward_stability/arxiv/figs/`

---

## Phase 4: Polish & Documentation (⏸️ PENDING)

### 4.1 Insert Experimental Data into Papers

**Paper 1**:
- Replace all `\input{figs/table.csv}` with real data
- Update quantitative claims in abstract/intro
- Verify all figure references

**Paper 2**:
- Replace Tables 1-3 placeholder data
- Update abstract with real numbers
- Add training curve figures

**Estimated Time**: 2-4 hours

---

### 4.2 Proofread Papers

**Checklist**:
- [ ] Abstract: concise, quantitative, no typos
- [ ] Introduction: motivates problem, states contributions
- [ ] Method: algorithm clearly described, notation consistent
- [ ] Experiments: setup described, results interpreted
- [ ] Discussion: limitations acknowledged, future work reasonable
- [ ] Conclusion: summarizes contributions
- [ ] All citations formatted correctly
- [ ] All figures/tables referenced in text
- [ ] No "TODO" or placeholder text remains

**Estimated Time**: 4-6 hours per paper

---

### 4.3 Compile Papers

**Commands**:
```bash
cd papers/P1_stable_slo/arxiv
latexmk -pdf main.tex

cd papers/P2_reward_stability/arxiv
latexmk -pdf main.tex
```

**Verify**:
- [ ] Both PDFs compile without errors
- [ ] All citations resolve
- [ ] All figures appear correctly
- [ ] Page count reasonable (P1: ~20 pages, P2: ~15 pages)

---

### 4.4 Update Documentation

**Files to Update**:
- `README.md`: Add reproducibility instructions
- `EXPERIMENTS.md`: Document how to reproduce all results
- `CITATIONS.md`: Ensure all data sources cited

**Estimated Time**: 2-3 hours

---

### 4.5 Final Review

**Checklist**:
- [ ] All code committed to git with clear commit messages
- [ ] All data files tracked or documented
- [ ] W&B runs logged and accessible
- [ ] Papers compile cleanly
- [ ] All quantitative claims have supporting data
- [ ] No placeholder text in papers
- [ ] Git tag created: `v1.0-camera-ready`

**Estimated Time**: 2-3 hours

---

## Critical Files Manifest

### Code (Modified in Phase 1)
1. `agent_stable_slo/rewards/composite.py` - 6-component reward
2. `agent_stable_slo/utils/config.py` - New config fields
3. `agent_stable_slo/train/grpo_train_loop.py` - Faithfulness + stability
4. `agent_stable_slo/eval/p1_eval_harness.py` - Tiers + warmup

### Papers (Modified in Phase 2)
5. `papers/P1_stable_slo/arxiv/main.tex` - PPO fixes, modes, tasks
6. `papers/P2_reward_stability/arxiv/main.tex` - Full 15-page expansion

### Configuration
7. `configs/criteria/p1_core_public_v2.yaml` - Canonical criteria

### Data
8. `tasks/clinc_en.jsonl` - T1 (500 examples)
9. `tasks/hotpot_dev.jsonl` - T2 (1000 examples)
10. `tasks/t3_tools.jsonl` - T3 (500 examples)

### Schemas
11. `tasks/schemas/clinc_nlu_schema.json`
12. `tasks/schemas/hotpot_explainer_schema.json`
13. `tasks/schemas/t3_tool_call_schema.json`

---

## Known Issues & Workarounds

### Issue 1: Remote Judge Endpoint
**Problem**: `configs/criteria/p1_core_public_v2.yaml` line 82 points to `http://10.0.0.72:1234/v1` (not accessible)

**Workaround**: Use `--disable-judge` or update config to localhost

### Issue 2: W&B Artifact Upload
**Problem**: Initial uploads of 6.3MB files take 30-60s

**Workaround**: Be patient; subsequent runs reuse cached artifacts

---

## Success Criteria

### Code
- ✅ Faithfulness integrated in training
- ✅ Stability measurement working
- ✅ Tier enforcement operational
- ✅ Warmup requests implemented
- ✅ Smoke test passed

### Papers
- ✅ No "PPO" mentions (changed to "policy gradient")
- ✅ Decoding modes match code (6 modes)
- ✅ Paper 2 expanded to 15 pages
- ⏸️ All quantitative claims have data (Phase 3)
- ⏸️ Both papers compile (Phase 4)

### Experiments
- ⏸️ 24 P1 evaluation runs complete
- ⏸️ P2 training runs complete
- ⏸️ All tables/figures generated
- ⏸️ W&B runs logged

### Submission
- ⏸️ Papers proofread
- ⏸️ All citations verified
- ⏸️ Reproducibility documented
- ⏸️ Git tagged for release

---

## Timeline Summary

- **Days 1-3** (Phase 1): ✅ Code fixes complete
- **Days 4-7** (Phase 2): ✅ Paper text complete
- **Days 8-10** (Phase 3): 🔄 Experiments in progress
- **Days 11-14** (Phase 4): ⏸️ Polish & documentation

**Current Day**: Day 8 (Phase 3 just started)

---

**Last Updated**: 2026-01-10
