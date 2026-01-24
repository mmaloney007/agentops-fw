# Full P2 Study: Time Estimates

## Study Design

### Training Conditions (6 total)
| Condition | Dataset | Examples |
|-----------|---------|----------|
| Single-T1 | `t1_structured.jsonl` | 10 |
| Single-T2 | `t2_expanded.jsonl` | 106 |
| Single-T3 | `t3_tools.jsonl` | 500 |
| Single-T4 | `t4_bfcl.jsonl` | 500 |
| Single-T5 | `t5_swebench.jsonl` | 300 |
| Multi-T1T5 | `t1t5_balanced.jsonl` | 500 |

### Models (5 key models for threshold investigation)
| Model | Size | Role in Study |
|-------|------|---------------|
| Qwen2.5-3B | 3B | Outlier (sometimes learns) |
| Qwen3-4B | 4B | Below threshold baseline |
| Yi-1.5-6B | 6B | Near threshold |
| Gemma-2-9B | 9B | Above threshold |
| Gemma-3-12B | 12B | Highest learner |

### Training Steps
- **Baseline**: 500 steps (1 epoch on balanced dataset)
- **Extended**: 1500 steps (3 epochs)
- **Full**: 2000 steps (4 epochs, for promising configurations)

### Seeds
- 3 seeds per configuration: 42, 123, 456

---

## Time Estimates

### Per-Model Time (500 steps, single seed)

| Model | M2 Max (MPS) | RTX 4090 (CUDA) |
|-------|--------------|-----------------|
| Qwen2.5-3B | 2h | 40min |
| Qwen3-4B | 2.5h | 50min |
| Yi-1.5-6B | 4h | 1.3h |
| Gemma-2-9B | 6h | 2h |
| Gemma-3-12B | 10h | 3.3h |
| **Average** | **~5h** | **~1.6h** |

### Full Study Calculation

```
Conditions: 6 (T1, T2, T3, T4, T5, Multi)
Models: 5
Seeds: 3
Step configs: 2 (500 + 1500)

Total runs = 6 × 5 × 3 × 2 = 180 runs
```

### Time by Hardware

#### MacBook Pro M2 Max (64GB)

| Configuration | Runs | Time/Run | Total |
|---------------|------|----------|-------|
| 500 steps × 3 seeds | 90 | ~5h avg | **450h (19 days)** |
| 1500 steps × 3 seeds | 90 | ~15h avg | **1350h (56 days)** |
| **TOTAL** | 180 | - | **1800h (75 days)** |

#### RTX 4090 (24GB)

| Configuration | Runs | Time/Run | Total |
|---------------|------|----------|-------|
| 500 steps × 3 seeds | 90 | ~1.6h avg | **144h (6 days)** |
| 1500 steps × 3 seeds | 90 | ~5h avg | **450h (19 days)** |
| **TOTAL** | 180 | - | **594h (25 days)** |

---

## Practical Execution Plans

### Option A: Full Study (25 days on 4090)
- Run everything
- Most complete evidence for paper
- Recommended if you have dedicated compute

### Option B: Focused Study (10 days on 4090)
Reduce scope while keeping key comparisons:

```
Models: 3 (Qwen2.5-3B, Yi-1.5-6B, Gemma-2-9B)
Conditions: 4 (Single-T1, Single-T3, Single-T5, Multi)
Seeds: 3
Steps: 500 baseline, 1500 only for Gemma-2-9B

Runs = 4 × 3 × 3 × 1 + 4 × 1 × 3 × 1 = 36 + 12 = 48 runs
Time = 48 × ~1.6h = ~77h (3.2 days baseline) + 12 × ~5h = ~60h (2.5 days extended)
Total = ~137h (5.7 days)
```

### Option C: Quick Validation (3 days on 4090)
Minimum viable comparison:

```
Models: 3 (Qwen2.5-3B, Yi-1.5-6B, Gemma-2-9B)
Conditions: 3 (Single-T3, Multi-Balanced, Single-T5)
Seeds: 1 (initial), then 3 for promising
Steps: 500 only

Initial runs = 3 × 3 × 1 = 9 runs = ~14h
Follow-up (2 more seeds for 6 configs) = 12 runs = ~19h
Total = ~33h (1.4 days)
```

### Option D: Hybrid (MacBook + 4090)
Use MacBook for small models, 4090 for large:

**MacBook (parallel):**
- Qwen2.5-3B: All 6 conditions × 3 seeds × 500 steps = 18 runs × 2h = 36h
- Qwen3-4B: All 6 conditions × 3 seeds × 500 steps = 18 runs × 2.5h = 45h
- Total: 81h (~3.4 days)

**RTX 4090 (parallel):**
- Yi-1.5-6B to Gemma-3-12B: 3 models × 6 conditions × 3 seeds × 500 steps = 54 runs
- Time: 54 × avg 2h = 108h (~4.5 days)

**Then extended runs on 4090:**
- 1500-step runs for Gemma models only
- 2 models × 6 conditions × 3 seeds = 36 runs × 5h = 180h (7.5 days)

**Total Hybrid Time: ~12 days** (with MacBook and 4090 running in parallel)

---

## Recommended Approach

### Phase 1: Quick Validation (Day 1-2)
Run on MacBook M2 Max while setting up 4090:
```bash
# Single-T3 vs Multi comparison for Qwen2.5-3B
./scripts/run_p2_multitask.sh --model Qwen/Qwen2.5-3B-Instruct \
    --tasks tasks/t3_tools.jsonl --steps 500 --seed 42

./scripts/run_p2_multitask.sh --model Qwen/Qwen2.5-3B-Instruct \
    --tasks tasks/t1t5_balanced.jsonl --steps 500 --seed 42
```

### Phase 2: Baseline Study (Days 3-8)
On RTX 4090:
```bash
# All single-task baselines + multi-task
for task in t1 t2 t3 t4 t5 multi; do
    for model in qwen2.5-3b yi-1.5-6b gemma-2-9b; do
        ./scripts/run_p2_multitask.sh --model $model --tasks $task --all-seeds
    done
done
```

### Phase 3: Extended Runs (Days 9-15)
On RTX 4090 for models above threshold:
```bash
# 1500-step runs for Gemma models
./scripts/run_p2_multitask.sh --model Gemma-2-9B --steps 1500 --all-seeds
./scripts/run_p2_multitask.sh --model Gemma-3-12B --steps 1500 --all-seeds
```

### Phase 4: Analysis (Day 16)
```bash
python scripts/analyze_multitask_results.py
python scripts/generate_paper_figures.py
```

---

## Summary Table

| Plan | Hardware | Duration | Completeness | Paper Quality |
|------|----------|----------|--------------|---------------|
| Full Study | 4090 | 25 days | 100% | Publication-ready |
| Focused | 4090 | 10 days | 70% | Strong for arXiv |
| Quick | 4090 | 3 days | 40% | Preliminary |
| Hybrid | M2 + 4090 | 12 days | 85% | **Recommended** |

---

## Files Ready

```
tasks/t1_structured.jsonl      # 10 examples
tasks/t2_expanded.jsonl        # 106 examples (NEW)
tasks/t3_tools.jsonl           # 500 examples
tasks/t4_bfcl.jsonl            # 500 examples
tasks/t5_swebench.jsonl        # 300 examples
tasks/t1t5_balanced.jsonl      # 500 examples (100 each)
tasks/t1t5_natural.jsonl       # 1416 examples (natural distribution)

scripts/run_p2_multitask.sh    # Portable training script
scripts/smoke_test_p2_training.py  # Validate before running
scripts/create_multitask_dataset.py  # Create datasets
scripts/expand_t2_tasks.py     # Generate T2 examples
```

## Quick Start Commands

```bash
# 1. Activate environment
micromamba activate

# 2. Run smoke test
python scripts/smoke_test_p2_training.py

# 3. Start first comparison (Qwen2.5-3B, ~4h total)
./scripts/run_p2_multitask.sh --model Qwen/Qwen2.5-3B-Instruct \
    --tasks tasks/t3_tools.jsonl --steps 500 --seed 42 \
    --out-dir out/p2_comparison/single_t3

./scripts/run_p2_multitask.sh --model Qwen/Qwen2.5-3B-Instruct \
    --tasks tasks/t1t5_balanced.jsonl --steps 500 --seed 42 \
    --out-dir out/p2_comparison/multi_balanced
```
