# Paper 2 Training Comparison Study

## Objective
Compare multi-task vs single-task training to determine:
1. Whether multi-task training helps transfer learning
2. Whether single-task training provides better specialization
3. How task diversity affects capacity threshold findings

## Study Design

### Conditions

| Condition | Training Data | Tasks | Examples |
|-----------|---------------|-------|----------|
| **Single-T1** | t1_structured.jsonl | T1 only (incident classification) | 10 |
| **Single-T3** | t3_tools.jsonl | T3 only (tool calling) | 500 |
| **Multi-Balanced** | t1t5_balanced.jsonl | T1-T5 (100 each) | 500 |
| **Multi-Natural** | t1t5_natural.jsonl | T1-T5 (natural sizes) | ~1300 |

### Models to Test

Focus on models near the capacity threshold:
- **Qwen2.5-3B** (3B) - Outlier that sometimes learns
- **Yi-1.5-6B** (6B) - Near threshold
- **Gemma-2-9B** (9B) - First reliable learner

### Training Protocol

- **Steps**: 500 (standard) + 1500 (extended)
- **Seeds**: 42, 123, 456
- **Metrics**:
  - Overall JSON validity %
  - Last-50 JSON validity %
  - Per-task JSON validity % (multi-task only)
  - Learning curve shape

### Key Questions

1. **Does multi-task help small models?**
   - Hypothesis: Multi-task training may help sub-threshold models learn by exposing them to easier tasks
   - Measure: Compare Qwen2.5-3B on Single-T3 vs Multi-Balanced

2. **Does multi-task hurt specialization?**
   - Hypothesis: Single-task training may achieve higher task-specific validity
   - Measure: Compare T3 validity between Single-T3 and Multi-Balanced conditions

3. **Is the capacity threshold task-dependent?**
   - Hypothesis: Some tasks may have lower thresholds than others
   - Measure: Per-task breakdown on Multi-Balanced for all models

4. **Does task diversity affect learning dynamics?**
   - Hypothesis: Learn-then-forget may be task-specific
   - Measure: Compare learning curves across conditions

## Execution Plan

### Phase 1: Dataset Preparation (1 hour)

```bash
# Create all datasets
python scripts/expand_t2_tasks.py --count 100 --include-original
python scripts/create_multitask_dataset.py -t 100 -o tasks/t1t5_balanced.jsonl
python scripts/create_multitask_dataset.py -s natural -o tasks/t1t5_natural.jsonl
```

### Phase 2: Single-Task Baselines (MacBook, ~20h)

```bash
# Single-T1 (tiny dataset, quick)
./scripts/run_p2_multitask.sh --model Qwen/Qwen2.5-3B-Instruct --tasks tasks/t1_structured.jsonl --all-seeds

# Single-T3 (larger dataset)
./scripts/run_p2_multitask.sh --model Qwen/Qwen2.5-3B-Instruct --tasks tasks/t3_tools.jsonl --all-seeds
```

### Phase 3: Multi-Task Comparison (MacBook, ~40h)

```bash
# Multi-Balanced
./scripts/run_p2_multitask.sh --model Qwen/Qwen2.5-3B-Instruct --tasks tasks/t1t5_balanced.jsonl --all-seeds

# Repeat for Yi-1.5-6B and Gemma-2-9B
```

### Phase 4: Analysis (2 hours)

```bash
# Extract per-task metrics from training logs
python scripts/analyze_multitask_results.py --conditions single-t1 single-t3 multi-balanced

# Generate comparison figures
python scripts/plot_comparison.py
```

## Expected Outcomes

### Positive Results (multi-task helps)
- Sub-threshold models show improved learning on multi-task
- Per-task validity comparable to single-task
- Recommending multi-task training in paper

### Negative Results (multi-task hurts)
- Multi-task shows lower validity than single-task
- Task interference visible in learning curves
- Recommending task-specific training in paper

### Neutral Results (no significant difference)
- Validity similar across conditions
- Capacity threshold unchanged
- Recommending multi-task for convenience, single-task for specialization

## Time Estimates (M2 Max 64GB)

| Model | Single-T1 | Single-T3 | Multi-Balanced | Total |
|-------|-----------|-----------|----------------|-------|
| Qwen2.5-3B | 2h | 6h | 6h | 14h |
| Yi-1.5-6B | 4h | 12h | 12h | 28h |
| Gemma-2-9B | 6h | 18h | 18h | 42h |
| **Per condition** | 12h | 36h | 36h | **84h total** |

*Estimates for 500 steps × 3 seeds per model per condition*

## Files Created

- `tasks/t2_expanded.jsonl` - 106 T2 examples (6 original + 100 generated)
- `tasks/t1t5_balanced.jsonl` - 500 examples (100 per task)
- `tasks/t1t5_natural.jsonl` - ~1300 examples (natural task sizes)
- `scripts/run_p2_multitask.sh` - Portable training script
- `scripts/smoke_test_p2_training.py` - Training validation

## Next Steps

1. Run smoke test: `python3 scripts/smoke_test_p2_training.py`
2. Create balanced dataset with expanded T2
3. Start with smallest model (Qwen2.5-3B) on MacBook
4. Monitor training logs for per-task breakdown
5. Transfer successful configs to RTX 4090 for larger models
