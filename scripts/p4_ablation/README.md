# P4 λ (Lambda) Ablation Experiment

## Overview

This directory contains the orchestration scripts for the **λ ablation experiment**, which tests the latency sensitivity mechanism in GRPO (Group Relative Policy Optimization) across three models exhibiting distinct failure modes.

## What This Experiment Tests

The λ ablation systematically varies the latency penalty weight (`LAMBDA_LATENCY`) to understand how much latency-awareness is needed to address each model's specific failure mode:

### Three Test Scenarios

1. **Qwen3-4B on T5 (Transient Latency Spikes)**
   - Baseline behavior: Produces occasional transient latency spikes within SLO bounds
   - Failure mode: High variance in latency; unpredictable performance
   - Question: Can latency penalty reduce variance without degrading output quality?

2. **Yi-1.5-6B on Mixed (Catastrophic SLO Violations)**
   - Baseline behavior: Consistently exceeds SLO budget on mixed-task workload
   - Failure mode: Entire runs violate latency constraints
   - Question: What λ is sufficient to force SLO-compliant behavior?

3. **Phi-3-mini on T2 (High Latency Tax)**
   - Baseline behavior: Small model on large batch size; high per-token latency penalty
   - Failure mode: Penalized for inherent architectural limitations
   - Question: Can moderate λ balance quality with unavoidable latency penalties?

## Experiment Design

### Configuration Matrix

- **Models**: 3 (Qwen3-4B, Yi-1.5-6B, Phi-3-mini)
- **λ values**: [0.0, 0.05, 0.1, 0.2]
- **Seeds**: [42, 123, 456]
- **Total runs**: 3 × 4 × 3 = **36 training runs**
- **Steps per run**: 1000 (approximately 50 minutes on RTX 4090)
- **Total compute**: ~30 GPU-hours

### Environment Variables

Each run sets:
- `LAMBDA_LATENCY`: The swept parameter (0.0, 0.05, 0.1, 0.2)
- `MU_COST`: Fixed at 0.05 (cost penalty weight)
- `GAMMA_STABILITY`: Fixed at 0.1 (stability penalty weight)

### Output Structure

```
out/p4_ablation/
├── qwen3_4b_T5_lambda0.0_seed42/
│   ├── manifest.json
│   ├── checkpoints/
│   ├── logs/
│   └── metrics/
├── qwen3_4b_T5_lambda0.0_seed123/
├── qwen3_4b_T5_lambda0.05_seed42/
├── ...
├── yi_6b_Mixed_lambda0.2_seed456/
├── phi3_mini_T2_lambda0.2_seed456/
└── ...
```

## Expected Outcomes

### λ = 0.0 (No Latency Penalty)

**Qwen3-4B on T5:**
- Baseline variance in latencies; transient spikes visible
- Highest average token count (no penalty for long outputs)
- Serves as control condition

**Yi-1.5-6B on Mixed:**
- Catastrophic SLO violations; significant percentage of steps exceed SLO
- High variance across seeds
- Demonstrates necessity of latency constraint

**Phi-3-mini on T2:**
- Lowest reward (high latency penalties from task, not model)
- May produce longer outputs despite inherent latency cost

### λ = 0.05 (Light Latency Penalty)

**Qwen3-4B on T5:**
- Slight reduction in variance
- Minimal quality degradation
- May still show occasional transient spikes

**Yi-1.5-6B on Mixed:**
- Improved SLO adherence, but may not fully resolve violations
- Begins learning to control output length
- Increased consistency across seeds

**Phi-3-mini on T2:**
- Balanced improvement; slightly shorter outputs
- Reduced penalty variance

### λ = 0.1 (Moderate Latency Penalty)

**Qwen3-4B on T5:**
- Significant variance reduction
- Transient spikes mostly eliminated
- Minor quality loss

**Yi-1.5-6B on Mixed:**
- Strong SLO compliance; violations largely resolved
- Consistent learning across seeds
- **Expected optimal point**

**Phi-3-mini on T2:**
- Good balance: acceptable output quality with reduced latency penalties

### λ = 0.2 (Strong Latency Penalty)

**Qwen3-4B on T5:**
- Over-cautious behavior; excessively short outputs
- Variance nearly zero (but artificially constrained)
- Notable quality degradation

**Yi-1.5-6B on Mixed:**
- Potential over-suppression; may sacrifice output quality for SLO
- Possible hyperparameter tradeoff visible

**Phi-3-mini on T2:**
- May become too conservative
- Quality-reward tradeoff clearly visible

## How to Interpret Results

### Key Metrics to Track

1. **SLO Violation Rate**: Percentage of inference steps exceeding latency budget
2. **Latency P95**: 95th percentile latency (variance indicator)
3. **Average Output Length**: Tokens per response (inverse proxy for speed)
4. **Task Reward**: Quality metric (should decrease as λ increases, up to a point)
5. **Stability**: Variance across seeds (should decrease with λ)

### Interpretation Checklist

- [ ] Do violation rates decrease monotonically with λ?
- [ ] Is there a "knee" in the quality-vs-latency curve?
- [ ] Do results replicate across seeds (low seed variance)?
- [ ] Does each model show its expected failure mode at λ=0.0?
- [ ] Are there model-specific differences in optimal λ?

### Analysis Plot Suggestions

```
For each model:
1. Subplots: [SLO violation rate vs λ] [Output length vs λ] [Reward vs λ]
2. Across models: Overlay violation rates by λ
3. Seed variance: Error bars on all metrics
4. Stability: Track reward distribution across seeds
```

## Usage

### Run Experiment

```bash
# Print experiment matrix without executing
./run_lambda_ablation.py --dry-run

# Run full experiment (36 runs, ~30 GPU-hours)
./run_lambda_ablation.py

# Resume from interruption (skip completed runs)
./run_lambda_ablation.py --resume

# Custom output directory
./run_lambda_ablation.py --output-dir /path/to/results

# Save experiment matrix to JSON for reference
./run_lambda_ablation.py --save-matrix experiment_matrix.json --dry-run
```

### Monitor Runs

```bash
# Watch real-time progress
watch -n 60 'ls -lah out/p4_ablation/ | wc -l'

# Check completed runs
find out/p4_ablation -name "manifest.json" | wc -l

# Analyze logs from a specific run
tail -f out/p4_ablation/qwen3_4b_T5_lambda0.1_seed42/logs/training.log
```

### Collect Results

After experiments complete, aggregate metrics:

```bash
# Extract key metrics from all runs
for run_dir in out/p4_ablation/*/; do
    echo "$(basename $run_dir): $(cat $run_dir/metrics/final_metrics.json)"
done
```

## Estimated Compute

- **Per run**: ~50 minutes on RTX 4090
- **Total**: 36 runs × 50 min ≈ **1800 minutes ≈ 30 GPU-hours**
- **Wall-clock time** (sequential): ~30 hours
- **Wall-clock time** (4 parallel jobs): ~7.5 hours
- **Wall-clock time** (6 parallel jobs): ~5 hours

### Hardware Requirements

- **GPU**: NVIDIA RTX 4090 (or equivalent ~24GB VRAM)
- **CPU**: 8+ cores recommended
- **Storage**: ~50 GB for all checkpoints + logs
- **RAM**: 32 GB minimum

## Related Files

- `run_lambda_ablation.py`: Main experiment orchestrator
- Training infrastructure: `agent_stable_slo.train.grpo_train_loop`
- Config presets: `p2_qwen3_4b`, `p2_yi_6b`, `p2_phi3_mini`

## Questions & Troubleshooting

**Q: Can I run a subset of the experiments?**
A: Modify the `EXPERIMENT_CONFIG` dict in `run_lambda_ablation.py` to filter models or λ values.

**Q: What if a run crashes?**
A: Use `--resume` flag to skip completed runs (identified by `manifest.json`).

**Q: How do I change the number of seeds?**
A: Edit `EXPERIMENT_CONFIG['seeds']` in the script.

**Q: Can I run multiple experiments in parallel?**
A: Yes, but ensure separate output directories. Consider a job queue (e.g., slurm, ray).

## Citation

This ablation experiment validates the latency-aware GRPO mechanism described in:
> "Stable SLO Optimization with Multi-Objective Reinforcement Learning" (P4 Section 8)

Results should confirm:
1. Latency penalty (λ) is necessary for SLO compliance
2. Optimal λ differs by model/task (0.05-0.1 range expected)
3. Excessive λ (0.2) shows quality degradation
