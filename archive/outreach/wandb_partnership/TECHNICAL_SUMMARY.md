# Technical Summary: W&B Integration in AgentOps-FW

**Author:** Mike Maloney <mike.maloney@unh.edu>

---

## Overview

AgentOps-FW is a framework for evaluating and training production-ready LLM agents on single-GPU hardware. Weights & Biases is integrated throughout the stack for experiment tracking, artifact management, and reproducibility.

## W&B Integration Points

### 1. Evaluation Logging (Paper 1)

Every evaluation episode is logged as a structured row in W&B Tables:

```python
# Example episode log structure
{
    "episode_id": "ep_001",
    "model": "qwen/qwen3-vl-4b",
    "decode_mode": "SPEC_DRIVEN",
    "task_id": "t1_clinc_42",

    # Structural metrics
    "json_valid": 1,
    "schema_valid": 1,

    # Task metrics
    "intent_correct": 1,
    "confidence": 0.92,

    # Latency metrics
    "latency_ms": 487,
    "ttft_ms": 123,
    "tokens_out": 45,

    # Faithfulness (T2 only)
    "faithfulness_score": 0.84,
    "contradiction_count": 0,

    # Stability
    "disagreement_rate": 0.0,

    # Full output (for debugging)
    "output_json": {"intent": "greeting", "domain": "general", "is_oos": false}
}
```

### 2. Artifact Versioning

Datasets and schemas are fingerprinted and logged as W&B artifacts:

```python
# Dataset artifact
artifact = wandb.Artifact(
    name="clinc150_tasks",
    type="dataset",
    metadata={
        "sha256": "6f9eab...",
        "num_records": 500,
        "schema_sha256": {"clinc_nlu_schema.json": "1364305..."}
    }
)
artifact.add_file("tasks/clinc_en.jsonl")
run.log_artifact(artifact)
```

This ensures exact reproducibility: any experiment can be re-run with the exact same data.

### 3. Training Metrics (Paper 2)

Policy gradient training logs real-time metrics:

```python
wandb.log({
    "step": step,
    "reward/total": reward,
    "reward/schema": r_schema,
    "reward/accuracy": r_success,
    "reward/faithfulness": r_faith,
    "loss/policy": policy_loss,
    "latency/avg_ms": avg_latency,
    "latency/p95_ms": p95_latency,
    "stability/disagreement": disagreement_rate,
    "tokens/output_avg": avg_tokens,
})
```

### 4. Model Checkpoints

LoRA adapters are saved as W&B artifacts:

```python
artifact = wandb.Artifact(
    name=f"lora_adapter_step_{step}",
    type="model",
    metadata={
        "base_model": "qwen/qwen3-vl-4b",
        "lora_rank": 16,
        "training_steps": step,
        "final_reward": reward,
    }
)
artifact.add_dir("out/adapter/")
run.log_artifact(artifact)
```

## W&B Features Utilized

| Feature | Usage |
|---------|-------|
| **Tables** | Episode-level structured logging |
| **Artifacts** | Dataset/schema versioning, model checkpoints |
| **Sweeps** | Hyperparameter optimization for reward weights |
| **Reports** | Automated experiment summaries |
| **Media** | Latency histograms, reward curves |

## Sample W&B Configuration

```yaml
# wandb config for evaluation runs
project: agentops-fw
entity: mike007
config:
  suite: p1_core
  models:
    - qwen/qwen3-vl-4b
    - mistralai/ministral-3-3b
    - google/gemma-3-12b
    - openai/gpt-oss-20b
  decode_modes:
    - UNCONSTRAINED
    - SPEC_DRIVEN
  tasks:
    - t1_clinc
    - t2_hotpot
    - t3_tools
```

## Reproducibility Guarantees

Every run captures:

1. **Git state:** Commit hash and diff of uncommitted changes
2. **Environment:** Python version, CUDA version, package versions
3. **Hardware:** GPU model, memory, compute capability
4. **Config:** Full hyperparameter dictionary
5. **Data fingerprints:** SHA256 of all input files

This allows any experiment to be exactly reproduced by checking out the git revision, restoring the W&B artifact, and re-running with logged config.

## Verified Training Results (January 2026)

Training runs on RTX 4090 with full W&B logging:

| Model | Steps | JSON Valid | Reward | Latency (ms) | Status |
|-------|-------|------------|--------|--------------|--------|
| Qwen3-4B | 250 | 95.6% | 2.00 | 1,625 | ✅ Complete |
| Qwen3-4B | 500 | 97.4% | 2.00 | 1,520 | ✅ Complete |
| Mistral-7B | 250 | 98.0% | 2.00 | 868 | ✅ Complete |
| Mistral-7B | 500 | 98.0% | 2.00 | 886 | ✅ Complete |
| Gemma-3-12B | 250 | TBD | TBD | TBD | 🔄 Downloading |
| Gemma-3-12B | 500 | TBD | TBD | TBD | ⏳ Pending |

**Key findings:**
- Both models achieve 2.0 reward ceiling with near-perfect JSON validity
- Mistral-7B runs 40% faster than Qwen3-4B (868ms vs 1,520ms avg latency)
- Training improves JSON validity: Qwen rises from 95.6% → 97.4% over 500 steps
- Mistral maintains stable 98% validity across both step counts

All training metrics logged to W&B with per-step granularity including:
- Composite reward breakdown (schema + accuracy + faithfulness + stability)
- Latency tracking (average, p95, p99)
- Loss curves and gradient norms
- Checkpoint artifacts at every 50 steps

## Key Metrics Dashboard

The W&B workspace includes dashboards for:

1. **Evaluation Summary:** Success@SLO across models and decode modes
2. **Latency Distribution:** p50/p95/p99 histograms per configuration
3. **Training Progress:** Reward curves, loss curves, gradient norms
4. **Model Comparison:** Side-by-side quality/latency trade-offs

## Research Program: Contract-First Agent Engineering

This work is part of a six-paper research program targeting a PhD by Publication:

### Paper Arc

```
P1 (Evaluation) → P2 (Training) → P3 (Benchmark) → P4 (MLOps) → P5 (Case Study) → P6 (Standard)
```

| Paper | Contribution | W&B Role |
|-------|--------------|----------|
| **P1** | SpecSLOEval framework | Episode Tables, Artifact versioning |
| **P2** | SLO-aware GRPO training | Training curves, Checkpoint artifacts |
| **P3** | Community benchmark suite | Public leaderboard, Artifact registry |
| **P4** | Continual improvement loops | Drift detection, Safe deployment gates |
| **P5** | Real-world deployment study | Production monitoring, Incident triage |
| **P6** | Proposed evaluation standard | Reference criteria.yaml with W&B |

### Coherent Theme

The unifying thesis: **agents should be designed contract-first, measured against SLOs, and improved continuously under operational constraints**. W&B provides the observability infrastructure that makes this possible.

### PhD by Publication Target (Accelerated)

This research program is structured for a PhD by Publication at University of Portsmouth:

| Milestone | Target |
|-----------|--------|
| P1 + P2 submitted | Feb 2026 |
| P3 (Benchmark) submitted | Mar 2026 |
| P4 (MLOps) submitted | Mar 2026 |
| P5 (Case Study) submitted | Apr 2026 |
| P6 (Standard) submitted | Apr 2026 |
| Register at Portsmouth | Apr 2026 |
| Portfolio + Commentary | Aug 2026 |
| Viva | Oct-Dec 2026 |
| **PhD Complete** | **EOY 2026** |

## Code Availability

All W&B integration code is open source:

- **Repository:** https://github.com/mmaloney007/agentops-fw
- **W&B utilities:** `agent_stable_slo/logging/wandb_utils.py`
- **Structured logger:** `agent_stable_slo/logging/structured.py`

---

*For access to the live W&B workspace, contact mike.maloney@unh.edu*
