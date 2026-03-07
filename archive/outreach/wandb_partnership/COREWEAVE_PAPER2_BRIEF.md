# Paper 2: Capacity Thresholds in Schema-Aware Training

**Why Small Models Can't Close the Deployment Gap**

---

## Abstract

Paper 1 documented the deployment gap — accuracy doesn't predict production success. This paper asks: can we train models to close the gap?

We train 3 models (3B, 4B, 12B parameters) using Group Relative Policy Optimization (GRPO) with a composite reward that includes schema validity, accuracy, faithfulness, stability, and latency.

**Key finding:** There exists a capacity threshold below which models cannot learn structured output generation through policy gradients.

| Model | Size | JSON Valid (Final 50 Steps) | Learning? |
|-------|------|----------------------------|-----------|
| Llama-3.2-3B | 3B | 0% | ❌ No |
| Qwen3-4B | 4B | 0% | ❌ No |
| Gemma-3-12B | 12B | **78%** | ✅ Yes |

The 12B model shows clear learning (validity improves throughout training). The 3B-4B models plateau and degrade.

**Implication:** Not all models can close the deployment gap. Model capacity is a gating factor.

---

## The Training Problem

Standard RL optimizes for accuracy:
```
reward = accuracy_score
```

This produces models that score well on benchmarks but:
- Emit broken JSON
- Hallucinate facts
- Give different answers to the same question
- Respond slowly

These are direct consequences of what you optimized for.

---

## The Solution: Composite Reward

We train with a reward that reflects deployment requirements:

```python
reward = (
    w1 * schema_validity +      # Does it parse?
    w2 * task_accuracy +        # Is it correct?
    w3 * faithfulness_score +   # Is it grounded?
    w4 * stability_score +      # Is it consistent?
    w5 * latency_penalty        # Is it fast enough?
)
```

**Key design:** Lexicographic gating — no reward for accuracy if JSON is invalid.

---

## Training Results (500 Steps, RTX 4090)

| Model | Size | JSON Valid | Last-50 Valid | Avg Reward | Learning Curve |
|-------|------|------------|---------------|------------|----------------|
| Llama-3.2-3B | 3B | 14.2% | 0% | -0.128 | Plateau then degrade |
| Qwen3-4B | 4B | 22.2% | 0% | 0.120 | Plateau then degrade |
| Gemma-3-12B | 12B | **41.4%** | **78%** | **0.263** | Continuous improvement |

### Learning Curves

**Gemma-3-12B:**
```
Steps 0-100:   ~15% valid
Steps 100-250: ~30% valid
Steps 250-400: ~45% valid
Steps 400-500: ~78% valid  ← Clear improvement
```

**Llama-3.2-3B / Qwen3-4B:**
```
Steps 0-100:   ~20% valid
Steps 100-250: ~25% valid
Steps 250-400: ~15% valid  ← Degradation
Steps 400-500: ~0% valid   ← Collapse
```

---

## The Capacity Threshold

### What We Observe

There appears to be a threshold around 7-12B parameters below which:
- Models cannot learn structured output generation
- Training initially improves, then degrades
- Final performance is worse than baseline

### Why It Matters

If your production requirements include schema compliance, and your model is below the capacity threshold, **training won't help**. You need a larger model.

This explains part of the deployment gap: small models literally cannot learn to meet production requirements, regardless of how you train them.

### What We Don't Know Yet

- Where exactly is the threshold? (Need 7B, 9B experiments)
- Is it task-dependent?
- Does it shift with more training steps?

---

## Implications

### For Model Selection

Don't just ask "which model is most accurate?" Ask "which model can learn to meet my requirements?"

### For Training Investment

If your model is below the capacity threshold, no amount of training will close the deployment gap. Invest in larger models or accept the limitations.

### For the Deployment Gap

This finding connects to Paper 1: the gap between accuracy and Success@SLO is partially explained by model capacity. Small models can't close it.

---

## Hardware

All training on single RTX 4090 (24GB VRAM):
- LoRA adapters (r=16, alpha=32)
- 4-bit quantization for 12B model
- Gradient checkpointing

Consumer hardware is sufficient for this research.

---

## Citation

```bibtex
@article{maloney2026capacity,
  title={Capacity Thresholds in Schema-Aware Training: Why Small Models Can't Close the Deployment Gap},
  author={Maloney, Michael},
  journal={arXiv preprint},
  year={2026}
}
```

---

**Contact:** mike@neuralift.ai
