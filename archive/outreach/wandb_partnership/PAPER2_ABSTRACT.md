# Paper 2: Capacity Thresholds in Schema-Aware Training

**Why Small Models Can't Close the Deployment Gap**

**Author:** Michael Maloney
**Affiliations:** Neuralift; University of New Hampshire
**Contact:** mike@neuralift.ai

---

## Abstract

Paper 1 documented the deployment gap — accuracy doesn't predict production success. This paper asks: can we train models to close the gap?

We train 3 models (3B, 4B, 12B parameters) using Group Relative Policy Optimization (GRPO) with a composite reward including schema validity, accuracy, faithfulness, stability, and latency.

**Key finding:** There exists a capacity threshold below which models cannot learn structured output generation through policy gradients.

| Model | Size | JSON Valid (Final 50 Steps) | Learning? |
|-------|------|----------------------------|-----------|
| Llama-3.2-3B | 3B | 0% | ❌ No |
| Qwen3-4B | 4B | 0% | ❌ No |
| Gemma-3-12B | 12B | **78%** | ✅ Yes |

The 12B model shows continuous improvement throughout training. The 3B-4B models plateau and then degrade.

**Implication:** Not all models can close the deployment gap. Model capacity is a gating factor. If your model is below the threshold, no amount of training will make it production-ready for structured output tasks.

---

## Results (500 Steps, RTX 4090)

| Model | Size | JSON Valid | Last-50 Valid | Avg Reward | Learning Curve |
|-------|------|------------|---------------|------------|----------------|
| Llama-3.2-3B | 3B | 14.2% | 0% | -0.128 | Plateau → degrade |
| Qwen3-4B | 4B | 22.2% | 0% | 0.120 | Plateau → degrade |
| Gemma-3-12B | 12B | **41.4%** | **78%** | **0.263** | Continuous improvement |

---

## Connection to Paper 1

The deployment gap is partially explained by model capacity:
- Small models (3B-4B) cannot learn to meet production requirements
- Large models (12B+) can close the gap through training
- Accuracy alone doesn't tell you which category a model is in

---

## W&B Integration

| Feature | Usage in Paper 2 |
|---------|------------------|
| **Training Curves** | Real-time JSON validity, reward, latency per step |
| **Checkpoints** | Adapter artifacts saved every 50 steps |
| **Sweeps** | Hyperparameter optimization for reward weights |
| **Artifacts** | Versioned adapters with full training metadata |

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
