# Paper 2: SLO-Aware Policy Gradient Training for Contract-First Agents

**Author:** Michael Maloney
**Affiliations:** Neuralift; University of New Hampshire
**Contact:** mike.maloney@unh.edu

---

## Abstract

Paper 1 showed how to measure whether an agent is production-ready. This paper shows how to make it production-ready.

The problem with standard RL for agents is that it optimizes the wrong thing. You train for helpfulness or accuracy, and you get a model that scores well on benchmarks but occasionally emits broken JSON, makes up facts, gives different answers to the same question, and takes forever to respond. These are not bugs to fix later—they are direct consequences of what you optimized for.

We fix this by training with a **composite reward** that reflects deployment requirements: schema validity, task accuracy, faithfulness to context, output stability across runs, and latency. We penalize slow responses and flaky outputs directly in the reward signal, not as afterthoughts. The whole thing runs on a single RTX 4090 using LoRA adapters.

**Verified Results (January 2026, GRPO Training):**

| Model | Steps | JSON Valid | Last-50 Valid | Avg Reward | Latency (ms) |
|-------|-------|------------|---------------|------------|--------------|
| Qwen3-4B | 500 | 22.2% | 0% | 0.120 | 3,203 |
| Llama-3.2-3B | 500 | 14.2% | 0% | -0.128 | 4,029 |
| Gemma-3-12B | 500 | **41.4%** | **78%** | **0.263** | 5,606 |

**Key Findings:**
- **Gemma-3-12B shows clear learning**: JSON validity improves to 78% in final 50 steps, with positive reward
- **Model capacity matters**: 12B model significantly outperforms 3B-4B models on structured output learning
- Smaller models (Qwen3-4B, Llama-3.2-3B) plateau with degraded late-training performance
- All training on consumer hardware (RTX 4090, $1,600 GPU)

---

## W&B Integration

| Feature | Usage in Paper 2 |
|---------|------------------|
| **Training Curves** | Real-time reward, loss, latency tracking per step |
| **Checkpoints** | Adapter artifacts saved every 50 steps |
| **Sweeps** | Hyperparameter optimization for reward weights |
| **Artifacts** | Versioned adapters with full training metadata |

---

## Citation

```bibtex
@article{maloney2026sloaware,
  title={SLO-Aware Policy Gradient Training for Contract-First Agents},
  author={Maloney, Michael},
  journal={arXiv preprint},
  year={2026}
}
```
