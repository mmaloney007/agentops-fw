# Paper 2: SLO-Aware Policy Gradient Training for Contract-First Agents

**Author:** Michael Maloney
**Affiliations:** Neuralift; University of New Hampshire
**Contact:** mike.maloney@unh.edu

---

## Abstract

Paper 1 showed how to measure whether an agent is production-ready. This paper shows how to make it production-ready.

The problem with standard RL for agents is that it optimizes the wrong thing. You train for helpfulness or accuracy, and you get a model that scores well on benchmarks but occasionally emits broken JSON, makes up facts, gives different answers to the same question, and takes forever to respond. These are not bugs to fix later—they are direct consequences of what you optimized for.

We fix this by training with a **composite reward** that reflects deployment requirements: schema validity, task accuracy, faithfulness to context, output stability across runs, and latency. We penalize slow responses and flaky outputs directly in the reward signal, not as afterthoughts. The whole thing runs on a single RTX 4090 using LoRA adapters.

**Verified Results (January 2026):**

| Model | Steps | JSON Valid | Reward | Latency (ms) |
|-------|-------|------------|--------|--------------|
| Qwen3-4B | 250 | 95.6% | 2.0 | 1,625 |
| Qwen3-4B | 500 | 97.4% | 2.0 | 1,520 |
| Mistral-7B | 250 | 98.0% | 2.0 | 868 |
| Mistral-7B | 500 | 98.0% | 2.0 | 886 |

**Key Findings:**
- Training improves JSON validity (Qwen: 95.6% → 97.4% over 500 steps)
- Both models converge to 2.0 reward ceiling (schema valid + task correct)
- Mistral-7B runs 40% faster than Qwen with higher validity
- All training on consumer hardware ($1,600 GPU)

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
