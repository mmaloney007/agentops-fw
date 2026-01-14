# Paper 2: Verified Training Results (January 2026)

**Hardware:** RTX 4090 (24GB VRAM, $1,600)
**Task:** CLINC150 Intent Classification
**Training:** GRPO with LoRA adapters, 4-bit quantization

## Summary Table

| Model | Steps | JSON Valid | Reward | Latency (ms) | Notes |
|-------|-------|------------|--------|--------------|-------|
| Qwen3-4B | 250 | 95.6% | 2.0 | 1,625 | Initial convergence |
| Qwen3-4B | 500 | 97.4% | 2.0 | 1,520 | Improved validity |
| Mistral-7B | 250 | 98.0% | 2.0 | 868 | Fast convergence |
| Mistral-7B | 500 | 98.0% | 2.0 | 886 | Stable at ceiling |
| Gemma-3-12B | 250 | — | — | — | In progress |
| Gemma-3-12B | 500 | — | — | — | Pending |

## Key Findings

1. **Both models hit reward ceiling (2.0)** = schema valid + task correct
2. **Mistral-7B runs 40% faster** than Qwen (868ms vs 1,520ms)
3. **Mistral achieves higher validity** (98% vs 97.4%) with less training
4. **Training improves JSON validity** (Qwen: 95.6% → 97.4% over 500 steps)
5. **All training on consumer hardware** - no cluster required

## Training Configuration

```yaml
# Common settings
load_in_4bit: true
lora_rank: 16
lora_alpha: 32
gradient_accumulation: 2
max_new_tokens: 64
temperature: 0.7
lr: 1e-5
```

## Output Locations

- Qwen3-4B 250: `out/qwen3-4b_250/`
- Qwen3-4B 500: `out/qwen3-4b_500/`
- Mistral-7B 250: `out/mistral-7b_250/`
- Mistral-7B 500: `out/mistral-7b_500/`
- Gemma-3-12B: `out/gemma-3-12b_250_overnight/` (in progress)

## Reproduction

```bash
# Run all models
./scripts/run_p2_all_models.sh

# Or individual models
python -m agent_stable_slo.train.grpo_train_loop \
  --base-model Qwen/Qwen3-4B \
  --tasks tasks/clinc_en.jsonl \
  --steps 500 \
  --load-in-4bit true
```

---
*Generated: January 13, 2026*
*Author: Mike Maloney <mike.maloney@unh.edu>*
