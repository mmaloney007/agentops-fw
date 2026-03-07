# P9 Follow-Up Plan: Seed Sweep First, Model Expansion Second

**Date:** 2026-03-07  
**Paper:** P9 — Heterogeneous On-Device Training  
**Priority:** Replace single-run evidence with replicated evidence before widening the model set

## Context

I reviewed the existing P9 planning docs:

- `docs/plans/2026-03-05-p9-complete-evidence.md`
- `docs/plans/2026-03-04-p9-dual-ane-grpo-spectrum-design.md`
- `docs/plans/2026-03-03-p9-ane-heterogeneous-grpo-design.md`

Those plans were correctly focused on building the first `4 paths x 2 models` evidence matrix. The paper now already states the next limitation clearly: each of the 8 experiments ran once, and the immediate next step is `3 seeds x 8 cells = 24 runs`. That needs to happen before any new-model work.

## Decision

Execution order for P9 follow-up work:

1. Run `3 seeds x 8 cells` for the existing matrix first.
2. Aggregate seeded results into mean/std tables and update the paper.
3. Only after the seeded matrix is complete, add `1-2` more models on MLX and ANE.

This order is the right tradeoff because it strengthens the current claims directly, while new models are only persuasive if the original matrix is already replicated.

## Stage 1: Replication Sweep

### Scope

- Models: `Qwen2.5-0.5B`, `SmolLM2-360M`
- Backends: `public`, `private`, `private-full`, `mlx`
- Seeds: `42`, `123`, `456`
- Total runs: `24`
- Keep the same task file, step count, group size, learning rate, temperature, and token budget as the current published matrix

### Deliverables

- Seed-aware logs for every run
- Aggregated timing, power, and learning summaries with standard deviations
- Updated paper tables/figures using replicated evidence rather than single-run observations

### Verification

- The public and private paths accept `--seed`
- All logs contain `seed`
- `24` log files exist under the seeded results tree
- The analyzer collapses seed-level runs into per-cell mean/std summaries
- A same-seed smoke run is reproducible on at least one CPU path and one ANE path before launching the full sweep

## Stage 2: Model Expansion

Do not start this until Stage 1 is complete and the seeded summaries are in hand.

### Model priority

1. `Stories110M`
   Reason: already wired for the Obj-C public/private stack and CoreML assets exist locally, so it is the lowest-friction ANE extension.

2. `Llama-3.2-1B-Instruct`
   Reason: materially strengthens the paper’s architecture/capacity story beyond the current sub-500M pair and is a better publication-facing addition than another tiny model.

Fallback if `Llama-3.2-1B` is blocked on ANE conversion:

- Use `Phi-3-mini` on MLX first, then promote to ANE only if conversion is tractable.

### Stage 2 execution order

1. MLX smoke test for the new model
2. ANE/CoreML conversion smoke test
3. `2-step` dry run on `public`, `private`, `private-full`, and `mlx`
4. Full `500-step` run only after all four paths complete the smoke pass

## Risks

- Seed sweeps may show high variance for borderline cells, especially `SmolLM2/public`
- ANE conversion for new models may fail on unsupported ops or model size
- The highest-value extra model may only be feasible on MLX initially

## Recommendation

Treat Stage 1 as the paper-quality gate. If the seeded sweep holds, P9 becomes much harder to dismiss. If it does not hold, that is still publishable, but it changes the claim. Either way, the new-model expansion should come after the replicated matrix, not before.
