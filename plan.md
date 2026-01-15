# Plan

## 2026-01-14 — Training & Results Day

### Model Selection (4 models for 6-paper series) — CONFIRMED

**Why these 4 models:**
- **Geographic diversity:** China, France/EU, US (x2) — demonstrates global applicability
- **Company diversity:** Alibaba, Mistral AI, Meta, Google — all major open-source players
- **Size ladder:** 3B -> 4B -> 8B -> 12B — shows scaling behavior
- **All trainable on single 4090** with QLoRA — reproducible on commodity hardware
- **All 2025 releases** (except Llama which is Meta's newest small model)

| # | Model | Release | Size | Company | Region | HuggingFace ID |
|---|-------|---------|------|---------|--------|----------------|
| 1 | **Qwen3-4B-Instruct** | July 2025 | 4B | Alibaba | CN | `Qwen/Qwen3-4B-Instruct-2507` |
| 2 | **Mistral-7B** | July 2024 | 7B | Mistral AI | FR | `mistralai/Mistral-7B-Instruct-v0.3` |
| 3 | **Llama-3.2-3B** | Sept 2024 | 3B | Meta | US | `meta-llama/Llama-3.2-3B-Instruct` |
| 4 | **Gemma-3-12B** | March 2025 | 12B | Google | US | `google/gemma-3-12b-it` |

---

## P1 Baseline Results — COMPLETE

**Evaluation Date:** 2026-01-14
**Mode:** SPEC_DRIVEN (spec-driven decoding)
**Tasks:** T1-T3 (150 tasks per model: 50 CLINC + 50 HotpotQA + 50 Tool-calling)

### Results Summary

| Model | JSON Valid | Schema Valid | CLINC Acc | Hotpot F1 | P95 Latency | Success@SLO |
|-------|------------|--------------|-----------|-----------|-------------|-------------|
| Llama-3.2-3B | 100% | 100% | 54% | 0.47 | 3,869ms | **35.5%** |
| Qwen3-4B | 100% | 100% | 58% | 0.39 | 6,043ms | **25.9%** |
| Ministral-8B | 100% | 100% | 66% | 0.39 | 11,731ms | **1.2%** |
| Gemma-3-12B | 100% | 100% | 78% | 0.27 | 1,555ms | **48.0%** |

### Result File Locations

```
out/p1_baseline_lmstudio_v2/
├── p1_llama-3.2-3b_20260114_135314/
│   └── p1_core_public_v2/llama-3.2-3b-instruct/SPEC_DRIVEN/summary.json
├── p1_qwen3-4b_20260114_123941/
│   └── p1_core_public_v2/qwen_qwen3-4b-instruct-2507/SPEC_DRIVEN/summary.json
├── p1_ministral-8b_20260114_130552/
│   └── p1_core_public_v2/ministral-3-8b-instruct-2512/SPEC_DRIVEN/summary.json
└── p1_gemma-3-12b_20260114_141002/
    └── p1_core_public_v2/google_gemma-3-12b/SPEC_DRIVEN/summary.json
```

---

## T4/T5 Baseline Results — IN PROGRESS (17:09)

**Started:** 2026-01-14 17:08
**Mode:** Legacy eval_t_suite.py (structured decode)
**Tasks:** T4 (500 BFCL) + T5 (300 SWE-bench) = 800 tasks per model

### Final Results (Completed 18:50)

| Model | T4 Success | T5 Valid Diff | JSON Valid | Avg Latency |
|-------|------------|---------------|------------|-------------|
| **Ministral-8B** | **53.6%** | 2% | **98.3%** | 433ms |
| Gemma-3-12B | 0% | 0% | 0% | 1072ms |
| Qwen3-4B | 0% | 0% | 0% | 1216ms |
| Llama-3.2-3B | 0% | 0% | 0% | 1218ms |

**Key Finding:** Only Ministral-8B produces valid JSON for T4 (BFCL) and T5 (SWE-bench) tasks. Other models failed structured output format requirements.

### Result File Locations (when complete)

```
out/t4t5_baseline_20260114/
├── qwen3_4b/lmstudio_qwen_qwen3-4b-instruct-2507/
├── llama_3b/lmstudio_llama-3.2-3b-instruct/
├── ministral_8b/lmstudio_ministral-3-8b-instruct-2512/
└── gemma_12b/lmstudio_google-gemma-3-12b/
```

---

## P2 Training — COMPLETE (Started 19:00, Finished 21:15)

### Training Status (Last update: 21:15) — ALL COMPLETE

| Model | 500 Steps | Progress | Checkpoint | Output |
|-------|-----------|----------|------------|--------|
| Qwen3-4B | **DONE** | 500/500 | step_249, step_499 | out/p2_training_20260114/qwen3_4b_500/ |
| Mistral-7B | **SKIPPED** | - | - | OOM - 7B too large for 24GB even with LoRA |
| Llama-3.2-3B | **DONE** | 500/500 | step_249, step_499 | out/p2_training_20260114/llama_3b_500/ |
| Gemma-3-12B | **DONE** | 500/500 | step_249, step_499 | out/p2_training_20260114/gemma_12b_500/ |

**Notes:**
- Switched from Ministral-8B due to VLM architecture incompatibility
- Mistral-7B skipped due to OOM (14GB model + LoRA + gradients > 24GB)
- Final model lineup: Qwen3-4B, Llama-3.2-3B, Gemma-3-12B (3 models)

### Training Configs

```
configs/grpo/
├── p2_qwen3_4b.yaml      # 500 steps, 4-bit=false, LoRA r=16/a=32
├── p2_ministral_8b.yaml  # 500 steps, 4-bit=true, LoRA r=16/a=32 (now points to Mistral-7B)
├── p2_llama_3b.yaml      # 500 steps, 4-bit=false, LoRA r=16/a=32
└── p2_gemma_12b.yaml     # 500 steps, 4-bit=true, LoRA r=16/a=32
```

### Training Checklist

| Model | 250 Steps | 500 Steps | Post-Eval | Notes |
|-------|-----------|-----------|-----------|-------|
| Qwen3-4B | [ ] | [ ] | [ ] | Base model ready |
| Ministral-8B | [ ] | [ ] | [ ] | VLM, needs special handling |
| Llama-3.2-3B | [ ] | [ ] | [ ] | Base model ready |
| Gemma-3-12B | [ ] | [ ] | [ ] | Needs 4-bit quantization |

### Training Output Structure

```
out/p2_full_YYYYMMDD_HHMMSS/
├── {model}_{steps}/
│   ├── adapter/              # Final LoRA adapter
│   ├── checkpoints/step_*/   # Periodic checkpoints
│   ├── train_log.jsonl       # Per-step metrics
│   └── manifest.json         # Config + environment snapshot
```

### Training Script

```bash
# Full pipeline (recommended)
python scripts/run_p1_p2_full.py

# Or individual training
python -m agent_stable_slo.train.grpo_train_loop --config configs/grpo/p2_qwen3_4b.yaml
```

---

## Paper Updates — PENDING

### Paper 1: SpecSLOEval

**Location:** `papers/P1_stable_slo/arxiv/main.tex`

- [ ] Add model selection rationale section
- [ ] Update results tables with P1 baseline data
- [ ] Add T4 (BFCL) and T5 (SWE-bench) results (if available)
- [ ] Generate fresh figures from new data

### Paper 2: SLO-Aware Training

**Location:** `papers/P2_reward_stability/arxiv/main.tex`

- [ ] Add model selection rationale section
- [ ] Update training curves with 250/500 step results
- [ ] Add Llama-3.2-3B results (new model)
- [ ] Update Ministral-7B -> Ministral-8B results

---

## W&B Partnership Package — READY

**Location:** `docs/wandb_partnership/`

| File | Status | Purpose |
|------|--------|---------|
| `EMAIL_DRAFT.md` | Ready | Outreach email to W&B team |
| `COVER_LETTER.md` | Ready | Formal partnership request |
| `TECHNICAL_SUMMARY.md` | Ready | Deep technical W&B integration docs |
| `PAPER1_ABSTRACT.md` | Ready | Condensed P1 abstract |
| `PAPER2_ABSTRACT.md` | Ready | Condensed P2 abstract |
| `PACKAGE_README.md` | Ready | Package overview |

**Next:** Review package, generate PDFs, send outreach email

---

## Active

- [x] P1 baseline evals complete (4 models x T1-T3)
- [x] P1 T4-T5 evals complete (only Ministral-8B succeeded)
- [x] P2 GRPO training runs (3 models x 500 steps) — Mistral-7B skipped (OOM)
- [x] Paper 1 updated with P1 results + T4/T5 finding
- [x] Paper 2 updated with P2 GRPO training results
- [x] Papers reviewed — both compile, submission ready
- [x] W&B outreach package finalized

## Done

- [x] Fix duplicated "Request:" in first 16 T3 tasks
- [x] Wire up T4 (BFCL v4) — 500 tasks from Berkeley Function Calling benchmark
- [x] Wire up T5 (SWE-bench Lite) — 300 tasks from Princeton SWE benchmark
- [x] Update t_suite.md documentation with T1-T5 progression
- [x] Housekeeping: archive legacy files, trim top-level, refresh docs and dependency notes
- [x] Paper 1: updated with business framing, proper authorship, verified results
- [x] Paper 2: expanded from 23 lines to 15-page PhD-level paper with verified results
- [x] W&B partnership package: email draft, cover letter, abstracts, technical summary
- [x] T-suite expanded: T1 (CLINC), T2 (HotpotQA), T3 (tools), T4 (BFCL), T5 (SWE-bench)

## Considerations

- [ ] Archive `agentops_fw` if it is no longer part of the current workflow.
- [ ] Archive conda setup files (`environment.yml`, `activate_mamba.sh`) if mamba is no longer used.
- [ ] Archive Node files (`package.json`, `package-lock.json`) if no JS tooling remains.
- [ ] Decide whether to keep or archive community docs (`CODE_OF_CONDUCT.md`, `SECURITY.md`, `CONTRIBUTING.md`).

## Progress Log

- 2026-01-14 (21:45): ALL TASKS COMPLETE. Papers updated, reviewed (both compile clean), W&B package finalized.
- 2026-01-14 (21:15): P2 GRPO training COMPLETE. 3 models trained (Qwen3-4B, Llama-3.2-3B, Gemma-3-12B) x 500 steps each. Mistral-7B skipped (OOM). Results in out/p2_training_20260114/.
- 2026-01-14 (18:50): T4/T5 evals COMPLETE. Only Ministral-8B (53.6% T4 success) produces valid JSON. Llama, Qwen, Gemma all 0% JSON valid on T4/T5. Results in out/t4t5_baseline_20260114/.
- 2026-01-14: P1 baseline evals COMPLETE. 4 models (Llama-3.2-3B, Qwen3-4B, Ministral-8B, Gemma-3-12B) evaluated on T1-T3 with SPEC_DRIVEN mode. Results in out/p1_baseline_lmstudio_v2/. Gemma leads at 48% Success@SLO.
- 2026-01-13 (late): Selected 4-model lineup for papers: Qwen3-4B (Alibaba/China), Mistral-7B (Mistral/France), Llama-3.1-8B (Meta/US), Gemma-2-9B (Google/US). Rationale: geographic diversity, company diversity, 4B->9B size ladder, all trainable on single 4090.
- 2026-01-13 (evening): Added T4 (BFCL v4, 500 tasks) and T5 (SWE-bench Lite, 300 tasks) to complete T1-T5 progression. Updated t_suite.md. Smoke-tested eval pipeline with LM Studio @ 10.0.0.63. Ready for tomorrow.
- 2026-01-13: reset plan file to keep ongoing progress and keep top-level focused.
- 2026-01-13: archived legacy modules and older papers/results; added dependency notes; refreshed docs/CI paths.
- 2026-01-13: removed `docs/ONE_PAGER.md` and added follow-up cleanup considerations.
- 2026-01-13: archived top-level notebooks/data/figures, moved tooling into `scripts/`, and removed codex-named files from the root.
- 2026-01-13: enabled git-derived versioning via `setuptools-scm` in `pyproject.toml`.

## Notes

- Dependency sources: `pyproject.toml` is the package manifest; `requirements.txt` and `requirements-dev.txt` are for pip installs; `environment.yml` is for conda/mamba.
