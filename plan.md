# Plan

## Tomorrow (2026-01-14) — Training & Results Day

### Model Selection (4 models for 6-paper series) — UPDATED 2025

**Why these 4 models:**
- **Geographic diversity:** China, France/EU, US (×2) — demonstrates global applicability
- **Company diversity:** Alibaba, Mistral AI, Meta, Google — all major open-source players
- **Size ladder:** 3B → 4B → 8B → 12B — shows scaling behavior
- **All trainable on single 4090** with QLoRA — reproducible on commodity hardware
- **All 2025 releases** (except Llama which is Meta's newest small model)

| # | Model | Release | Size | Company | Region | HuggingFace ID |
|---|-------|---------|------|---------|--------|----------------|
| 1 | **Qwen3-4B-Instruct** | Aug 2025 | 4B | Alibaba | 🇨🇳 | `Qwen/Qwen3-4B-Instruct` |
| 2 | **Ministral-3-8B** | Dec 2025 | 8B | Mistral AI | 🇫🇷 | `mistralai/Ministral-3-8B-Instruct-2512` |
| 3 | **Llama-3.2-3B** | Sept 2024 | 3B | Meta | 🇺🇸 | `meta-llama/Llama-3.2-3B-Instruct` |
| 4 | **Gemma-3-12B** | March 2025 | 12B | Google | 🇺🇸 | `google/gemma-3-12b-it` |

**Changes from previous lineup:**
- Mistral: Ministral-3-8B (Dec 2025) replaces Mistral-7B (May 2024)
- Google: Gemma-3-12B (March 2025) replaces Gemma-2-9B (June 2024)
- Meta: Llama-3.2-3B (Sept 2024) — no 8B in 3.2/3.3, this is their latest small model

**Paper alignment notes (discuss in morning):**
- P1 already uses Qwen3-VL-4B and Gemma-3-12B ✓
- P2 references Qwen3-4B, Mistral-7B, Gemma-3-12B, Ministral
- Need to update P2: Mistral-7B → Ministral-3-8B, add Llama-3.2-3B results

### Morning: Setup & Quick Fixes
- [ ] **CONFIRM MODEL PICKS** — discuss 2025 lineup changes before proceeding
- [ ] **MATCH MODELS** — ensure Mac and Alienware have same model downloads
- [ ] Fix TODO in p1_eval_harness.py:200 — re-enable online requirement
- [ ] Download updated models to GPU server (Ministral-3-8B, Llama-3.2-3B, Gemma-3-12B)
- [ ] Verify T4/T5 task files on GPU server
- [ ] Start P1 eval baseline runs (T1-T5, all 4 models)

### Training Runs (GPU Server — RTX 4090)
- [ ] P1 baseline eval: 4 models × 5 tiers = 20 runs
- [ ] P2 training: SLO-aware GRPO on Qwen3-4B-Instruct (500 steps)
- [ ] P2 training: SLO-aware GRPO on Ministral-3-8B (500 steps)
- [ ] P2 training: SLO-aware GRPO on Llama-3.2-3B (500 steps)
- [ ] P2 training: SLO-aware GRPO on Gemma-3-12B (500 steps)
- [ ] Post-training eval: repeat T1-T5 on all trained checkpoints

### Paper Updates (after results)
- [ ] Paper 1: add T4 (BFCL) and T5 (SWE-bench) results to tables
- [ ] Paper 2: update training curves with new runs
- [ ] Generate fresh figures from new data

## Active
- [x] Fix duplicated "Request:" in first 16 T3 tasks
- [x] Wire up T4 (BFCL v4) — 500 tasks from Berkeley Function Calling benchmark
- [x] Wire up T5 (SWE-bench Lite) — 300 tasks from Princeton SWE benchmark
- [x] Update t_suite.md documentation with T1-T5 progression

## Done
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
- 2026-01-13 (late): Selected 4-model lineup for papers: Qwen3-4B (Alibaba/China), Mistral-7B (Mistral/France), Llama-3.1-8B (Meta/US), Gemma-2-9B (Google/US). Rationale: geographic diversity, company diversity, 4B→9B size ladder, all trainable on single 4090.
- 2026-01-13 (evening): Added T4 (BFCL v4, 500 tasks) and T5 (SWE-bench Lite, 300 tasks) to complete T1-T5 progression. Updated t_suite.md. Smoke-tested eval pipeline with LM Studio @ 10.0.0.63. Ready for tomorrow.
- 2026-01-13: reset plan file to keep ongoing progress and keep top-level focused.
- 2026-01-13: archived legacy modules and older papers/results; added dependency notes; refreshed docs/CI paths.
- 2026-01-13: removed `docs/ONE_PAGER.md` and added follow-up cleanup considerations.
- 2026-01-13: archived top-level notebooks/data/figures, moved tooling into `scripts/`, and removed codex-named files from the root.
- 2026-01-13: enabled git-derived versioning via `setuptools-scm` in `pyproject.toml`.

## Notes
- Dependency sources: `pyproject.toml` is the package manifest; `requirements.txt` and `requirements-dev.txt` are for pip installs; `environment.yml` is for conda/mamba.
