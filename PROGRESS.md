# PROGRESS.md - Checkpoint Tracker

**Last Updated**: 2026-02-04 20:00

## P1 Evaluation Results (13 Models × 5 Task Types)

| Model | Size | Vendor | T1 | T2 | T3 | T4 | T5 | Avg Lat | P95 | Success@SLO |
|-------|------|--------|----|----|----|----|----|---------|----|-------------|
| llama-3.2-1b | 1B | Meta | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| llama-3.2-3b | 3B | Meta | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| qwen2.5-3b | 3B | Alibaba | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| phi-3-mini | 3.8B | Microsoft | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| qwen3-4b | 4B | Alibaba | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| yi-1.5-6b | 6B | 01.AI | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| mistral-7b-v0.3 | 7B | Mistral | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| falcon-mamba-7b | 7B | TII | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| gpt-oss-20b | 20B | OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| ministral-8b | 8B | Mistral | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| llama-3.1-8b | 8B | Meta | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| gemma-2-9b | 9B | Google | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |
| gemma-3-12b | 12B | Google | ✅ | ✅ | ✅ | ✅ | ✅ | - | 0 | - |

**Progress**: 65/65 task combinations (100.0%)

---

## P3 + P4 Paper Drafts (2026-02-04)

### P3: AgentSLO-Bench Paper
- **Status**: First draft complete
- **Location**: `papers/P3_benchmark/arxiv/main.tex`
- **Content**: 7 sections, all tables populated from P1 data, 3 SLO tiers
- **Code**: `agent_stable_slo/bench/` (slo_tiers.py, benchmark_runner.py, leaderboard.py, cli.py)
- **CLI**: `agentslo-bench baseline|leaderboard|run` verified working
- **Key finding**: Spearman rho = +0.09 at 2s tier (accuracy vs S@SLO uncorrelated)

### P4: Training Dynamics Paper
- **Status**: First draft complete
- **Location**: `papers/P4_training_dynamics/arxiv/main.tex`
- **Content**: 9 sections, 4 RQs, tables from real analysis data
- **Analysis scripts**: `scripts/p4_analysis/` (5 scripts, all verified)
- **Results**: 63 training curves classified, 82.5% early prediction accuracy
- **Key findings**:
  - Reward decomposition: latency penalty grows from 0.13 (1B) to 0.59 (4B Qwen3)
  - Forgetting: 4 robust, 2 selective, 4 catastrophic profiles
  - Curve taxonomy: 41 sustained, 7 transient, 15 flat (63 total; 36/2/15 single-task only)
  - Early prediction: 81% LOO-CV accuracy from first 50 steps

### Files Created (24 new files)
- P3 paper: main.tex, refs.bib, Makefile
- P3 code: slo_tiers.py, benchmark_runner.py, leaderboard.py, cli.py
- P4 paper: main.tex, refs.bib, Makefile
- P4 scripts: reward_decomposition.py, forgetting_matrix.py, curve_taxonomy.py, early_prediction.py, generate_p4_figures.py
- P4 results: reward_decomposition.json, forgetting_matrix.json, curve_taxonomy.json, early_prediction.json
- P4 pgfplots: p4_reward_decomp.dat, p4_forgetting.dat, p4_curves.dat, p4_reward_components.dat, p4_forgetting_heatmap.dat, p4_curve_scatter.dat, p4_size_taxonomy.dat, 3 sample curve .dat files

### Quality Pass 1: Data Accuracy (2026-02-04)
- **P1 fixes**: Caption line 1024 (26% → 44.9% latency overhead), Section→Figure ref line 1366
- **P2**: Clean, 0 issues found
- **P3**: All tables verified against `all_results.json`
- **P4 fixes**:
  - Reward decomposition table: all 11 models corrected to match `reward_decomposition.json`
  - Forgetting profiles: Gemma-2-9B reclassified robust→selective (δ=-0.068)
  - Profile counts: 5/1/4 → 4/2/4 (robust/selective/catastrophic)
  - Curve taxonomy table: corrected to single-task counts (36/2/15 = 53 runs)
  - Abstract curve counts per model: corrected all 10 rows
  - Early prediction section: replaced placeholder with exact results (82.5%/81.0%, weights, per-model)
  - Prose references: updated all specific numbers (Rl, Rs, R̄) to match JSON

### Quality Pass 2: Diagrams and Figures (2026-02-04)
- **P1**: 11 TikZ figures all syntactically correct; fixed missing `tab:main-results-1s` ref (line 958)
- **P1**: Fixed bib access date (2025-12-22 → 2025-01-18)
- **P2**: 2 figures clean, no issues
- **P3**: Added TikZ rank inversion diagram (Figure 1) replacing TODO
- **P4**: Added pgfplots curve shapes figure (Figure 1) replacing TODO

### Quality Pass 3: Cross-Paper Consistency (2026-02-04)
- **P1**: Fixed "six tasks" → "five tasks" (abstract, intro, conclusion)
- **P1**: Fixed "42,900" → "29,900" and "3,300" → "2,300" (evaluation counts)
- **P1**: Fixed abstract table header "Accuracy" → "JSON Valid" (was showing JSON validity rates)
- **P2**: Standardized author block to match P1/P3/P4 format
- **Verified consistent**: Model names, Spearman rho (+0.09), 13 eval/11 train model counts, self-citations

### Recommendation Implementation (2026-02-04 evening)

Implemented 6 critique recommendations:

**1. Real SLO tiers from per-request data**
- New script: `scripts/p3_analysis/compute_real_tiers.py`
- Processed 42,900 predictions from `out/p1_full_eval/`
- Fixed T4 scoring (gold format mismatch → function-name matching)
- Output: `results/p3_analysis/real_slo_tiers.json` + 4 pgfplots .dat files
- Key finding: Spearman rho = +0.17 (2s, p=0.57), -0.17 (5s, p=0.57), +0.02 (30s, p=0.94) — all non-significant, all CIs span zero

**2. P4 threshold sensitivity analysis**
- New script: `scripts/p4_analysis/threshold_sensitivity.py`
- Taxonomy is 100% stable from sustained=40% to 65%, 94.3% stable at ±10%
- Added Section 6.1 to P4 paper with sensitivity table

**3. P3 paper updated with real data**
- Replaced all tables with per-request S@SLO (no extrapolation)
- Updated abstract, intro, all 4 tables, analysis section
- Added Figure 2 (accuracy vs S@SLO scatter), Figure 3 (latency-accuracy tradeoff)
- Added threshold sensitivity subsection, per-request methodology discussion
- Updated evaluation count: 3,300 per model (42,900 total)
- Spearman now reported with bootstrapped 95% CI and p-values

**4. P1 tone adjustments**
- Abstract: removed "dirty secret" / "devastating" language
- Intro: removed "gut punch", "This isn't a hypothetical. It's Tuesday."
- Conclusion: "selecting for failure" → "insufficient for deployment decisions"

**5. P1 ↔ P3 reconciled**
- P1 keeps 29,900 from comprehensive dataset
- P3 uses 42,900 from full_eval with per-request data
- P3 explicitly explains the extension: "We extend the evaluation protocol from Paper I with a larger prompt set (3,300 per model vs. 2,300)"

**6. Code updates**
- `benchmark_runner.py`: added `compute_from_predictions()` for per-request S@SLO
- `leaderboard.py`: added `bootstrap_spearman()` with 10K resamples, 95% CI, p-value

---

## Run Log
