# P1-P6 Consistency Checklist

Last updated: 2026-02-09
Owner: series editor
Scope: P1, P2, P3, P4, P5, P6 + top-level repo narrative docs

## 1) Series Status Alignment
- [ ] P5 is described consistently as either `study design / projected` or `results`, never both.
- [ ] P6 statements about P5 match P5's actual status.
- [ ] README paper-status table matches current paper directories and draft maturity.
- [ ] `papers/future_papers_outline.md` status language matches current manuscripts.

## 2) Core Numeric Claims (Single Source of Truth)
- [ ] Choose one source-of-truth artifact for P1 headline metrics and lock it in:
  - `out/p1_comprehensive_20260118/all_results.json`
- [ ] Reconcile and align all headline values across P1/P3/P6 + README:
  - Spearman rho (and sign)
  - Success@SLO percentages
  - Evaluation count totals
- [ ] Ensure all rate metrics explicitly state whether they are percentages or proportions.
- [ ] Ensure every paper that cites rho also cites CI and p-value with the same numbers.

## 3) Terminology and Definitions
- [ ] Use one canonical metric name format everywhere (`Success@SLO` or `S@SLO`).
- [ ] Ensure tier definitions are identical across papers:
  - Interactive: 2s
  - Standard: 5s
  - Batch: 30s
- [ ] Ensure task naming (T1-T5 and variants) is consistent across P1-P6.

## 4) Evidence vs Projection Labeling
- [ ] Every projected quantity in P5 is explicitly labeled as projected/hypothesized.
- [ ] Every P6 statement that depends on P5 production evidence is phrased conditionally unless P5 results exist.
- [ ] Remove any language that implies completed production validation if P5 is still design-only.

## 5) Methods and Dataset Consistency
- [ ] Per-request vs percentile-estimated S@SLO methodology is described consistently where referenced.
- [ ] Model counts and run counts are harmonized (13 eval models, 11 trainable models, 185 runs, etc.).
- [ ] Hardware context (RTX 4090 single-GPU) is stated the same way across P1-P4.

## 6) Citation and Cross-Reference Hygiene
- [ ] Paper-series forward references are chronological and non-contradictory.
- [ ] Citations to companion papers use correct status tags (`published`, `in prep`, `forthcoming`).
- [ ] Section references resolve correctly in each manuscript.

## 7) Repo Narrative and CLI Consistency
- [ ] README command examples match actual CLI flags (e.g., `--base-model` vs `--model`).
- [ ] Benchmark docs mention the actual task file used for T4 (`tasks/t4_bfcl.jsonl`).
- [ ] `papers/REVISION_CHECKLIST.md` "all passed" claims reflect current manuscript state.

## 8) Final Pre-Submission Gate
- [ ] Regenerate all paper tables/figures from current result artifacts.
- [ ] Re-run benchmark/statistics scripts and confirm manuscript numbers.
- [ ] Spot-check 10 random numeric claims against source files.
- [ ] Freeze a series snapshot commit hash and include it in all six papers.

