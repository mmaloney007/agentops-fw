# Paper drop-in replacements (P1 + P2)

This folder contains updated, drop-in LaTeX sources for:

- `papers/P1_stable_slo/arxiv/main.tex`
- `papers/P1_stable_slo/arxiv/refs.bib`
- `papers/P2_reward_stability/arxiv/main.tex`
- `papers/P2_reward_stability/arxiv/refs.bib`

## What changed (high level)

### P1 (`P1_stable_slo`)
- Rewrote the **Roadmap** paragraph to match the P1–P6 plan (P3 benchmark, P4 continual loop, P5 case study, P6 standard).
- Rewrote the **Contributions** bullets to:
  - stay contract-first + SLO-aware,
  - explicitly state the eval-as-reward link to P2,
  - avoid overclaiming and frame the 250+ suite as the scaling target.
- Rewrote **Section 5.3 Tasks and Datasets** to be explicit and citable:
  - T1 = CLINC150 / OOS intent classification
  - T2 = HotpotQA distractor (grounded QA + structured summarization)
  - T3 = tool-using episodes with schema-validated tool args
- Rewrote **Section 5.5 Metrics and Reporting** to remove speculative language and state that:
  - W&B logging is **online**
  - figures/tables are exported from pinned artifacts
- Rewrote **Sections 5.6–5.8** to remove “we expect …” language:
  - kept pilot measurements as pilot measurements
  - clarified what the full measurement grid is and how it is produced
- Added `\label{sec:faithfulness}` for consistent referencing.
- Moved bibliography to the end (standard arXiv layout).
- Cleaned **refs.bib**:
  - replaced “Anonymous” entries with real author/title metadata where possible
  - replaced the “atomic facts” reference with **FActScore** (more appropriate for the described method)
  - updated/cleaned keys and removed unused entries by filtering to cited keys

### P2 (`P2_reward_stability`)
- Replaced the placeholder with a full draft paper:
  - CMDP framing + SLO/stability costs
  - PPO/GRPO + LoRA/QLoRA implementation details
  - reward design section mapping P1 metrics -> RL reward
  - system section (single GPU + endpoints + online W&B)
  - related work and appendix checklist
- Replaced refs with a minimal, correct bib set (and fixed the GRPO citation).

## How to compile

From each paper directory:

```bash
cd papers/P1_stable_slo/arxiv
latexmk -pdf main.tex

cd ../../P2_reward_stability/arxiv
latexmk -pdf main.tex
```

If you do not have `latexmk`, run `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.

## How to “fix the rest” (to finalize camera-ready)

1. **Freeze the measurement grid**
   - Decide the exact U / P / P+V / G / S configs for each backend and model.
   - Pin *everything* (dataset JSONL, schema versions, criteria.yaml) as W&B artifacts.

2. **Run the full evaluation matrix**
   - For each model/backend/config:
     - run deterministic pass
     - run Disagreement@k stability pass
     - run concurrency sweep
   - Export: CSV + LaTeX tables + latency CDF plots.

3. **Replace pilot text with final tables**
   - Convert the pilot-only paragraphs in P1/P2 into final tables/figures.
   - Add ablation tables:
     - validation-only vs grammar-only vs spec-driven
     - self-consistency budget sweeps (k vs Success@SLO)
     - retry policy variants

4. **W&B must be online**
   - Ensure `WANDB_MODE=online` everywhere.
   - Store run group names in the paper (or artifact IDs) so reviewers can reproduce.

5. **Tighten claims**
   - Every quantitative claim in Intro/Abstract should correspond to a table/figure.
   - Keep “future work” only for truly unmeasured extensions.

