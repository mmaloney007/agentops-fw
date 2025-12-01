# PLAN.md — RL/SLO Portfolio & Weekly Focus (LM Studio @ 10.0.0.63)

**Owner:** Mike Maloney (Neuralift • UNH)  
**Repo:** `agent-stable-slo` (mono-repo) • **Runtime:** LM Studio (OpenAI API) @ `http://10.0.0.63:1234` (Qwen3-4B-Thinking)  
**Logging:** W&B required; Weave optional

---

## 0) This Week (keep at top)

- **P1**: rerun baseline → QPS sweep → stability harness on **10.0.0.63**; regenerate figs; add a short **“where we’re going” intro** that previews the RL-heavy roadmap (P2–P6).  
- **P2**: run **λ/μ sweeps (γ optional)** on 10.0.0.63, refresh Pareto, pick operating point.  
- **Writing**: update P1/P2 LaTeX with fresh numbers/figs and forward-looking framing; ensure RL emphasis is clear for academic + industry readers.  
- **Packaging**: rebuild arXiv bundles for P1/P2; tag W&B runs (`p1-remote-qwen`, `p2-remote-qwen`).  
- **Sanity**: keep everything reproducible from the **mamba base/agent-slo** env; no virtualenvs.

---

## 1) Environment (mamba-first)

```bash
# Activate (prefers agent-slo; base is fine if already provisioned)
source activate_mamba.sh base   # or: source activate_mamba.sh agent-slo

# Remote LM Studio @ 10.0.0.63 (default); optional 10.0.0.72 with \`openai/gpt-oss-20b\`
export AOFW_PROVIDER=lmstudio
export OPENAI_API_BASE=http://10.0.0.63:1234/v1
export OPENAI_API_KEY=lm-studio
export LMSTUDIO_MODEL="qwen/qwen3-4b-thinking-2507"
export MAX_THOUGHT_TOKENS=196

# Observability
wandb login
export WANDB_PROJECT=agent-stable-slo
export WANDB_ENTITY=mike007
export WANDB_DIR=$(pwd)/wandb_logs
# optional: export WEAVE_PROJECT=agent-stable-slo

# Connectivity check
curl -s http://10.0.0.63:1234/v1/models | jq -r '.data[].id' | head
```

> Shortcuts: `source scripts/env_lm_remote.sh` or `source scripts/codex_profile.sh` set the same defaults for automation. Use **mamba**; avoid ad-hoc venvs.

---

## 2) Portfolio (6 papers, RL-forward & PhD-worthy)

- **P1 — Stable, SLO-grounded tool-using agents on a single GPU**  
  Deterministic decoding + schema contracts; tail latency + success@SLO + disagreement. **Include a “roadmap” paragraph** in the intro that frames P2–P6 as the RL and deployment arc built on this baseline.
- **P2 — Multi-objective RL for SLOs & stability (reward shaping)**  
  Reward `R = R_schema + R_succ − λL − μC − γD`; sweep (λ, μ, γ) to chart Pareto fronts and operating points.
- **P3 — Spec-constrained decoding & RL-aware control**  
  Schema-constrained vs looser decoding; quantify validity vs latency; show how reward shaping leverages constraint scores.
- **P4 — Single-GPU decode scheduling & KV policies**  
  Scheduling/KV cache strategies under fixed QPS; p95/p99 + throughput; connects to RL by informing reward penalties for tail events.
- **P5 — Budgeted self-consistency vs stability/latency**  
  N-sample voting under token/latency budgets; analyze when extra samples help or hurt; tie to reward terms for cost.
- **P6 — Operational SLO playbook (industry-ready)**  
  End-to-end deployment recipes; rollback levers; mapping λ/μ/γ and decode settings to business SLOs with observability hooks.

**Uniqueness:** Open-weight, single-GPU, **RL-first SLO program** that unifies schema adherence, stability, and tail latency. Data and configs are re-used across papers; P1/P2 generate artifacts that seed P3–P6. Academic rigor (metrics, ablations, bib) + industry readiness (W&B lineage, playbooks, rollback).

---

## 3) P1 — Immediate Execution

**Definition of Done**
- Fresh runs on **10.0.0.63 (Qwen)**: latency hist, QPS→p95, success@SLO, stability (N=20).  
- Figures refreshed in `papers/p1_arxiv_src/figs/`; LaTeX updated with p95/p99/TTFT, QPS numbers, stability stats.  
- Add **“Where we’re going”** intro paragraph linking to P2–P6 RL path and why stability baseline is the anchor.

**Commands**
```bash
make p1-mac
make bench-mac
make stability-mac
make agg figs
python tools/build_arxiv_sources.py  # optional mid-week; mandatory Fri
```

**Text updates**
- p95/p99/avg TTFT from `out/aggregate/eval_summary.csv`.  
- p95 @ QPS 1/4/8 from `out/*bench_q*.txt`.  
- Stability median/worst from `out/*stability*.jsonl`.  
- One-paragraph forward look: how this baseline enables RL reward shaping (P2) and constraint/control studies (P3–P6).

---

## 4) P2 — Immediate Execution

**Definition of Done**
- λ/μ sweeps (γ=0 baseline; optional γ∈{0.0,0.1,0.2}) on 10.0.0.63 with Pareto plot.  
- Operating point (λ*, μ*, γ*) chosen for success@SLO target with minimal p95 and low disagreement.  
- LaTeX updated; `papers/p2_arxiv_src/figs/pareto.png` refreshed; arXiv bundle rebuilt.

**Commands**
```bash
SWEEP_STEPS=40 make sweeps   # override if you want shorter runs (default 400)
make agg figs
# Optional γ sweep
# edit scripts/run_sweeps.py -> GAMMAS = [0.0, 0.1, 0.2]
python scripts/run_sweeps.py
python tools/aggregate_eval.py --glob "out/sweeps/**/eval.jsonl" --out out/aggregate/p2_sweeps_eval.csv
python figures/generate_all.py
python tools/build_arxiv_sources.py  # when ready to package
```

**Text updates**
- Insert Pareto; discuss tradeoff as λ/μ rise.  
- Note stability movement if γ>0.  
- Deployment recipe: pick λ/μ to meet contract success + p95 budget; add γ if disagreement drifts.

---

## 5) Bibliography Seeds (extend per paper)

`tail_at_scale`, `jsonschema`, `trl`, `qlora`, `lmstudio`, `ollama`, `wandb`, `schulman2017ppo`, `williams1992reinforce` (see refs.bib for full entries; expand with constrained decoding, KV scheduling, self-consistency, and LLMOps case studies).

---

## 6) Weekly Schedule (Mon–Fri)

- **Mon**: Env check (curl 10.0.0.63), run `make p1-mac && make bench-mac && make stability-mac && make agg figs`, update P1 text + roadmap paragraph.  
- **Tue**: `make sweeps && make agg figs`; insert Pareto + operating point in P2.  
- **Wed**: Optional γ sweep; tighten P2 text; second pass on P1 figures/wording.  
- **Thu**: Optional ablation (MAX_THOUGHT_TOKENS {128,196,256}); doc polish.  
- **Fri**: `python tools/build_arxiv_sources.py`; check figs/zips; stage commits/tags.

---

## 7) Iteration Loop & Quality Gates

1. Plan → small diff → run one target.  
2. Verify outputs (`out/aggregate/*.csv`, `papers/*/figs/*.png`, W&B runs tagged).  
3. Refine thresholds; rerun if needed.  
4. Gates: `ruff check .`, `black --check .`, `pytest -q` (even for doc edits if time allows).

---

## 8) Risks & Rollback

- **Remote endpoint flaky** → fall back to local LM Studio/Ollama; rerun quick baselines.  
- **Invalid JSON / schema drift** → keep temp=0; add “return JSON only” preface; 2-pass parsing.  
- **Latency drift** → reduce `MAX_THOUGHT_TOKENS`, lower QPS, warm server.  
- **W&B outage** → `WANDB_MODE=offline`, then `wandb sync` later.

---

## 9) Ownership & Tags

- W&B project: `agent-stable-slo`; tags: `p1`, `p2`, `remote-10-0-0-63`, `qwen`, `lambda-mu-sweep`, `gamma-sweep`, `max-tok-196`.  
- Git: tag when P1/P2 bundles ready (e.g., `v0.4.1` after this week’s refresh).
