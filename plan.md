# Plan — Fix P1 & P2 This Week (LM Studio @ 10.0.0.72:1234, Qwen)

**Objective:** Produce **final, logged results** for **P1** (baseline SLO/stability) and **P2** (multi‑objective reward frontiers) using **LM Studio** at **http://10.0.0.72:1234** with a **Qwen Instruct** model. Refresh figures and text for both papers, and rebuild arXiv source bundles.

**Repo:** `agent-stable-slo` (stable) · **Provider:** `lmstudio` (remote) · **Model:** `Qwen2.5-7B-Instruct` (or whatever `/v1/models` lists)  
**Logging:** **W&B** required; **Weave** optional (no‑fail stubs present)

---

## Environment (use in every shell)

```bash
# Base env (Mac or Ubuntu is fine) — remote LM Studio + Qwen on 10.0.0.72
export AOFW_PROVIDER=lmstudio
export OPENAI_API_BASE=http://10.0.0.72:1234/v1
export OPENAI_API_KEY=lm-studio
export LMSTUDIO_MODEL=Qwen2.5-7B-Instruct
export MAX_THOUGHT_TOKENS=196

# W&B (required)
wandb login
export WANDB_PROJECT=agent-stable-slo
export WANDB_DIR=$(pwd)/wandb_logs
# optional: export WANDB_ENTITY=<your_wandb_username_or_org>

# (Optional) Weave
# pip install weave
# export WEAVE_PROJECT=agent-stable-slo
```

**Connectivity check**  
```bash
curl -s http://10.0.0.72:1234/v1/models | jq -r '.data[].id' | head
```

---

## Deliverables (by Friday)

- **P1** figures regenerated from real runs:  
  `latency_hist.png`, `qps_vs_p95.png`, `success_at_slo.png`
- **P2** figure regenerated:  
  `pareto.png` (from λ/μ sweeps; γ optional)
- **Text updates** (P1 & P2) reflecting measured p95/p99/TTFT, success@SLO, and stability
- **arXiv** sources rebuilt: `dist/p1_arxiv_src.zip`, `dist/p2_arxiv_src.zip`

---

## Today (Mon Nov 24) — P1 data & figures (remote LM Studio)

**Run P1: baseline → QPS → stability → aggregate → figures**
```bash
cd ~/projects/agent-stable-slo
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# ensure env vars from section above are exported
make p1-mac
make bench-mac
make stability-mac
make agg figs
```

**Edit P1 text (15–30 min)**
- Insert **p95/p99/avg TTFT** (from `out/aggregate/eval_summary.csv`)
- Insert **p95 @ QPS 1/4/8** (from `out/*bench_q*.txt`, Fig. qps_vs_p95)
- Insert **stability** (median/worst `disagreement_rate` from `out/*stability*.jsonl`)

**Checkpoint**
- W&B shows three runs (baseline, bench, stability)
- Figures under `papers/P1_stable_slo/arxiv/figs/` replaced (no placeholders)

---

## Tomorrow (Tue Nov 25) — P2 λ/μ sweeps (remote LM Studio)

```bash
source .venv/bin/activate
# env vars from top of file must be set
make sweeps          # small grid over (lambda, mu), γ=0
make agg figs        # regenerates Pareto
```

**Edit P2 (15–30 min)**
- Add `papers/P2_reward_stability/arxiv/figs/pareto.png`
- Discuss the **tradeoff** as λ or μ increase
- Choose **operating point** (λ*, μ*) that meets success@SLO target with minimal p95

---

## Wednesday (Wed Nov 26) — γ stability (optional) & polish

**Optional γ sweeps**
```bash
# edit scripts/run_sweeps.py -> GAMMAS = [0.0, 0.1, 0.2]
python scripts/run_sweeps.py
python tools/aggregate_eval.py --glob "out/sweeps/**/eval.jsonl" --out out/aggregate/p2_sweeps_eval.csv
python figures/generate_all.py
```

**Finalize text**
- Short “stability vs latency” paragraph in P2
- Quick Mac vs remote comparison note in P1 (if you also run locally)

---

## Friday (Fri Nov 28) — package & submit

```bash
python tools/build_arxiv_sources.py
ls dist/
# p1_arxiv_src.zip  p2_arxiv_src.zip
```

*(Optional)* Compile PDFs locally:
```bash
cd papers/P1_stable_slo/arxiv && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd ../../P2_reward_stability/arxiv && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Research Tasks (parallel work, short blocks)

- **Lit refresh (copy notes to refs.bib):**
  - RL for LLM agents (GRPO/ARA/REINFORCE variants, stability‑aware rewards)
  - Structured decoding & schema control (json‑schema, constrained decoding)
  - Tail latency & SLO engineering (“Tail at Scale” & follow‑ups)
- **Ablations to queue (can run next week if time runs out):**
  - `MAX_THOUGHT_TOKENS` ∈ {96, 128, 196}
  - Qwen vs a second local model (if time): `llama-3.1-8b-instruct`
  - Streaming on/off (impact on TTFT vs overall latency)
- **Writing improvements:**
  - Add a 1‑paragraph “deployment playbook” in P2 (how to pick λ, μ, γ in production)
  - Tighten threats‑to‑validity (local hardware, schema strictness differences, determinism)


Great—here’s a **single, Codex‑ready `PLAN.md`** that covers:

* The **overall 6‑paper portfolio** (purpose, scope, artifacts, seed bibliographies).
* A **concrete plan for P1 & P2** (commands, DoD, text updates).
* A **detailed schedule for this week** (Mon–Fri).
* A **Codex iteration loop** and **quality gates** to keep changes crisp and reversible.

You can **download it** or copy it inline below.

---

## ⬇️ Downloads

* **PLAN.md** (complete):
  **[plan_complete_codex_v1.md](sandbox:/mnt/data/plan_complete_codex_v1.md)**

* **ZIP (just the PLAN.md)**:
  **[plan_complete_codex_v1.zip](sandbox:/mnt/data/plan_complete_codex_v1.zip)**

---

## 📄 PLAN.md (full text)

````markdown
# PLAN.md — Portfolio & Weekly Execution (Codex‑ready)

**Owner:** Mike Maloney (Neuralift • UNH)  
**Repo:** `agent-stable-slo` (stable mono‑repo) • **Runtime:** LM Studio (OpenAI server) @ `http://10.0.0.72:1234` (Qwen Instruct)  
**Logging:** Weights & Biases (required) • Weave (optional)

---

## 0) Executive Summary

- We are shipping a 6‑paper portfolio on **single‑GPU, SLO‑aware, stability‑aware agents** with **open‑weights**.  
- **This week’s focus:** finalize **P1** and **P2** with runs on **LM Studio remote @ 10.0.0.72** using **Qwen**; regenerate figures; lock text; prepare arXiv bundles.  
- All experiments are **reproducible** via `make` targets, log to **W&B**, and write figures directly into each paper’s `arxiv/figs/` directory for zero‑diff LaTeX.

---

## 1) Repository Contracts (what Codex can assume)

- Repo **name never changes**: `agent-stable-slo`. Version bump via `VERSION` file.
- **Providers:** `lmstudio` (preferred), `ollama`, `hf` — switch by `AOFW_PROVIDER`.
- **Remote LM Studio default for this plan:** `OPENAI_API_BASE=http://10.0.0.72:1234/v1`, `LMSTUDIO_MODEL=Qwen2.5-7B-Instruct`.
- **W&B always on** when `WANDB_PROJECT=agent-stable-slo` is set. `WANDB_MODE=online` preferred.
- Figures always regenerate to `papers/<P*>/arxiv/figs/*.png`; arXiv bundles built from `tools/build_arxiv_sources.py`.

**Quick env (per shell):**
```bash
# Base repo env (OK anywhere), then override to remote LM Studio:
source scripts/env_mac.sh
export OPENAI_API_BASE=http://10.0.0.72:1234/v1
export LMSTUDIO_MODEL="Qwen2.5-7B-Instruct"
export AOFW_PROVIDER=lmstudio
export MAX_THOUGHT_TOKENS=196

wandb login
export WANDB_PROJECT=agent-stable-slo
export WANDB_DIR=$(pwd)/wandb_logs
# Optional Weave:
# export WEAVE_PROJECT=agent-stable-slo

# Sanity
curl -s http://10.0.0.72:1234/v1/models | jq -r '.data[].id' | head
````

---

## 2) Portfolio Map (6 Papers) — Purpose, Scope, Artifacts, Seed Bib

Below are the six papers with **one‑paragraph purpose**, **scope/method**, **expected artifacts**, **target venues**, and **seed bibliography** (to expand in `refs.bib`).

### P1 — *Stable, SLO‑Aware Tool‑Using Agents on a Single GPU*

**Why it matters:** Production agents are judged by **tail latency (p95/p99)**, **contract adherence (JSON schema/tooling)**, and **stability**. P1 establishes a **replicable baseline** on commodity hardware (Mac + 4090) with **LM Studio/Ollama**.
**Scope/Method:** Deterministic decoding with `response_format=json_schema`; benchmark tail latency with a QPS sweep; compute success@SLO; measure disagreement across repeated runs.
**Artifacts:** Latency histogram, QPS→p95 curve, success@SLO curves, stability table.
**Venue target:** JOSS short paper or arXiv systems note + reproducibility package.
**Seed bib:** `tail_at_scale` (tails); `jsonschema` (contracts); `trl` (RL infra mention); `qlora` (efficient finetune); `wandb` (logging); `lmstudio`, `ollama` (runtimes).

### P2 — *Rewarding Stability: Multi‑Objective RL for SLOs*

**Why:** Tuning only for accuracy ignores latency, token budget, and answer drift. P2 turns these into **first‑class objectives** with a composite reward and explores frontiers.
**Scope/Method:** Reward `R = R_schema + R_succ − λL − μC − γD`; sweep (λ, μ) and optionally γ; plot Pareto (success vs p95, stability).
**Artifacts:** Pareto plot, operating‑point selection guide, W&B sweep cards.
**Venue:** PeerJ CS or arXiv ML systems.
**Seed bib:** `tail_at_scale`; `trl`; PPO/REINFORCE (policy gradient); `qlora`; W&B sweeps docs.

### P3 — *Spec‑Constrained Decoding for Agents (JSON/K/V Contracts)*

**Why:** Contract violations dominate downstream failures. P3 shows **schema‑aware decoding** reduces invalids without large latency cost and pairs with tool contracts.
**Scope/Method:** Compare strict `json_schema` vs looser decoding; measure validity%, latency deltas, and downstream tool success.
**Artifacts:** Validity vs latency tradeoff plots; ablation on schema complexity.
**Seed bib:** JSON Schema; constrained decoding techniques; structured prompting.

### P4 — *Single‑GPU Decode Scheduling & KV Policies for Tail Control*

**Why:** On a single GPU, **KV cache** and **token scheduling** drive tail behavior under load. P4 evaluates **decode policies** under fixed QPS.
**Scope/Method:** Simulate bursts; vary max_new_tokens and scheduling; measure p95/p99 and throughput.
**Artifacts:** p95/p99 vs policy curves; scheduler pseudo‑code.
**Seed bib:** Tail latency, CUDA/GPU scheduling notes, KV‑cache efficiency papers.

### P5 — *Budgeted Self‑Consistency vs Stability and Accuracy*

**Why:** Multiple‑sample voting can help accuracy but may **hurt tails**. P5 tests **N-sample self‑consistency** under a latency/token budget.
**Scope/Method:** Vary N and aggregation; plot success gains against p95 and cost.
**Artifacts:** Accuracy‑vs‑p95 curves at fixed budgets.
**Seed bib:** Self‑consistency; token/latency budgeting literature.

### P6 — *Operationalizing Agents: SLO Playbook for Real Workflows*

**Why:** Engineering teams need a **deployment guide** that ties model, decoding, and reward settings to **business SLOs**.
**Scope/Method:** Case‑study pipelines (e.g., contract triage, incident summaries) with chosen operating points and rollback levers.
**Artifacts:** End‑to‑end diagrams; SLO runbooks; before/after metrics.
**Seed bib:** Observability, reliability engineering, AIOps/LLMOps playbooks, W&B/Weave best practices.

> **Note:** P3–P6 can leverage data produced while doing P1/P2; keep runs and configs tagged in W&B to reuse.

---

## 3) Immediate Plan for P1 (this week)

### Goals (Definition of Done)

* Regenerate **latency hist**, **QPS→p95**, **success@SLO** on **10.0.0.72 (Qwen)**.
* Run **stability harness (N=20)**; report median + worst disagreement.
* Update P1 LaTeX with measured numbers and figures; produce `dist/p1_arxiv_src.zip`.

### Commands

```bash
# Env (every shell)
source scripts/env_mac.sh
export OPENAI_API_BASE=http://10.0.0.72:1234/v1
export LMSTUDIO_MODEL="Qwen2.5-7B-Instruct"
export AOFW_PROVIDER=lmstudio
wandb login; export WANDB_PROJECT=agent-stable-slo; export WANDB_DIR=$(pwd)/wandb_logs

# Run
make p1-mac
make bench-mac
make stability-mac
make agg figs

# Package (optional now; mandatory Friday)
python tools/build_arxiv_sources.py
```

### Text updates (P1)

* Report **p95/p99** and **avg TTFT** from `out/aggregate/eval_summary.csv`.
* Report **p95 at QPS 1/4/8** from `out/*bench_q*.txt`.
* Report **stability** median/worst from `out/*stability*.jsonl`.
* Short paragraph: “**SLO policy** — choose threshold, measure success@SLO; prefer stable operating points even at a slight latency cost.”

---

## 4) Immediate Plan for P2 (this week)

### Goals (Definition of Done)

* Run **λ/μ sweeps (γ=0)** on 10.0.0.72 and generate **Pareto**.
* Optionally add **γ stability** sweep; capture movement in disagreement rate.
* Update P2 LaTeX with figure and chosen (λ*, μ*, γ*); produce `dist/p2_arxiv_src.zip`.

### Commands

```bash
# Env as above (LM Studio remote)
make sweeps
make agg figs

# Optional stability weight
# edit scripts/run_sweeps.py: GAMMAS = [0.0, 0.1, 0.2]
python scripts/run_sweeps.py
python tools/aggregate_eval.py --glob "out/sweeps/**/eval.jsonl" --out out/aggregate/p2_sweeps_eval.csv
python figures/generate_all.py
```

### Text updates (P2)

* Insert `figs/pareto.png`; narrate **tradeoff** as λ or μ increase.
* Choose **operating point** subject to success@SLO(min) and stability(max), minimizing p95.
* Add short **deployment recipe**: how to pick λ/μ/γ for a new task.

---

## 5) Bibliography Seeds (copy into refs.bib and expand)

Below are **starter BibTeX entries** you can paste into each paper’s `refs.bib` (extend as you add lit).

```bibtex
@article{tail_at_scale, 
  title={The Tail at Scale}, author={Jeffrey Dean and Luiz André Barroso}, 
  journal={Communications of the ACM}, year={2013}}

@misc{jsonschema, title={JSON Schema}, howpublished={\\url{https://json-schema.org/}}}

@misc{trl, title={TRL: Transformer Reinforcement Learning}, howpublished={\\url{https://github.com/huggingface/trl}}}

@misc{qlora, title={QLoRA: Efficient Finetuning of Quantized LLMs}, author={Tim Dettmers and et al.}, 
  year={2023}, howpublished={\\url{https://arxiv.org/abs/2305.14314}}}

@misc{lmstudio, title={LM Studio}, howpublished={\\url{https://lmstudio.ai}}}
@misc{ollama, title={Ollama}, howpublished={\\url{https://ollama.com}}}

@misc{wandb, title={Weights \\& Biases}, author={Lukas Biewald}, year={2020}, howpublished={\\url{https://wandb.ai}}}

@inproceedings{schulman2017ppo, 
  title={Proximal Policy Optimization Algorithms}, author={John Schulman and et al.}, 
  booktitle={arXiv preprint arXiv:1707.06347}, year={2017}}

@article{williams1992reinforce, 
  title={Simple statistical gradient-following algorithms for connectionist reinforcement learning}, 
  author={Ronald J. Williams}, journal={Machine Learning}, year={1992}}
```

> **How to expand**: For each paper, add 3–6 domain‑specific citations (e.g., constrained decoding variants for P3; KV cache scheduling for P4; self‑consistency for P5; LLMOps case studies for P6).

---

## 6) This Week’s Detailed Schedule

**Mon Nov 24 — P1 end‑to‑end (LM Studio remote)**

* [ ] Configure env; verify `curl /v1/models` shows Qwen
* [ ] `make p1-mac && make bench-mac && make stability-mac && make agg figs`
* [ ] Update P1 LaTeX with numbers + figs; brief interpretation paragraph
* [ ] W&B check: runs exist, summaries show p95/p99; tag run “p1-remote-qwen”

**Tue Nov 25 — P2 λ/μ sweeps (γ=0)**

* [ ] `make sweeps && make agg figs`
* [ ] Insert Pareto into P2; pick (λ*, μ*) and justify
* [ ] W&B check: sweeps grouped, chart pinned

**Wed Nov 26 — γ & polish**

* [ ] Optional: set `GAMMAS=[0.0,0.1,0.2]` → rerun `scripts/run_sweeps.py`
* [ ] Update P2 text (stability); second pass on P1

**Thu Nov 27 — Buffer / small ablation**

* [ ] Optional ablation: `MAX_THOUGHT_TOKENS ∈ {128,196,256}` → note p95 impact

**Fri Nov 28 — Package & commit**

* [ ] `python tools/build_arxiv_sources.py` → `dist/p1_arxiv_src.zip`, `dist/p2_arxiv_src.zip`
* [ ] `git add -A && git commit -m "finalize P1/P2 (LM Studio 10.0.0.72)" && git tag v0.4.1`

---

## 7) Codex Iteration Loop (ULTRATHINK)

1. **Plan → Code**: Open a small issue with the intended change, acceptance criteria, and rollback.
2. **Minimal diff**: Prefer editing `scripts/run_sweeps.py` or environment only; avoid structural churn.
3. **Run**: Use profile prelude then a single make target, e.g.

   ```bash
   bash -lc "source scripts/codex_profile.sh && make p1-mac"
   ```
4. **Verify**: W&B run artifacts present; `out/aggregate/*.csv` updated; figures regenerated.
5. **Refine**: Tighten thresholds, re‑run quick (shorter steps), commit with clear message.
6. **Quality gates**: `ruff check .`, `black --check .` (add tests later).

---

## 8) Risks & Rollback

* **Remote endpoint issues** → fallback to local LM Studio or Ubuntu + Ollama provider; re‑run P1/P2 quickly.
* **Schema adherence** → keep temperature 0; if invalid %, insert a terse “Return JSON only” preface; consider 2‑pass parsing.
* **Latency drift** → reduce `MAX_THOUGHT_TOKENS`; limit QPS; warm the server before bench.
* **W&B outages** → `export WANDB_MODE=offline` then `wandb sync` later.

---

## 9) Ownership & Tags

* W&B project: `agent-stable-slo` • Suggested tags: `remote-lmstudio`, `qwen`, `p1`, `p2`, `gamma-0`, `gamma-sweep`, `max-tok-196`
* Git tagging: `v0.4.1` for this week’s finalized P1/P2; increment minor for figure or text updates.

---

*End of PLAN.md*

```

---

If you want me to **inject this PLAN.md into your repo ZIP** (and bump to `v0.4.1`), I can ship that immediately—just say “add to repo”.
::contentReference[oaicite:0]{index=0}
```
