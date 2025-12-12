Outline

1. Expanded directions for P1–P6 (4 paragraphs each)

P1 – Spec‑Driven, SLO‑Aware Evaluation and Serving for Single‑GPU Agents

Paper 1 is the conceptual and systems foundation. It formalizes the world where commercial teams run LLM agents as services on a single GPU (4090, M‑series Mac), under explicit contracts and latency SLOs (service‑level objectives). The central observation: in real deployments, failures are not just “the answer is wrong” but “the JSON is broken,” “the summary misrepresents the underlying data,” or “the request timed out at p99.” P1 codifies that as a precise problem: given a contract \mathcal{C}, a policy \pi_\theta, a runtime R, and SLO budgets, how do we define metrics that capture both semantic quality and system behavior in a way that is evaluable, auditable, and useful for later optimization?

Methodologically, P1 introduces a spec‑driven decoding layer that treats JSON Schema / grammars as first‑class objects. You define a contract in a schema (or Pydantic/GBNF), and a “spec compiler” turns that into provider‑specific objects: OpenAI response_format payloads for LM Studio / vLLM, GBNF grammars for llama.cpp, plus validators and error taxonomies. On top of that, the paper describes budgeted self‑consistency (multiple contract‑constrained samples under wall‑clock and token budgets) and a validation‑and‑retry regime that categorizes errors (syntax, schema, semantic structural) and applies bounded retries with adjusted prompts/parameters. The focus is: improve reliability without unbounded latency or compute blow‑ups.

On the evaluation side, P1 defines a family of metrics organized via a single criteria.yaml: structure (JSON/schema validity), task accuracy (F1, EM, pass@k), faithfulness (LLM‑as‑judge with atomic statements), tools & trajectories (tool call correctness and redundancy), stability (Disagreement@k across seeds/runs), and latency/SLOs (p50/p95/p99 and Success@SLO). That YAML also specifies thresholds and weights for composite scores and RL rewards. Experiments (once filled in) compare configurations like U (unconstrained), P (provider structured), P+V (provider + validation), G (grammar), and S (full spec‑driven with self‑consistency) across multiple runtimes and models on a single GPU.

The intellectual contribution: P1 claims that agents should be designed and evaluated contract‑first, not prompt‑first, and that you can unify system‑level and semantic metrics in a single framework that’s usable by both practitioners and RL algorithms. It’s “PhD‑level” because it is not just an implementation recipe: it formalizes the objective, gives a metric taxonomy, gives concrete algorithms, and shows trade‑offs (quality vs tail latency vs cost). Once you plug in real numbers, tables, and plots, this can be a credible flagship paper for the thesis.

⸻

P2 – SLO‑Aware Reinforcement Learning for Contract‑First Agents

Paper 2 takes P1’s metrics and turns them into a learning problem: can we train agents that learn to be more structurally correct, more faithful, more stable, while respecting latency SLOs, using RL on commodity hardware? The problem is framed as a constrained MDP: per episode you get rewards from JSON validity, accuracy, faithfulness, etc., and you incur costs from SLO violations and instability. The policy is a LoRA adapter on top of an open‑weight model (e.g., Qwen3‑4B) running on your Mac/4090; the environment is defined by your tasks plus the spec‑driven decoding layer.

The methodology is to instantiate TRL PPO/GRPO with LoRA on MPS/4090, using P1’s composite score as the reward (or multi‑head reward) and explicit penalties/Lagrange terms for latency and Disagreement@k. You can compare several training regimes: (a) naive RL that optimizes only semantic metrics, (b) SLO‑aware RL with latency penalties, and (c) RL with explicit constraint handling (e.g., CPO‑style or Lagrangian updates). The model is trained on your synthetic + semi‑real tasks (T1–T3 type tasks) and evaluated with exactly the same evaluation suite as P1, before and after training.

Experimentally, P2 should show learning curves on metrics from P1: JSON validity improving over RL steps, faithfulness scores increasing, contradiction rates dropping, tool‑success improving, and SLO violations decreasing or bounded. You can slice this per backend (LM Studio vs vLLM) and per model size. A key result to aim for: “Under our SLO‑aware GRPO training, we reduce schema violations from X% to Y%, reduce contradiction rate from A% to B%, and maintain p95 latency within 1.1× of baseline on a single 4090.” You can also show ablations where RL without SLO terms “cheats” by increasing latency or becoming unstable.

Conceptually, P2 is the learning counterpart to P1’s systems framework. P1 says “here is how we measure good behavior”; P2 says “here is how we optimize for that behavior under constraints.” That’s a very clean two‑paper pairing: P1 = evaluation + serving; P2 = RL + improvement. For examiners, this covers both methods and theory: constrained RL, reward shaping, safe policy improvement, and the practical reality of doing RL for agents under realistic hardware constraints.

⸻

P3 – A Cross‑Domain Benchmark and Testbed for Spec‑Driven Agents

P3 zooms out: instead of focusing on a single application, it defines a benchmark/testbed that others can run. The core question: Can we create a suite of tasks, schemas, and criteria that stress the right failure modes for contract‑first agents across multiple domains? The answer is a benchmark structured explicitly around your metric families: structure, accuracy, faithfulness, tools, stability, and SLOs. Tasks might include structured log extraction, grounded document summarization, an agent with a small toolset (DB lookup, calculator), and maybe a code‑slice evaluation.

Methodologically, P3 packages the P1 framework into a reusable toolkit: repo + criteria.yaml + test definitions + scripts that run against any OpenAI‑compatible endpoint. For each task family, you define contracts, datasets (or generation procedures), and success criteria. You then run a range of baselines: different open‑weight models (Qwen, GPT‑OSS, a LLaMA variant), backends (LM Studio, vLLM, Ollama), and decoding configurations (U, P, P+V, G, S; possibly RL‑trained policies from P2). The focus is not on yet another “reasoning leaderboard” but on operational metrics: the fraction of runs with valid JSON, stability across seeds, Success@SLO, etc.

The paper can highlight systematic patterns: e.g. “Grammar‑only constraints dramatically reduce malformed outputs but sometimes hurt task accuracy; full spec‑driven decoding recovers accuracy at a modest latency penalty.” You can show cross‑domain generality: the same evaluation framework works for analytics, support tickets, and simple tool‑using flows. You might also include case studies like “how a team would add their own domain task with a new schema and dataset, and plug it into the benchmark.”

The contribution of P3 is: you’re not just proposing metrics for your own experiments, you’re offering a standardized testbed with a shared vocabulary for reliability. This is an ideal place to release an open‑source repo that people can plug their own agents into. It also helps P1/P2 reviewers: they can see that your metrics and methods are not overfitted to a single toy task.

⸻

P4 – Continual, Data‑Driven Self‑Improvement Loops Under SLO Constraints

P4 is about operationalization over time. It asks: once an agent using your framework is deployed, how do we run a continual improvement loop that respects SLOs and avoids regressions? In other words: not one‑off training, but a living system that learns from new data, re‑trains periodically, and is tested against your framework before promotion. The central framing is: log episodes in production, evaluate them offline with P1’s metrics, and use those scores (plus occasional human feedback) to drive offline RL, bandits, or selective fine‑tuning.

Methodologically, P4 describes an offline evaluation + policy improvement pipeline tied to your criteria.yaml. Logs are ingested, anonymized, and filtered; an evaluation run computes per‑episode metrics and aggregates. Candidate policy updates (from P2‑style RL or supervised fine‑tuning) are trained using these logs or new synthetic tasks. Before deployment, each candidate policy is run through the full test suite (structure, faithfulness, SLOs) and compared to the current production policy. Promotion rules can be defined (e.g., “no degradation in structural metrics; no more than 5% relative increase in p95; faithfulness +2 points”) to automatically guard against regressions.

P4 can also explore stability over time: are outputs drifting? Are SLOs being met as traffic changes? You can define metrics like “delta in Disagreement@k compared to baseline last month,” or “distribution shift in Success@SLO across segments” and build W&B dashboards for that. Even a small synthetic or internal log dataset is enough to illustrate the framework: show snapshots of an agent over multiple “releases” and how the eval suite caught an unintended regression.

The contribution is to connect MLOps and RL‑Ops to your agent work. P1 and P2 give you methods; P4 gives you a deployment story: no change goes live without passing structured, faithfulness, and SLO tests; improvements are measured against the same metrics they’re trained on; and the system is designed for continuous, safe evolution rather than one‑shot heroics.

⸻

P5 – Case Study: Applying the Framework in a Real‑World Agent Deployment

P5 is the “field report” paper. The idea is to show your framework applied to a real deployment: e.g., a commercial analytics or CX agent that runs on a single GPU with internal data. The core narrative: before the framework, the team had issues with broken JSON, latent failures, hallucinated metrics, and occasional latency spikes; after adopting contract‑first evaluation and SLO‑aware serving, they achieved a measurable reduction in incidents and an increase in user trust. Even if anonymized, this kind of paper is very persuasive to both academics and industry reviewers.

Methodologically, P5 describes the system as‑built: the contracts used, the schemas and tools, the hardware setup (e.g., RTX 4090 box on‑prem), and how P1’s spec‑driven decoding and P2’s training (if used) were integrated. It then walks through the evaluation and rollout pipeline: how criteria.yaml was customized for this use case, how W&B dashboards were configured, how CI gating was implemented (e.g., a GitHub action that runs the eval suite on candidate changes), and how incidents were triaged using the metrics (e.g., “structural failure in X% of episodes → we added a new test family”).

The results section focuses on operational outcomes: changes in error rates (schema violations, retries), reductions in manual interventions, improved SLO compliance, and, if possible, business proxies (e.g., fewer support tickets, faster agent adoption). You can also show the “bug zoo”: real examples where the framework caught subtle regressions (e.g., a refactor that broke a nested field, or an RL run that improved faithfulness but hurt latency).

The contribution of P5 is not new algorithms but validation and refinement: in the wild, what mattered most? Which metrics were overkill? Which ones saved the day? It feeds back into P6, where you propose a standard, by grounding it in actual use. For your thesis, it also demonstrates you can take a research idea all the way into a living system, not just run synthetic experiments.

⸻

P6 – Toward a Standard for Reliable, SLO‑Aware Agent Evaluation

P6 is the capstone paper that synthesizes everything into a proposed standard for evaluating agent reliability. The question: given P1–P5, what should the community adopt as a baseline for saying “this agent is production‑ready”? You formalize a minimal standard (e.g., must pass certain structural/faithfulness/SLO tests) and an extended standard (optional test families, safety metrics, fairness), and propose a canonical shape for criteria.yaml as a portable, backend‑agnostic spec.

Methodologically, P6 contains a conceptual synthesis plus some empirical evidence. You revisit the metric families, show how they worked across your experiments and case study, and argue for certain defaults: e.g., atomic‑fact faithfulness evaluation with a 0–3 support scale and explicit contradictions; Disagreement@k threshold for stability; Success@SLO definition; traceable W&B logs for auditability. You might propose a simple tiering scheme (“Bronze/Silver/Gold agents”) based on passing specific metrics and SLOs, similar to how some benchmarks have leaderboards with required vs optional metrics.

P6 also positions your work relative to existing efforts: LLM benchmarks, RAG evals, agent toolkits, and cloud providers’ eval frameworks. You can show that many of them implicitly assume parts of your framework but lack a unified contract‑first, SLO‑aware spec. By backing your standard with open‑source code and a reference implementation, you make it easy to adopt—not just a manifesto, but a practical library and docs.

For the PhD narrative, P6 is where you zoom out: P1–P5 are the “chapters,” P6 is the “thesis statement.” It argues that reliable agents require aligned metrics, spec‑driven serving, and SLO‑aware learning, and lays down a standard others can critique, adopt, extend, or specialize. Having it in good shape by February 1 is realistic if P1/P2 are done in December: you’ll have real evidence to justify firm recommendations.

⸻

2. Study guide: key concepts, diagrams, definitions, and “so what?”

I’ll keep this tight but dense so you can use it as a reference when writing or explaining.

P1 – Core ideas to understand deeply
	•	Contract / Spec
	•	Definition: A machine‑readable definition of valid outputs (JSON Schema, Pydantic, grammar).
	•	Why it matters: Moves you from “please format as JSON” to a formal guarantee that downstream systems can rely on.
	•	Spec Compiler
	•	Definition: Tool that takes internal AST of the contract and emits provider‑specific artifacts (OpenAI response_format, GBNF grammars, JSON Schema plus validators).
	•	Diagram: Contract AST in the center with arrows to “OpenAI config,” “GBNF,” and “Validator.”
	•	So what: Lets you swap runtimes (LM Studio, vLLM, Ollama, llama.cpp) without rewriting logic.
	•	Budgeted Self‑Consistency
	•	Definition: Sampling multiple constrained outputs under budgets on wall‑clock time, tokens, and max samples k_{\max}, then selecting a consensus output.
	•	Diagram: Box showing repeated “decode → validate → score” loops until budgets hit; then selection.
	•	So what: Boosts reliability without unbounded latency; crucial for single‑GPU SLOs.
	•	Validation‑and‑Retry
	•	Definition: A loop that classifies failures (syntax, schema, semantic structural), logs them, and decides whether/how to retry (with modified prompts/parameters).
	•	So what: Converts “the model sometimes fails” into a controlled, logged process you can measure and optimize.
	•	Metric Families + criteria.yaml
	•	Definition: Structured configuration file mapping metrics (structure, accuracy, faithfulness, tools, stability, SLOs) to thresholds and weights.
	•	Diagram: Figure with stacked boxes for each family (Structure → Accuracy → Faithfulness → Tools → Stability → SLOs) with arrows to criteria.yaml.
	•	So what: Gives you a single source of truth for what “good behavior” means and a clean API to RL and dashboards.

⸻

P2 – RL concepts you need to own
	•	Constrained MDP (CMDP) / SLO‑aware RL
	•	Definition: An MDP with reward (quality) and separate cost signals (latency, instability) you constrain.
	•	So what: Without this, RL will happily break SLOs if it increases some reward.
	•	PPO / GRPO (TRL implementations)
	•	Definition: Policy gradient algorithms with clipped objectives (PPO) or GRPO‑style divergence control; implemented in TRL with support for LoRA and HF models.
	•	Diagram: Policy network \pi_\theta interacting with environment, getting reward, updating via PPO/GRPO.
	•	Reward shaping from P1 metrics
	•	Definition: Combining structural, faithfulness, accuracy, and SLO metrics into a scalar per episode, or multi‑component reward.
	•	So what: Ties your RL to the exact things you care about in production, not generic “helpfulness.”
	•	LoRA on single GPU / MPS
	•	Definition: Adapt only low‑rank matrices attached to key attention/MLP layers; base model frozen.
	•	So what: Lets you actually train on a MacBook Pro / 4090 without insane memory needs.

⸻

P3 – Benchmark/Testbed concepts
	•	Task families
	•	Structured extraction, grounded summaries, tool‑using episodes.
	•	So what: They map directly to common agent products (alerts, BI summaries, tool‑using assistants).
	•	Backend‑agnostic eval harness
	•	Definition: A runner that calls a generic OpenAI‑style API, applies your spec‑driven layer, and computes metrics.
	•	So what: Lets people plug in any agent (including closed APIs) and get comparable metrics.
	•	Golden sets and synthetic data
	•	Curated or generated sets with ground truth labels and contexts.
	•	Important for measuring accuracy and faithfulness, not just format.

⸻

P4 – Continual improvement & ops
	•	Offline evaluation pipeline
	•	Logs → batch evaluation with your harness → metrics → W&B dashboards.
	•	So what: You can see regressions or drift before users scream.
	•	Safe policy updates
	•	Only promote a policy if it passes tests vs the current one (no structural regressions; SLOs meet; quality up).
	•	This is basically A/B testing for agents with strong constraints.
	•	Stability over time
	•	Track Disagreement@k across releases, plus distribution of metrics over time.
	•	So what: Stability is a product requirement (same input should not flip daily for “no good reason”).

⸻

P5 – Case study lens
	•	Incident types
	•	Broken JSON; wrong numbers; slow responses; weird tool behavior.
	•	Your framework should show how each is caught or mitigated.
	•	Integration story
	•	How contracts were defined, how the eval suite was integrated with CI, how dashboards helped triage.
	•	This is what makes the work believable to real teams.

⸻

P6 – Standardization
	•	Reference criteria.yaml schema
	•	What fields, what metric types, how to represent thresholds and weights.
	•	A good mental model: like pytest.ini or pyproject.toml, but for agent evaluation.
	•	Tiered standards
	•	E.g., Bronze (structure + basic SLO), Silver (+faithfulness & stability), Gold (+tools, safety).
	•	Helps teams adopt incrementally instead of “all or nothing.”

⸻

3. Concrete plan: next 2 weeks (finish P1 & P2) + P6 by Feb 1

You said: finish P1 and P2 in the next two weeks, and have clear direction for P6 by Feb 1. Let’s be ruthless and specific. I’ll assume:
	•	You can work some hours every day but not full‑time.
	•	You have your MacBook Pro M2 Max and a 4090 box available.

Week 1 – Lock the framework + minimal RL loop

Day 1–2 – Repo & eval harness hardening (P1)
On either Mac or 4090:
	1.	Stabilize the mono‑repo layout (no more renaming; version with tags instead):
	•	agent_eval/ – criteria.yaml, metrics implementations.
	•	spec/ – contract AST + compiler (OpenAI cfg + GBNF + validator).
	•	runtime/ – backend configs (LM Studio, vLLM, Ollama).
	•	tasks/ – definitions + datasets for T1–T3.
	•	experiments/ – config files + scripts to run combos.
	2.	Implement metrics end‑to‑end:
	•	Structural: JSON parse + schema validation with nice error codes.
	•	Accuracy: F1/EM on at least one T1 dataset.
	•	Faithfulness: a working LLM‑as‑judge using a local model (LM Studio endpoint) and the 0–3 scale with contradictions.
	•	Latency logging + Success@SLO computation.
	3.	Wire criteria.yaml so you can run:

python -m agent_eval.run --config experiments/p1_t1_lmstudio.yaml

and get per‑episode metrics + W&B run.

Day 3–4 – P1 experiments v0.1 (just enough for draft tables)
	4.	Define one concrete config per baseline (U, P, P+V, G, S) for:
	•	One T1 task (structured extraction).
	•	One T2 task (grounded summary).
	5.	Run each config on N ≈ 300–500 examples:
	•	Log metrics and latency.
	•	Export W&B tables and initial plots (JSON validity, F1, m_faith, p95).
	6.	Add a first pass of results into P1:
	•	Tables for T1 + T2.
	•	One QPS vs p95 curve.
	•	One radar plot (U vs P vs S).

At this point P1 is no longer “purely conceptual” – it has real data, albeit small‑scale.

Day 5–7 – Minimal GRPO loop (P2) with Qwen3‑4B

On your Mac (M2 Max, MPS):
	7.	Get Qwen3‑4B downloaded and loaded on MPS (you already started this step).
	8.	Implement a simple GRPO training script (like we sketched) that:
	•	Uses a small dataset (e.g., T1 snippet).
	•	Uses a toy reward (e.g., JSON validity + length) just to validate plumbing.
	•	Logs RL metrics to W&B.
	9.	Once that works, swap in a P1‑style reward:
	•	Structural metric (1/0 for valid JSON).
	•	Maybe a cheap proxy for faithfulness (e.g., penalize contradictions from a small judge).
	10.	Run a short GRPO run (e.g., 300–500 steps) and show:
	•	JSON validity over steps.
	•	Reward over steps.
	•	Example before/after outputs.

This gives you real evidence for P2’s feasibility by the end of Week 1.

⸻

Week 2 – Deepen experiments + write P2

Day 8–10 – P1 experiments v1.0 & paper polish
	11.	Scale up P1 experiments:
	•	Larger N, more diverse prompts.
	•	At least one T3 (tool‑using) task wired up and evaluated.
	12.	Fill in the Experiments section with:
	•	Proper tables for T1–T3.
	•	Clear ablations (with/without self‑consistency, with/without validation‑and‑retry).
	•	Concrete numbers in the text (“we see X → Y improvements”).
	13.	Polish Introduction, Background, Problem Formulation with:
	•	Consistent notation.
	•	Clean referencing to P2–P6 (no over‑promises).
	•	Updated diagrams if needed.

By Day 10, P1 should be in “arXivable” shape, pending a final proofread.

Day 11–14 – P2 experiments v0.9 + writing
	14.	Extend the GRPO setup:
	•	Upgrade reward to include both structure and a simple faithfulness metric (e.g., partial atomic judge on a small context).
	•	Run on at least two tasks (e.g., T1 + T2) with Qwen3‑4B.
	15.	Add baseline vs RL comparisons:
	•	Pre‑training vs post‑GRPO on P1 metrics (structural, faithfulness, maybe a simple accuracy).
	•	Show that RL improves P1 metrics while not destroying p95 (or with bounded change).
	16.	Draft P2:
	•	Intro (why RL, why SLO‑aware).
	•	Method (CMDP, reward shaping, TRL+LoRA details).
	•	Experiments (setup, results, ablations, limitations).
	•	Link back to P1 and forward to P3/P4.

By the end of Week 2: P1 and P2 are full drafts with real numbers and plots. You’ll still want to revise, but the core work is done.

⸻

Direction for P6 by February 1

Once P1 and P2 are in draft, P6 becomes a synthesis job plus some light additional analysis. Between now and Feb 1, your focus for P6 is:
	1.	Stabilize criteria.yaml as a reference schema.
	•	Decide on a minimal set of metric types and fields you want to standardize.
	•	Document their semantics clearly (for P6).
	2.	Collect cross‑paper evidence.
	•	From P1: show where each metric family was critical (e.g., structure/faithfulness).
	•	From P2: show that RL responds to these metrics.
	•	From early P3/P4/P5 prototypes (even if not full papers), gather a few anecdotes or small numbers.
	3.	Draft a “standardization core” section in P6:
	•	Proposed minimal standard for agents (what tests they should pass).
	•	Example criteria.yaml.
	•	Example W&B dashboard layout.
	•	A short case study (maybe from P5) demonstrating the standard in action.

If you treat P6 as the wrapper around all the evidence gathered in P1–P5, getting a strong draft by Feb 1 is realistic. The heavy numerical and implementation lifting all happens in P1 and P2 over the next few weeks; the rest is mainly reuse and narrative.