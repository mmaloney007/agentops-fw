# Future Papers: P3 through P6

**Last updated**: February 4, 2026
**Status**: Post-P1/P2 results, incorporating empirical findings
**Prior outline**: `archive/papers/Outline.md` (conceptual, pre-results)

---

## P3: AgentSLO-Bench: A Deployment-First Benchmark for Production Agent Evaluation

### Thesis

Current LLM benchmarks (MMLU, HumanEval, HELM, Chatbot Arena) measure capability but not deployability. Teams use these rankings to select models, then discover in production that their "best" model fails 40-99% of requests due to latency, malformed output, or hallucination. AgentSLO-Bench is the first benchmark that ranks models by Success@SLO---the metric that actually predicts production readiness.

### Motivation and Context

Paper 1 demonstrated that accuracy rankings *inversely* predict deployment success: across 13 models from 8 vendors and 4 countries, the Spearman correlation between accuracy rank and Success@SLO rank is rho = -0.82. Llama-3.2-1B (the smallest, least accurate model) achieves 49.1% Success@SLO while Gemma-3-12B (the most accurate) achieves 0.0% under a 2-second interactive SLO. This isn't a quirk of our setup---it's a systematic property of how model size trades off against latency. AgentSLO-Bench operationalizes this finding as a community resource.

### Benchmark Design

The benchmark comprises 5 task families (2,500+ examples total): T1 (intent classification, CLINC-150), T2 (grounded QA, HotpotQA), T3 (tool calling), T4 (function routing, BFCL), and T5 (code patching, SWE-bench). Each task includes schema contracts, ground-truth labels, and faithfulness contexts. Evaluation uses the lexicographic metric stack from P1: structure > accuracy > faithfulness > stability > latency > Success@SLO. Three SLO tiers---Interactive (2s), Standard (5s), Batch (30s)---let teams evaluate models against their actual deployment constraints.

The benchmark ships as a pip-installable package with a CLI (`agentslo-bench run --model <endpoint> --tier interactive`) that works against any OpenAI-compatible API. Results are uploaded to a public leaderboard that shows both the traditional accuracy ranking and the Success@SLO ranking side by side, making the deployment gap visible at a glance. We include baseline results for all 13 models from P1, establishing reference points across the full 1B-20B range.

### Paper Structure

1. **Introduction**: The benchmark gap---why existing benchmarks mislead deployment teams
2. **Related Work**: HELM, MMLU, function-calling benchmarks, Chatbot Arena; what they measure vs. what deployment needs
3. **Benchmark Design**: Tasks, metrics, SLO tiers, submission protocol
4. **Baseline Results**: 13-model evaluation across all tiers, highlighting rank inversions
5. **Analysis**: What predicts Success@SLO? (Architecture > raw accuracy; latency is the dominant factor under tight SLOs)
6. **Case Studies**: Adding a custom domain task; evaluating a closed-source API; comparing serving backends
7. **Discussion**: Limitations (hardware-specific SLOs, task coverage), community adoption path
8. **Reproducibility**: Single-GPU setup, open-source toolkit, automated validation

### Key Contributions

- First deployment-oriented benchmark with tiered SLO evaluation
- Public leaderboard showing accuracy rank vs. deployment rank side by side
- Open-source evaluation toolkit compatible with any OpenAI-style endpoint
- Baseline results across 13 models establishing the deployment gap as a measurable, reproducible phenomenon
- Extensible framework: teams can add domain-specific tasks with schema contracts

### Target Venue

MLSys 2027 (systems focus), NeurIPS 2026 Datasets & Benchmarks track, or EMNLP 2026 (if framed around NLP tasks). Strong fit for industry workshops (Efficient ML, Foundation Models in the Wild).

---

## P4: Training Dynamics and Reward Decomposition in Schema-Aware RL

### Thesis

Paper 2 established that a capacity threshold (~9B parameters) separates models that can learn structured output through RL from those that cannot. But *why* does this threshold exist? Paper 4 investigates the training dynamics in detail: reward decomposition (which reward components drive learning vs. which cause interference), forgetting analysis (how multi-task training degrades single-task performance), and the learn-then-forget pattern that characterizes sub-threshold models.

### Motivation and Context

The P2 training matrix (11 models x 6 tasks x 3 seeds x 1000 steps = 185 completed runs) generated rich step-by-step training logs with per-step rewards, JSON validity, schema compliance, latency, and token counts. This data reveals phenomena that P2's summary statistics obscure: (1) The "learn-then-forget" curve where sub-9B models peak at steps 100-200 then regress to baseline; (2) Task-dependent thresholds where tool-calling (T3/T4) has a lower capacity requirement (~3B) than grounded QA (T2, ~9B); (3) Multi-task interference where Mixed training achieves only 52.1% validity vs. 99.8% for single-task T1; (4) Reward component interactions where schema rewards and accuracy rewards sometimes compete.

### Research Questions

This paper addresses four specific questions that emerged from P2:
- **RQ1 (Reward Decomposition)**: Which of the 6 reward components (schema, accuracy, faithfulness, latency, cost, stability) contribute most to learning, and do any actively interfere with each other? We ablate each component individually and in pairs.
- **RQ2 (Forgetting Analysis)**: When training on Mixed (all tasks), which single-task capabilities degrade? Is forgetting proportional to task complexity or model size? We compare Mixed performance against individual task baselines.
- **RQ3 (Training Curves)**: What is the characteristic shape of reward curves for learners vs. non-learners? Can we predict at step 50 whether a model will sustain learning through step 1000?
- **RQ4 (Threshold Mechanism)**: Is the capacity threshold about representational bandwidth (the model can't simultaneously maintain format + content + coherence) or optimization dynamics (gradient signal is too noisy for small models)?

### Experimental Design

We reuse the P2 training infrastructure (GRPO + LoRA on RTX 4090) with targeted ablation runs. For RQ1, we train Gemma-2-9B (a reliable learner) with each reward component zeroed out, measuring the effect on all other metrics. For RQ2, we compare Mixed-trained models against single-task models on held-out evaluation sets for each task. For RQ3, we fit learning curves to all 185 runs and extract characteristic features (peak step, decay rate, final plateau). For RQ4, we examine gradient statistics (norm, variance, direction) across model sizes to test whether small models receive noisier signals.

The paper produces: (a) Reward decomposition heatmaps showing component contributions, (b) Forgetting matrices (model x task x metric), (c) Learning curve taxonomy (sustained, transient, flat), (d) Gradient analysis across the size spectrum.

### Paper Structure

1. **Introduction**: Beyond summary statistics---what training dynamics reveal about RL for structured output
2. **Background**: GRPO, multi-objective RL, catastrophic forgetting in continual learning
3. **Experimental Setup**: Reusing P2 infrastructure, 185 completed runs + targeted ablations
4. **Reward Decomposition** (RQ1): Ablation results, component interaction matrices
5. **Forgetting Analysis** (RQ2): Mixed vs. single-task, per-task degradation curves
6. **Training Curve Taxonomy** (RQ3): Characteristic shapes, early prediction of learning outcome
7. **Threshold Mechanism** (RQ4): Gradient analysis, representational bandwidth hypothesis
8. **Discussion**: Implications for curriculum design, reward engineering, model selection
9. **Conclusion**: Practical guidelines for practitioners choosing models and reward structures

### Key Contributions

- First systematic reward decomposition for multi-component RL training of LLM agents
- Forgetting analysis quantifying multi-task interference in schema-aware training
- Training curve taxonomy enabling early termination of non-viable training runs
- Evidence for the mechanism behind the capacity threshold (representational bandwidth vs. optimization dynamics)

### Training Budget

Approximately 50-80 additional training runs on RTX 4090:
- 6 reward ablations x 3 seeds x 2 model sizes = 36 runs
- Gradient logging on existing runs (reanalysis, no new training needed for some)
- Extended 2000-step runs for convergence verification = 6-12 runs

### Target Venue

ICML 2026 or NeurIPS 2026 main track (RL + LLM training dynamics). Could also target ICLR 2027.

---

## P5: Closing the Gap: Production Deployment of SLO-Aware Agents

### Thesis

Papers 1-4 establish the deployment gap, demonstrate training solutions, and analyze dynamics in controlled experiments. Paper 5 is the field report: applying the full framework to a real-world agent deployment, measuring operational outcomes before and after adoption, and identifying which components of the framework matter most in practice.

### Motivation and Context

The credibility gap in ML research is that controlled experiments rarely transfer cleanly to production. Real deployments face challenges that benchmarks don't capture: changing data distributions, infrastructure failures, team workflow integration, monitoring fatigue, and the gap between "the model improved on our metrics" and "the team trusts the model more." P5 bridges this gap by deploying the contract-first, SLO-aware framework in a real system and reporting honestly on what worked, what didn't, and what surprised us.

### Deployment Context

The target deployment is a commercial analytics/CX agent running on commodity hardware (RTX 4090 or equivalent). The agent handles structured tasks: generating JSON reports from operational data, summarizing incidents, routing tool calls, and answering grounded questions. Before the framework, common failure modes include: malformed JSON crashing downstream pipelines (estimated 5-15% of requests), hallucinated metrics in summaries, inconsistent outputs across identical requests, and occasional latency spikes causing timeout errors.

The integration story covers: (1) Defining contracts using JSON Schema/Pydantic for each agent endpoint, (2) Deploying the spec-driven decoding layer from P1, (3) Setting up criteria.yaml with task-specific thresholds and weights, (4) Configuring W&B dashboards for real-time monitoring of structure, faithfulness, stability, and SLO compliance, (5) Implementing CI gating (GitHub Actions running the evaluation suite on candidate model changes), and (6) Optional: deploying a GRPO-trained adapter from P2 to improve specific failure modes.

### Results and Analysis

The results section focuses on operational metrics over a multi-week deployment window: reduction in schema violation incidents, decrease in manual interventions, improved SLO compliance rates, and (where measurable) business proxies such as fewer support tickets or faster agent adoption. We also present the "bug zoo"---real examples where the framework caught subtle regressions that would have reached production: a refactor that broke a nested field, a model update that improved accuracy but degraded latency, an RL training run that improved faithfulness but introduced instability.

Critically, P5 reports on what didn't work or needed adjustment: which metrics were overkill for this use case, which thresholds needed tuning, how the team's workflow adapted (or resisted) the framework. This honest assessment feeds directly into P6's standardization recommendations.

### Paper Structure

1. **Introduction**: The gap between research evaluation and production deployment
2. **System Description**: The agent, its tasks, the hardware, the team
3. **Framework Integration**: Contract-first design, spec-driven decoding, evaluation pipeline, monitoring
4. **Evaluation Protocol**: Before/after comparison, A/B deployment, metric selection
5. **Results**: Operational metrics, incident reduction, SLO compliance, business outcomes
6. **Bug Zoo**: Case studies of regressions caught by the framework
7. **Lessons Learned**: What worked, what didn't, what surprised us
8. **Discussion**: Generalizability, team adoption challenges, cost of framework overhead
9. **Conclusion**: Practical recommendations for teams adopting contract-first evaluation

### Key Contributions

- First end-to-end deployment report of SLO-aware agent evaluation in production
- Quantified operational improvements (schema violations, SLO compliance, incident rates)
- Honest assessment of framework overhead vs. benefit
- Practical integration patterns (CI gating, monitoring dashboards, incident triage)
- Evidence base for P6's standardization recommendations

### Target Venue

AAAI 2027 (applications track), MLSys 2027, or ACL 2026/2027 Industry Track. Strong fit for practitioner-oriented conferences.

---

## P6: Toward a Standard for Production-Ready Agent Evaluation

### Thesis

Papers 1-5 progressively build evidence that production agent reliability requires more than accuracy benchmarks. Paper 6 synthesizes this evidence into a proposed standard: a minimal, portable specification for evaluating whether an agent is "production-ready." The standard defines metric families, threshold defaults, and a tiered certification scheme (Bronze/Silver/Gold) that teams can adopt incrementally.

### Motivation and Context

The current state of LLM agent evaluation is fragmented. MMLU measures knowledge, HumanEval measures coding ability, Chatbot Arena measures preference, but none measure whether an agent will actually work in production. Meanwhile, every company building agents is reinventing the same metrics: "does the JSON parse?", "is the answer grounded?", "does it respond in time?" There is no shared vocabulary, no portable format, and no community agreement on what "production-ready" means.

P6 addresses this by proposing `criteria.yaml` as a reference specification---analogous to `pyproject.toml` for Python packages or `openapi.yaml` for REST APIs. The specification defines six metric families (structure, accuracy, faithfulness, tools, stability, SLOs), their measurement protocols, default thresholds, and composition rules. It is designed to be backend-agnostic (works with any OpenAI-compatible endpoint), task-agnostic (teams define their own contracts and datasets), and incrementally adoptable (you don't need all six families on day one).

### Tiered Certification

The standard proposes three tiers of production readiness:

- **Bronze**: Structure + basic SLO. The agent produces valid JSON/schema output for >= 95% of requests and meets latency SLOs. This is the minimum bar---most agents should achieve this with spec-driven decoding alone.
- **Silver**: Bronze + faithfulness + stability. The agent's outputs are grounded in provided context (faithfulness >= 0.8) and consistent across identical requests (disagreement@3 <= 0.15). This requires either careful prompt engineering or RL training.
- **Gold**: Silver + tools + advanced SLOs. The agent correctly uses tools, handles multi-step reasoning, and meets stringent SLO targets (p99 under deadline, not just p95). This represents a fully production-hardened agent.

These tiers are backed by empirical evidence from P1-P5: P1 shows which models can achieve Bronze (most, with spec-driven decoding), P2 shows what training is needed for Silver (9B+ models with SLO-aware GRPO), P3 provides the benchmark for measuring compliance, P4 shows how to maintain tiers over time, and P5 validates the tiers in production.

### Paper Structure

1. **Introduction**: The need for a production-readiness standard for LLM agents
2. **Related Work**: Existing benchmarks, evaluation frameworks, and agent toolkits
3. **The `criteria.yaml` Specification**: Metric families, measurement protocols, threshold defaults, composition rules
4. **Tiered Certification**: Bronze/Silver/Gold definitions, evidence from P1-P5
5. **Reference Implementation**: Open-source toolkit, CLI, integration patterns
6. **Cross-Paper Evidence**: Summary of findings from P1-P5 supporting each standard component
7. **Community Adoption**: How teams can adopt the standard, contribute tasks, extend metrics
8. **Limitations and Future Work**: Safety metrics, fairness, multi-agent systems, closed-source model evaluation
9. **Conclusion**: A call for shared vocabulary in agent evaluation

### Key Contributions

- First proposed standard for production-ready agent evaluation with empirical backing
- `criteria.yaml` as a portable, extensible specification for agent contracts
- Tiered certification scheme (Bronze/Silver/Gold) enabling incremental adoption
- Open-source reference implementation with CLI and CI integration
- Synthesis of evidence from 5 prior papers demonstrating the framework's effectiveness

### Target Venue

Nature Machine Intelligence or Communications of the ACM (for broad impact), alternatively NeurIPS 2026 position paper track or a dedicated workshop paper. The capstone nature of P6 makes it suitable for venues that value synthesis and community impact over novelty.

---

## Cross-Paper Dependencies

```
P1 (Evaluation)  ──> P2 (Training)  ──> P4 (Dynamics)
      │                    │                   │
      v                    v                   v
P3 (Benchmark)      P5 (Case Study)    P6 (Standard)
```

- **P3 depends on P1**: Uses the same metrics and evaluation framework as a benchmark
- **P4 depends on P2**: Deepens the training analysis from P2 with ablations and dynamics
- **P5 depends on P1+P2**: Deploys the evaluation and training framework in production
- **P6 depends on P1-P5**: Synthesizes all evidence into a standard

## Timeline

| Paper | Draft | Target Submission | Dependency |
|-------|-------|-------------------|------------|
| P3 | Feb-Mar 2026 | Apr 2026 | P1 results |
| P4 | Mar-Apr 2026 | May 2026 | P2 training data |
| P5 | Apr-Jun 2026 | Jul 2026 | P1+P2 deployed |
| P6 | Jun-Aug 2026 | Sep 2026 | P1-P5 evidence |
