# SpecSLOEval: The Deployment Gap

**Why benchmark accuracy fails to predict production readiness.**

This repository contains code, data, and papers for evaluating LLM agents under realistic deployment constraints. The core finding: accuracy rankings are negatively correlated with deployment success when latency matters.

## Key Results

| Model | Accuracy Rank | Success@SLO | SLO Rank |
|-------|---------------|-------------|----------|
| Ministral-8B | #1 (84.7%) | 28.5% | #6 |
| Gemma-3-12B | #4 (61.6%) | 0.0% | #13 |
| Llama-3.2-1B | #13 (27.3%) | **49.1%** | **#1** |

The smallest model wins. The accuracy leaders fail.

## Papers

| Paper | Title | Status |
|-------|-------|--------|
| P1 | The Deployment Gap: Why Benchmark Accuracy Fails to Predict Production Readiness | Ready for arXiv |
| P2 | Capacity Thresholds in Schema-Aware Training | Ready for arXiv |
| P3 | AgentSLO-Bench (benchmark paper) | Planned |

Papers are in `papers/P1_stable_slo/arxiv/` and `papers/P2_reward_stability/arxiv/`.

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/activate
pip install -r requirements.txt

# Run evaluation (requires LM Studio with model loaded)
python scripts/run_p1_comprehensive.py

# Run training
python -m agent_stable_slo.train.grpo_train_loop \
    --model Qwen/Qwen2.5-3B-Instruct \
    --tasks tasks/t3_tools.jsonl \
    --steps 500
```

## Tasks

Five task types covering common agent workloads:

| Task | File | Examples | Description |
|------|------|----------|-------------|
| T1 | `tasks/clinc_en.jsonl` | 500 | Intent classification (CLINC-150) |
| T2 | `tasks/hotpot_dev.jsonl` | 1000 | Multi-hop QA (HotpotQA) |
| T3 | `tasks/t3_tools.jsonl` | 500 | Tool calling |
| T4 | `tasks/t4_bfcl.jsonl` | 500 | Function routing (BFCL) |
| T5 | `tasks/t5_swebench.jsonl` | 300 | Code patching (SWE-bench) |

See `tasks/README.md` for full documentation.

## P2 Training Matrix

Full-scale training: 13 models × 6 tasks × 3 seeds = 234 runs.

```bash
# Run full training matrix
python scripts/run_p2_training_matrix.py --out-dir out/p2_training --steps 1000

# Resume after crash (auto-retries OOM failures)
python scripts/run_p2_training_matrix.py --resume out/p2_training

# Aggregate results
python scripts/aggregate_p2_results.py --input out/p2_training --latex --csv
```

## Results

Evaluation results are in `out/p1_comprehensive_20260118/all_results.json`.

Training results are in `results/p2_training_results.csv`.

## Hardware

All experiments run on a single RTX 4090 (24GB) with 4-bit quantization. No cluster required.

## Structure

```
agentops-fw/
├── papers/           # LaTeX source for P1 and P2
├── tasks/            # Task files (T1-T5) and schemas
├── scripts/          # Evaluation and training scripts
├── out/              # Evaluation outputs
├── results/          # Training results CSV
└── plan.md           # Research plan and progress log
```

## Author

Mike Maloney
Neuralift; University of New Hampshire
mike.maloney@unh.edu

## License

MIT
