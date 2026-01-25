# AgentOps-FW — Contract-grounded agents on a single GPU (RTX 4090)

**Author:** Mike Maloney — Co-Founder & CDO, Neuralift; Lecturer, University of New Hampshire

## Contact

| | |
|---|---|
| **Primary Email** | mike.maloney@unh.edu |
| **Personal Email** | mikey.maloney@gmail.com |
| **Work Email** | mike@neuralift.ai |
| **LinkedIn** | [linkedin.com/in/mike-maloney-5229274](https://www.linkedin.com/in/mike-maloney-5229274/) |
| **GitHub** | [github.com/mmaloney007](https://github.com/mmaloney007) |
| **W&B** | [wandb.ai/mike007](https://wandb.ai/mike007) |

## TL;DR
A minimal, reproducible framework for **reliable, SLA‑aware, tool‑using LLM agents** that runs on a **single RTX 4090** with **open‑weight models**. It supports:
- **Schema‑constrained vs post‑hoc** structured generation
- **Typed contracts & runtime monitors** (toy plugin shown)
- **Budgeted self‑consistency** (toy plugin shown)
- **W&B logging** for tables & reports

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q

# Generate results (and log to W&B if configured)
python -m agentops_fw.cli --tasks tasks/pilot.json --mode posthoc --out out/posthoc.csv --project agentops-fw
python -m agentops_fw.cli --tasks tasks/pilot.json --mode constrained --out out/constrained.csv --project agentops-fw
```

Dependencies and environment notes live in `docs/requirements.md`.

## GRPO/LoRA training presets (4090 + Mac)
Preset script: `scripts/run_grpo_presets.sh` (defaults timestamp outputs and uses W&B if configured).

1) Activate env + LM Studio defaults (optional):
```bash
source activate_mamba.sh base
source scripts/env_lm_remote.sh   # sets OPENAI_API_BASE/KEY/LMSTUDIO_MODEL, etc.
```
2) Enable W&B (skip for no logging):
```bash
export WANDB_PROJECT=agent-stable-slo
export WANDB_ENTITY=mike007
export WANDB_DIR=$(pwd)/wandb_logs          # optional local dir
# offline logging: export WANDB_MODE=offline
```
3) Make sure models are local (skip if already downloaded):
```bash
huggingface-cli download Qwen/Qwen3-4B-Thinking-2507 \
  --local-dir ./models/qwen3-4b-thinking-2507 --local-dir-use-symlinks False
# GPT-OSS-20B requires local weights at ./models/gpt-oss-20b
```
4) Run a preset on 4090:
```bash
bash scripts/run_grpo_presets.sh 4090-qwen3
# or: bash scripts/run_grpo_presets.sh 4090-gptoss
```
5) Run the Mac/MPS preset:
```bash
bash scripts/run_grpo_presets.sh mac-qwen3
```

Useful overrides (set env before the script): `MODEL_DIR`, `OUT` (defaults to timestamped names), `STEPS`, `MAX_NEW_TOKENS`, `GRAD_ACC`, `LOAD_IN_4BIT`, `WANDB_MODE`. Keep `HF_HUB_OFFLINE=1` if you want to stay offline after weights are present.

## Configure W&B
```bash
pip install wandb pandas
wandb login
export WANDB_ENTITY=mike007
export WANDB_PROJECT=agentops-fw
python scripts/wandb_bootstrap.py
```

## P2 Training Matrix (13 Models × 6 Tasks × 3 Seeds)

Full-scale training experiment across 13 open-weight models on 6 task types.

### Quick Start
```bash
# 1. Expand T1 dataset (10 → 100 samples)
python scripts/expand_t1_t2_samples.py --output-dir tasks/

# 2. Dry run to preview (234 total runs)
python scripts/run_p2_training_matrix.py --out-dir out/p2_training --dry-run

# 3. Run full training matrix
python scripts/run_p2_training_matrix.py --out-dir out/p2_training --steps 1000

# 4. Resume after crash
python scripts/run_p2_training_matrix.py --resume out/p2_training

# 5. Aggregate results for paper
python scripts/aggregate_p2_results.py --input out/p2_training --latex --csv
```

### Models (smallest → largest)
| Model | Params | 4-bit | Grad Accum |
|-------|--------|-------|------------|
| Llama-3.2-1B | 1B | No | 1 |
| Llama-3.2-3B | 3B | No | 1 |
| Qwen2.5-3B | 3B | No | 1 |
| Phi-3-mini | 3.8B | No | 1 |
| Qwen3-4B | 4B | No | 1 |
| Yi-1.5-6B | 6B | No | 2 |
| Mistral-7B-v0.3 | 7B | Yes | 2 |
| Falcon-Mamba-7B | 7B | Yes | 2 |
| Ministral-8B | 8B | Yes | 2 |
| Llama-3.1-8B | 8B | Yes | 2 |
| Gemma-2-9B | 9B | Yes | 2 |
| Gemma-3-12B | 12B | Yes | 4 |
| GPT-OSS-20B | 20B | Yes | 4 |

### Tasks
- **T1**: Incident classification (100 samples)
- **T2**: Grounded summarization (106 samples)
- **T3**: Tool selection (500 samples)
- **T4**: Function calling / BFCL (500 samples)
- **T5**: SWE-bench patches (300 samples)
- **Mixed**: Balanced T1-T5 (500 samples)

### Output Structure
```
out/p2_training/
├── progress_state.json          # Resume tracking
├── llama-3.2-1b/
│   ├── T1/seed_42/train_log.jsonl
│   ├── T1/seed_123/...
│   └── ...
├── aggregated_results.json      # After aggregation
└── latex_tables/table_training.tex
```

## GitHub CI
This repo includes GitHub Actions (Python 3.12) to lint (ruff/black) and run tests (pytest).

## Paper
See `papers/P1_stable_slo/arxiv/main.tex` and `papers/P2_reward_stability/arxiv/main.tex`.

## Archive
Legacy drafts and older results live under `archive/`.

## License
MIT. See `LICENSE`.
