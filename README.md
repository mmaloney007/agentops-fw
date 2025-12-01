# AgentOps-FW — Contract-grounded agents on a single GPU (RTX 4090)

**Author:** Mike Maloney — Co‑Founder & CDO, Neuralift; Lecturer, UNH  
Email: mikey.maloney@gmail.com • mike.maloney@unh.edu  
GitHub: https://github.com/mmaloney007 • W&B: https://wandb.ai/mike007

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
python -m agentops_fw.cli --tasks tasks/pilot.json --mode posthoc --out results/posthoc.csv --project agentops-fw
python -m agentops_fw.cli --tasks tasks/pilot.json --mode constrained --out results/constrained.csv --project agentops-fw
```

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

## GitHub CI
This repo includes GitHub Actions (Python 3.11) to lint (ruff/black) and run tests (pytest).

## Paper
See `paper/paper.md` (JOSS).

## License
MIT. See `LICENSE`.
