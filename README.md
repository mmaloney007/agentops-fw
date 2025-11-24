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
