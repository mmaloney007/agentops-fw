# Dependencies and Environment

This repo uses both a Python package manifest and pip requirements files. Keep them aligned.

## Python
- `pyproject.toml`: canonical package metadata and minimal runtime dependencies.
- `requirements.txt`: runtime dependencies for local runs.
- `requirements-dev.txt`: lint/test/dev tooling.
- `environment.yml`: preferred conda/mamba environment spec (includes Python version and core deps).
- `activate_mamba.sh`: helper script that activates `agent-slo` and sets repo runtime defaults.
Requires Python 3.12+.
Versioning is derived from git tags via `setuptools-scm` (build metadata includes commit info).

Preferred setup:
```bash
micromamba create -f environment.yml -y
source ./activate_mamba.sh agent-slo
pip install -r requirements-dev.txt
```

Fallback setup:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Tooling Direction
- Standardize on `mamba` right now for reproducible local development and easier GPU-specific torch installs.
- Add `pixi` when you want checked-in task runners and lockfile-style reproducibility across multiple OS targets.

## Node (optional)
- `package.json` / `package-lock.json`: any Node-based tooling or docs pipelines.

If a dependency is added, update `pyproject.toml` and the appropriate requirements file(s).
