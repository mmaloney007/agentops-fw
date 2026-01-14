# Dependencies and Environment

This repo uses both a Python package manifest and pip requirements files. Keep them aligned.

## Python
- `pyproject.toml`: canonical package metadata and minimal runtime dependencies.
- `requirements.txt`: runtime dependencies for local runs.
- `requirements-dev.txt`: lint/test/dev tooling.
- `environment.yml`: optional conda/mamba environment spec (includes Python version and core deps).
Requires Python 3.12+.

Typical setup:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Node (optional)
- `package.json` / `package-lock.json`: any Node-based tooling or docs pipelines.

If a dependency is added, update `pyproject.toml` and the appropriate requirements file(s).
