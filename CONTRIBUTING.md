# Contributing to AgentOps-FW

Thanks for helping improve reliable, contract-grounded agents!

## Development setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

## Running tests
```bash
pytest -q
```

## Submitting changes
- Open a PR with a clear description.
- Ensure CI is green.
- Link to a W&B run/report when relevant.
