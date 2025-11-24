.PHONY: setup test lint format run-pilot

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check .

format:
	black .

run-pilot:
	python -m agentops_fw.cli --tasks tasks/pilot.json --mode constrained --out results/constrained.csv --project agentops-fw
