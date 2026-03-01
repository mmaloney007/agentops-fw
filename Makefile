.PHONY: setup setup-mamba setup-mamba-dev test lint format run-pilot

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt

setup-mamba:
	@if command -v micromamba >/dev/null 2>&1; then \
		if micromamba env list | grep -E '^[^#]*agent-slo' >/dev/null 2>&1; then \
			micromamba env update -f environment.yml -y; \
		else \
			micromamba create -f environment.yml -y; \
		fi; \
	elif command -v mamba >/dev/null 2>&1; then \
		if mamba env list | grep -E '^[^#]*agent-slo' >/dev/null 2>&1; then \
			mamba env update -f environment.yml -y; \
		else \
			mamba env create -f environment.yml -y; \
		fi; \
	else \
		echo "micromamba/mamba not found. Install one and retry."; \
		exit 1; \
	fi
	@echo "Next: source ./activate_mamba.sh agent-slo"
	@echo "Then install torch for your platform and dev deps:"
	@echo "pip install torch==2.4.1 && pip install -r requirements-dev.txt"

setup-mamba-dev:
	bash -lc 'source ./activate_mamba.sh agent-slo && pip install -r requirements-dev.txt'

test:
	pytest -q

lint:
	ruff check .

format:
	black .

run-pilot:
	python -m agentops_fw.cli --tasks tasks/pilot.json --mode constrained --out out/constrained.csv --project agentops-fw
