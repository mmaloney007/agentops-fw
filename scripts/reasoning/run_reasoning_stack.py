#!/usr/bin/env python3
"""Convenience wrapper around `agent_stable_slo.reasoning_stack.cli`."""

import sys
from pathlib import Path

# Allow running via `python scripts/reasoning/run_reasoning_stack.py ...`
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from agent_stable_slo.reasoning_stack.cli import main as _main

    _main()


if __name__ == "__main__":
    main()
