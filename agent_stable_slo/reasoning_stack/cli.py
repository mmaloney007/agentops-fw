"""CLI entry for reasoning stack orchestration."""

from __future__ import annotations

import argparse
import json

from .config import load_reasoning_stack_config
from .pipeline import run_reasoning_stack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run two-stage local training (base LM + reasoning) on CUDA or MLX",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to reasoning stack YAML config",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_reasoning_stack_config(args.config)
    summary = run_reasoning_stack(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
