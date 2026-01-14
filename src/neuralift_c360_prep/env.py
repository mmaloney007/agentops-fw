#!/usr/bin/env python3
"""
Helpers for loading .env files in local development.

Author: Mike Maloney - Neuralift, Inc.
Updated: 2025-01-07
Copyright © 2026 Neuralift, Inc.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, find_dotenv, load_dotenv

COILED_ENV_PREFIXES = ("DATABRICKS_", "OPENAI_", "WANDB_", "AWS_", "NL_")


def _find_dotenv_path(dotenv_path: str | Path | None = None) -> Path | None:
    if dotenv_path:
        path = Path(dotenv_path)
        return path if path.exists() else None
    found = find_dotenv(usecwd=True)
    if not found:
        return None
    return Path(found)


def load_dotenv_file(dotenv_path: str | Path | None = None) -> Path | None:
    path = _find_dotenv_path(dotenv_path)
    if not path:
        return None
    load_dotenv(path, override=False)
    return path


def dotenv_env_vars(dotenv_path: str | Path | None = None) -> dict[str, str]:
    path = _find_dotenv_path(dotenv_path)
    if not path:
        return {}

    load_dotenv(path, override=False)
    values = dotenv_values(path)
    env_vars: dict[str, str] = {}
    for key, value in values.items():
        if not key or value is None:
            continue
        env_vars[key] = os.environ.get(key, value)
    return env_vars


def collect_coiled_env_vars(
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    env_vars = dotenv_env_vars()
    for key, value in os.environ.items():
        if any(key.startswith(prefix) for prefix in COILED_ENV_PREFIXES):
            env_vars.setdefault(key, value)
    if extra:
        env_vars.update({k: v for k, v in extra.items() if v is not None})
    return {k: v for k, v in env_vars.items() if v not in (None, "")}


__all__ = [
    "COILED_ENV_PREFIXES",
    "collect_coiled_env_vars",
    "dotenv_env_vars",
    "load_dotenv_file",
]
