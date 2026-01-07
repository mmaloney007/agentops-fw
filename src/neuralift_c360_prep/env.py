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


__all__ = ["dotenv_env_vars", "load_dotenv_file"]
