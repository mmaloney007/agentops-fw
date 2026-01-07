#!/usr/bin/env python3
"""
Generate a pip-only lock file from pyproject.toml.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import tomllib


_SKIP_PACKAGES = {"pip", "setuptools", "wheel"}


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _req_name(req: str) -> str:
    for i, ch in enumerate(req):
        if ch in "<>!=~ ;[":
            return req[:i].strip()
    return req.strip()


def _add_req(reqs: dict[str, str], req: str) -> None:
    name = _req_name(req)
    key = _normalize_name(name)
    existing = reqs.get(key)
    if existing and existing != req:
        raise SystemExit(f"Conflicting requirement for {name}: {existing} vs {req}")
    reqs[key] = req


def _add_mapping_deps(
    mapping: dict[str, object],
    reqs: dict[str, str],
    skipped: list[str],
) -> None:
    for name, spec in mapping.items():
        if isinstance(spec, dict):
            if "path" in spec:
                skipped.append(name)
                continue
            version = spec.get("version", "")
        else:
            version = spec
        if version in (None, "", "*"):
            req = name
        else:
            req = f"{name}{version}"
        _add_req(reqs, req)


def _collect_requirements(pyproject_path: Path) -> tuple[list[str], list[str]]:
    data = tomllib.loads(pyproject_path.read_text())
    reqs: dict[str, str] = {}
    skipped: list[str] = []

    project = data.get("project", {})
    for dep in project.get("dependencies", []) or []:
        _add_req(reqs, dep)

    pixi = data.get("tool", {}).get("pixi", {})
    _add_mapping_deps(pixi.get("dependencies", {}), reqs, skipped)
    _add_mapping_deps(pixi.get("pypi-dependencies", {}), reqs, skipped)

    requirements = [reqs[key] for key in sorted(reqs)]
    return requirements, skipped


def _venv_bin(venv_dir: Path, name: str) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _install_and_freeze(
    requirements: Iterable[str],
    output_path: Path,
    python_exe: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        req_in = tmp_root / "requirements.in"
        req_in.write_text("\n".join(requirements) + "\n")

        venv_dir = tmp_root / "venv"
        subprocess.run([python_exe, "-m", "venv", str(venv_dir)], check=True)

        pip = _venv_bin(venv_dir, "pip")
        env = os.environ.copy()
        env.update({"PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1"})

        subprocess.run([str(pip), "install", "--upgrade", "pip"], check=True, env=env)
        subprocess.run([str(pip), "install", "-r", str(req_in)], check=True, env=env)

        result = subprocess.run(
            [str(pip), "freeze"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        lines = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("-e "):
                continue
            if " @ file://" in line:
                raise SystemExit(f"Found local path requirement in freeze output: {line}")
            pkg = line.split("==", 1)[0].strip().lower()
            if pkg in _SKIP_PACKAGES:
                continue
            lines.append(line)

        if not lines:
            raise SystemExit("No packages resolved; check pyproject dependencies.")

        tmp_out = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_out.write_text("\n".join(lines) + "\n")
        tmp_out.replace(output_path)
        return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a pip-only lock file from pyproject.toml.",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--output",
        default="coiled/requirements.txt",
        help="Path to write the locked requirements",
    )
    args = parser.parse_args()

    pyproject_path = Path(args.pyproject)
    if not pyproject_path.exists():
        raise SystemExit(f"pyproject.toml not found at {pyproject_path}")

    requirements, skipped = _collect_requirements(pyproject_path)
    if skipped:
        skipped_list = ", ".join(sorted(skipped))
        print(f"Skipping local path deps: {skipped_list}", file=sys.stderr)

    count = _install_and_freeze(
        requirements=requirements,
        output_path=Path(args.output),
        python_exe=sys.executable,
    )
    print(f"Wrote {args.output} with {count} packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
