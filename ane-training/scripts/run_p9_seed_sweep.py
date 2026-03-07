#!/usr/bin/env python3
"""
Run the Paper 9 seed sweep.

Stage 1 is the current 2-model x 4-backend matrix replicated across 3 seeds.
Logs are written to:

    results/seed_sweep/<model>_<backend>/seed_<seed>/grpo_log.jsonl

This keeps stage 1 compatible with follow-on stage 2 model expansion without
changing the directory layout again.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = Path("/Users/maloney/.local/share/mamba/bin/python")

DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_MODELS = ["qwen", "smollm2"]
DEFAULT_BACKENDS = ["public", "private", "private-full", "mlx"]


MODEL_SPECS = {
    "qwen": {
        "public_model": "qwen2.5-0.5b",
        "private_config": "qwen05b",
        "weights": ROOT / "weights/qwen2.5-0.5b/model.safetensors",
        "tokenizer": ROOT / "weights/qwen2.5-0.5b/tokenizer.json",
        "coreml_dir": ROOT / "models/qwen05b_coreml",
        "mlx_model": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    "smollm2": {
        "public_model": "smollm2-360m",
        "private_config": "smollm2",
        "weights": ROOT / "weights/smollm2-360m/model.safetensors",
        "tokenizer": ROOT / "weights/smollm2-360m/tokenizer.json",
        "coreml_dir": ROOT / "models/smollm2_coreml",
        "mlx_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    },
    "stories110m": {
        "public_model": "stories110m",
        "private_config": "stories110m",
        "weights": ROOT / "weights/stories110m/model.safetensors",
        "tokenizer": ROOT / "weights/stories110m/tokenizer.json",
        "coreml_dir": ROOT / "models/stories110m_coreml",
        "mlx_model": None,
    },
}


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def build_command(
    *,
    model_key: str,
    backend: str,
    seed: int,
    tasks: str,
    steps: int,
    group_size: int,
    lr: float,
    temperature: float,
    max_tokens: int,
    python_bin: str,
    run_dir: Path,
) -> list[str]:
    spec = MODEL_SPECS[model_key]
    if backend == "public":
        return [
            str(ROOT / "grpo_public"),
            "--model", spec["public_model"],
            "--weights", str(spec["weights"]),
            "--tokenizer", str(spec["tokenizer"]),
            "--tasks", tasks,
            "--steps", str(steps),
            "--temperature", str(temperature),
            "--group-size", str(group_size),
            "--lr", str(lr),
            "--max-tokens", str(max_tokens),
            "--seed", str(seed),
            "--out-dir", str(run_dir),
        ]

    if backend in {"private", "private-full"}:
        cmd = [
            str(ROOT / "grpo_private"),
            "--model", str(spec["weights"]),
            "--tokenizer", str(spec["tokenizer"]),
            "--tasks", tasks,
            "--config", spec["private_config"],
            "--coreml-dir", str(spec["coreml_dir"]),
            "--steps", str(steps),
            "--group-size", str(group_size),
            "--lr", str(lr),
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
            "--seed", str(seed),
            "--out", str(run_dir / "grpo_log.jsonl"),
        ]
        if backend == "private-full":
            cmd.append("--backward-ane")
        return cmd

    if backend == "mlx":
        mlx_model = spec.get("mlx_model")
        if not mlx_model:
            raise ValueError(f"Model {model_key} does not have an MLX mapping yet")
        return [
            python_bin,
            str(ROOT / "scripts/run_mlx_grpo.py"),
            "--model", mlx_model,
            "--tasks", tasks,
            "--steps", str(steps),
            "--group-size", str(group_size),
            "--lr", str(lr),
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
            "--seed", str(seed),
            "--out", str(run_dir / "grpo_log.jsonl"),
        ]

    raise ValueError(f"Unsupported backend: {backend}")


def write_manifest(run_dir: Path, payload: dict) -> None:
    manifest_path(run_dir).write_text(json.dumps(payload, indent=2) + "\n")


def run_one(
    *,
    model_key: str,
    backend: str,
    seed: int,
    tasks: str,
    steps: int,
    group_size: int,
    lr: float,
    temperature: float,
    max_tokens: int,
    python_bin: str,
    results_dir: Path,
    resume: bool,
    dry_run: bool,
) -> dict:
    cell_name = f"{model_key}_{backend}"
    run_dir = results_dir / cell_name / f"seed_{seed}"
    log_path = run_dir / "grpo_log.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    if resume and log_path.exists() and log_path.stat().st_size > 0:
        return {
            "cell": cell_name,
            "seed": seed,
            "status": "skipped",
            "log": str(log_path),
        }

    cmd = build_command(
        model_key=model_key,
        backend=backend,
        seed=seed,
        tasks=tasks,
        steps=steps,
        group_size=group_size,
        lr=lr,
        temperature=temperature,
        max_tokens=max_tokens,
        python_bin=python_bin,
        run_dir=run_dir,
    )
    cmd_str = " ".join(shlex.quote(part) for part in cmd)

    if dry_run:
        return {
            "cell": cell_name,
            "seed": seed,
            "status": "dry-run",
            "command": cmd_str,
            "log": str(log_path),
        }

    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0

    payload = {
        "cell": cell_name,
        "model": model_key,
        "backend": backend,
        "seed": seed,
        "status": "ok" if result.returncode == 0 else "failed",
        "elapsed_s": round(elapsed, 2),
        "command": cmd,
        "log": str(log_path),
    }
    write_manifest(run_dir, payload)

    if result.returncode != 0:
        raise RuntimeError(f"{cell_name} seed {seed} failed with exit code {result.returncode}")
    if not log_path.exists():
        raise RuntimeError(f"{cell_name} seed {seed} did not produce {log_path}")
    return payload


def validate_args(args: argparse.Namespace) -> None:
    for model_key in args.models:
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_key}")
        spec = MODEL_SPECS[model_key]
        ensure_exists(spec["weights"], f"{model_key} weights")
        ensure_exists(spec["tokenizer"], f"{model_key} tokenizer")
        ensure_exists(spec["coreml_dir"], f"{model_key} CoreML directory")

    for backend in args.backends:
        if backend not in {"public", "private", "private-full", "mlx"}:
            raise ValueError(f"Unknown backend: {backend}")

    ensure_exists(Path(args.tasks), "tasks file")
    ensure_exists(ROOT / "grpo_public", "grpo_public binary")
    ensure_exists(ROOT / "grpo_private", "grpo_private binary")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Paper 9 3-seed replication sweep.")
    parser.add_argument("--seeds", default="42,123,456",
                        help="Comma-separated seeds (default: 42,123,456)")
    parser.add_argument("--models", default="qwen,smollm2",
                        help="Comma-separated model keys (default: qwen,smollm2)")
    parser.add_argument("--backends", default="public,private,private-full,mlx",
                        help="Comma-separated backend keys")
    parser.add_argument("--tasks", default=str(ROOT / "scripts/hard_tasks.jsonl"),
                        help="Tasks JSONL path")
    parser.add_argument("--steps", type=int, default=500,
                        help="GRPO steps per run")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Rollouts per step")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max generation tokens")
    parser.add_argument("--results-dir", default=str(ROOT / "results/seed_sweep"),
                        help="Output directory for logs")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)),
                        help="Python executable for MLX runs")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs that already have non-empty logs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    args.seeds = parse_csv_ints(args.seeds)
    args.models = parse_csv_strings(args.models)
    args.backends = parse_csv_strings(args.backends)
    validate_args(args)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    plan = [
        (model_key, backend, seed)
        for model_key in args.models
        for backend in args.backends
        for seed in args.seeds
    ]

    print(
        f"Running P9 seed sweep: {len(args.models)} models x "
        f"{len(args.backends)} backends x {len(args.seeds)} seeds = {len(plan)} runs",
        file=sys.stderr,
    )

    completed = []
    for idx, (model_key, backend, seed) in enumerate(plan, start=1):
        print(
            f"[{idx}/{len(plan)}] {model_key} / {backend} / seed {seed}",
            file=sys.stderr,
        )
        result = run_one(
            model_key=model_key,
            backend=backend,
            seed=seed,
            tasks=args.tasks,
            steps=args.steps,
            group_size=args.group_size,
            lr=args.lr,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            python_bin=args.python,
            results_dir=results_dir,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        completed.append(result)
        if args.dry_run:
            print(result["command"])

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(completed, indent=2) + "\n")
    print(f"Summary written to {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
