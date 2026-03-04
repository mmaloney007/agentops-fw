#!/usr/bin/env python3
"""
MLX evaluation wrapper -- runs eval_t_suite against Apple-Silicon MLX models.

Sets AOFW_PROVIDER=mlx_local and runs each model sequentially (one loaded at
a time) to stay within unified-memory limits.

Usage:
  python scripts/eval_mlx_suite.py --all
  python scripts/eval_mlx_suite.py --dense-only
  python scripts/eval_mlx_suite.py --models llama-3.2-1b qwen2.5-3b
  python scripts/eval_mlx_suite.py --models mlx-community/Llama-3.2-1B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Model registry: short-name -> mlx-community HuggingFace ID
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, str] = {
    "llama-3.2-1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "llama-3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "qwen2.5-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "qwen3-4b": "mlx-community/Qwen3-4B-4bit",
    "phi-3-mini": "mlx-community/Phi-3-mini-4k-instruct-4bit",
    "mistral-7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "ministral-8b": "mlx-community/Ministral-8B-Instruct-2410-4bit",
    "gemma-2-9b": "mlx-community/gemma-2-9b-it-4bit",
    # Qwen3.5 dense family (Gated DeltaNet, released Mar 2 2026)
    "qwen3.5-0.8b": "mlx-community/Qwen3.5-0.8B-4bit",
    "qwen3.5-2b": "mlx-community/Qwen3.5-2B-4bit",
    "qwen3.5-4b": "mlx-community/Qwen3.5-4B-4bit",
    "qwen3.5-9b": "mlx-community/Qwen3.5-9B-4bit",
    # Qwen3.5 MoE family
    "qwen3-30b-moe": "mlx-community/Qwen3-30B-A3B-4bit",
    "qwen3.5-35b-moe": "mlx-community/Qwen3.5-35B-A3B-4bit",
}

MOE_MODELS = {"qwen3-30b-moe", "qwen3.5-35b-moe"}

# Dense models = everything except MoE models
DENSE_MODELS = [k for k in MODEL_REGISTRY if k not in MOE_MODELS]

DEFAULT_TASKS = [
    "tasks/clinc_en.jsonl",
    "tasks/hotpot_dev.jsonl",
    "tasks/t3_tools.jsonl",
    "tasks/t4_bfcl.jsonl",
    "tasks/t5_swebench.jsonl",
    "tasks/public_gsm8k.jsonl",
]

# ---------------------------------------------------------------------------
# Hardware metadata helpers
# ---------------------------------------------------------------------------


def _sysctl(key: str) -> str:
    """Read a sysctl value; return empty string on failure."""
    try:
        return (
            subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def get_hardware_info() -> Dict[str, Any]:
    """Collect Apple-Silicon hardware metadata."""
    info: Dict[str, Any] = {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "core_count_total": _sysctl("hw.ncpu"),
        "core_count_perf": _sysctl("hw.perflevel0.logicalcpu_max"),
        "core_count_eff": _sysctl("hw.perflevel1.logicalcpu_max"),
        "memory_gb": "",
        "macos_version": platform.mac_ver()[0],
        "macos_build": platform.mac_ver()[2],
        "python_version": platform.python_version(),
        "mlx_version": "",
        "mlx_lm_version": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Physical memory
    try:
        mem_bytes = int(_sysctl("hw.memsize"))
        info["memory_gb"] = round(mem_bytes / (1024**3), 1)
    except (ValueError, TypeError):
        pass

    # MLX version
    try:
        import mlx

        info["mlx_version"] = getattr(mlx, "__version__", "unknown")
    except ImportError:
        info["mlx_version"] = "not installed"

    # mlx-lm version
    try:
        import mlx_lm

        info["mlx_lm_version"] = getattr(mlx_lm, "__version__", "unknown")
    except ImportError:
        info["mlx_lm_version"] = "not installed"

    return info


# ---------------------------------------------------------------------------
# Resolve model names
# ---------------------------------------------------------------------------


def resolve_model(name: str) -> str:
    """Map a short name to full HF ID, or pass through if already qualified."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    # Accept raw HF IDs (contain a '/')
    if "/" in name:
        return name
    raise ValueError(
        f"Unknown model '{name}'. Use a registry key {list(MODEL_REGISTRY)} "
        f"or a full HuggingFace ID (org/repo)."
    )


def slug_for(hf_id: str) -> str:
    """Filesystem-safe slug from HF ID."""
    return hf_id.replace("/", "_").replace(":", "-")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def run_single_model(
    hf_id: str,
    tasks: List[str],
    out_dir: Path,
    run_name: str,
    extra_args: List[str],
) -> Dict[str, Any]:
    """
    Evaluate one MLX model by calling eval_t_suite.main() in a fresh subprocess.

    Using subprocess ensures each model starts with a clean memory slate.
    """
    slug = slug_for(hf_id)
    model_spec = f"mlx_local:{hf_id}"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval_t_suite.py"),
        "--models",
        model_spec,
        "--tasks",
        *tasks,
        "--out-dir",
        str(out_dir),
        "--run-name",
        run_name,
        *extra_args,
    ]

    env = os.environ.copy()
    env["AOFW_PROVIDER"] = "mlx_local"
    env["MLX_MODEL"] = hf_id
    # Ensure project root is on PYTHONPATH so subprocess can find agent_stable_slo
    project_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n{'='*72}")
    print(f"  Model : {hf_id}")
    print(f"  Slug  : {slug}")
    print(f"  Tasks : {len(tasks)} files")
    print(f"{'='*72}\n")

    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    elapsed = time.time() - t0

    status = "ok" if result.returncode == 0 else "error"
    print(f"\n  --> {slug}: {status} ({elapsed:.1f}s)")

    return {
        "model": hf_id,
        "slug": slug,
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run MLX eval suite on Apple Silicon models."
    )

    # Model selection (mutually exclusive convenience flags)
    model_group = ap.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--models",
        nargs="+",
        help="Model short-names or full HuggingFace IDs.",
    )
    model_group.add_argument(
        "--all",
        action="store_true",
        help="Run all 9 models in the registry.",
    )
    model_group.add_argument(
        "--dense-only",
        action="store_true",
        help="Run the 8 dense models (skip MoE).",
    )

    ap.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task JSONL files to evaluate.",
    )
    ap.add_argument(
        "--out-dir",
        default="results/mlx_eval",
        help="Root output directory (default: results/mlx_eval).",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Run name; defaults to timestamped name.",
    )
    ap.add_argument(
        "--capture-detailed",
        action="store_true",
        help="Pass --capture-detailed to eval_t_suite.",
    )
    ap.add_argument(
        "--stability-runs",
        type=int,
        default=1,
        help="Number of stability runs per prompt.",
    )
    ap.add_argument(
        "--slo-budget-ms",
        type=float,
        default=2000.0,
        help="SLO budget in milliseconds.",
    )
    ap.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Cap total records per model (0 = no cap).",
    )

    args = ap.parse_args()

    # Resolve model list
    if args.all:
        model_ids = list(MODEL_REGISTRY.values())
    elif args.dense_only:
        model_ids = [MODEL_REGISTRY[k] for k in DENSE_MODELS]
    else:
        model_ids = [resolve_model(m) for m in args.models]

    # Validate task files exist
    for t in args.tasks:
        if not Path(t).exists():
            raise SystemExit(f"Task file not found: {t}")

    run_name = args.run_name or f"mlx_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect and save hardware info
    hw_info = get_hardware_info()
    hw_path = out_dir / run_name
    hw_path.mkdir(parents=True, exist_ok=True)
    with open(hw_path / "hardware_info.json", "w", encoding="utf-8") as f:
        json.dump(hw_info, f, indent=2)
    print(f"Hardware info saved to {hw_path / 'hardware_info.json'}")
    print(f"  Chip       : {hw_info['chip']}")
    print(f"  Memory     : {hw_info['memory_gb']} GB")
    print(f"  macOS      : {hw_info['macos_version']}")
    print(f"  MLX        : {hw_info['mlx_version']}")
    print(f"  mlx-lm     : {hw_info['mlx_lm_version']}")
    print(f"  Models     : {len(model_ids)}")
    print(f"  Tasks      : {args.tasks}")
    print(f"  Out dir    : {out_dir / run_name}")

    # Build extra args to forward
    extra_args: List[str] = []
    if args.capture_detailed:
        extra_args.append("--capture-detailed")
    if args.stability_runs > 1:
        extra_args.extend(["--stability-runs", str(args.stability_runs)])
    if args.slo_budget_ms != 2000.0:
        extra_args.extend(["--slo-budget-ms", str(args.slo_budget_ms)])
    if args.max_records > 0:
        extra_args.extend(["--max-records", str(args.max_records)])

    # Sequential evaluation
    run_log: List[Dict[str, Any]] = []
    overall_t0 = time.time()

    for i, hf_id in enumerate(model_ids, 1):
        print(f"\n[{i}/{len(model_ids)}] Starting {hf_id}...")
        summary = run_single_model(
            hf_id=hf_id,
            tasks=args.tasks,
            out_dir=out_dir,
            run_name=run_name,
            extra_args=extra_args,
        )
        run_log.append(summary)

    overall_elapsed = time.time() - overall_t0

    # Save run manifest
    manifest = {
        "run_name": run_name,
        "models": model_ids,
        "tasks": args.tasks,
        "hardware": hw_info,
        "results": run_log,
        "total_elapsed_s": round(overall_elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = hw_path / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Print summary table
    print(f"\n{'='*72}")
    print(f"  MLX Eval Complete -- {len(model_ids)} models in {overall_elapsed:.0f}s")
    print(f"{'='*72}")
    print(f"  {'Model':<50} {'Status':<8} {'Time':>8}")
    print(f"  {'-'*50} {'-'*8} {'-'*8}")
    for entry in run_log:
        print(f"  {entry['model']:<50} {entry['status']:<8} {entry['elapsed_s']:>7.1f}s")
    print(f"\n  Manifest: {manifest_path}")

    # Exit with error if any model failed
    failures = [e for e in run_log if e["status"] != "ok"]
    if failures:
        print(f"\n  WARNING: {len(failures)} model(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
