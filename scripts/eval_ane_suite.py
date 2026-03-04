#!/usr/bin/env python3
"""
ANE (Apple Neural Engine) evaluation wrapper -- runs eval_t_suite against
CoreML models converted via Anemll, targeting CPU_AND_NE compute units.

Sets AOFW_PROVIDER=ane_local, ANE_META_DIR, and ANE_HF_MODEL for each model,
then runs eval_t_suite.py in a fresh subprocess (one model at a time) to keep
memory clean between runs.

With --include-mlx-baseline, also evaluates MLX GPU baseline models through
the same harness (AOFW_PROVIDER=mlx_local) for head-to-head ANE vs GPU
comparison on identical tasks.

Usage:
  python scripts/eval_ane_suite.py --all
  python scripts/eval_ane_suite.py --models qwen3.5-0.8b qwen3.5-2b
  python scripts/eval_ane_suite.py --models qwen3.5-4b --measure-power
  python scripts/eval_ane_suite.py --all --include-mlx-baseline
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
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Model registry: short-name -> (HuggingFace ID, ANE model directory)
# ---------------------------------------------------------------------------

# Each entry maps: short_name -> (hf_id, ane_dir)
# ane_dir is the path to the Anemll-converted CoreML model directory
# (must contain meta.yaml and .mlmodelc artifacts).
MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "models/ane/qwen2.5-0.5b"),
    "llama-3.2-1b": ("meta-llama/Llama-3.2-1B-Instruct", "models/ane/llama-3.2-1b"),
    "gemma-3-1b":   ("google/gemma-3-1b-it", "models/ane/gemma-3-1b"),
}

# ---------------------------------------------------------------------------
# MLX GPU baseline registry: short-name -> MLX model ID
# Used with --include-mlx-baseline for head-to-head ANE vs GPU comparison.
# ---------------------------------------------------------------------------

MLX_BASELINE_REGISTRY: Dict[str, str] = {
    "qwen3.5-0.8b-mlx": "mlx-community/Qwen3.5-0.8B-4bit",
}

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
    """Collect Apple-Silicon hardware metadata (ANE-specific variant)."""
    info: Dict[str, Any] = {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "core_count_total": _sysctl("hw.ncpu"),
        "core_count_perf": _sysctl("hw.perflevel0.logicalcpu_max"),
        "core_count_eff": _sysctl("hw.perflevel1.logicalcpu_max"),
        "memory_gb": "",
        "macos_version": platform.mac_ver()[0],
        "macos_build": platform.mac_ver()[2],
        "python_version": platform.python_version(),
        "coremltools_version": "",
        "anemll_version": "",
        "compute_target": "ANE (CPU_AND_NE)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Physical memory
    try:
        mem_bytes = int(_sysctl("hw.memsize"))
        info["memory_gb"] = round(mem_bytes / (1024**3), 1)
    except (ValueError, TypeError):
        pass

    # coremltools version
    try:
        import coremltools as ct

        info["coremltools_version"] = getattr(ct, "__version__", "unknown")
    except ImportError:
        info["coremltools_version"] = "not installed"

    # anemll version (optional dependency; graceful fallback)
    try:
        import anemll

        info["anemll_version"] = getattr(anemll, "__version__", "unknown")
    except ImportError:
        info["anemll_version"] = "not installed"

    return info


# ---------------------------------------------------------------------------
# Resolve model names
# ---------------------------------------------------------------------------


def resolve_model(name: str) -> Tuple[str, str]:
    """Map a short name to (hf_id, ane_dir), or raise ValueError."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    raise ValueError(
        f"Unknown model '{name}'. Use a registry key {list(MODEL_REGISTRY)}."
    )


def slug_for(name: str) -> str:
    """Filesystem-safe slug from a model name or HF ID."""
    return name.replace("/", "_").replace(":", "-")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def run_single_model(
    name: str,
    model_info: Tuple[str, str],
    tasks: List[str],
    out_dir: Path,
    run_name: str,
    extra_args: List[str],
    measure_power: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate one ANE model by calling eval_t_suite.py in a fresh subprocess.

    Sets AOFW_PROVIDER=ane_local, ANE_META_DIR, and ANE_HF_MODEL so the
    ane_local provider knows which model to load.

    Using subprocess ensures each model starts with a clean memory slate.

    Parameters
    ----------
    name:
        Short registry name (e.g. "qwen3.5-2b").
    model_info:
        (hf_id, ane_dir) tuple from MODEL_REGISTRY.
    tasks:
        List of JSONL task file paths.
    out_dir:
        Root output directory.
    run_name:
        Run label used for sub-directory naming.
    extra_args:
        Additional flags forwarded to eval_t_suite.py.
    measure_power:
        If True, wrap the subprocess call with PowerMonitor and save
        a power_summary.json alongside the model results.
    """
    hf_id, ane_dir = model_info
    slug = slug_for(name)

    # Verify the ANE model directory exists before attempting evaluation.
    ane_dir_path = Path(ane_dir)
    if not ane_dir_path.exists():
        print(
            f"\n  [SKIP] ANE model directory not found: {ane_dir_path.resolve()}\n"
            f"         Convert the model first with Anemll, then re-run.\n"
        )
        return {
            "model": name,
            "hf_id": hf_id,
            "ane_dir": str(ane_dir_path),
            "slug": slug,
            "status": "skipped (ane_dir missing)",
            "returncode": None,
            "elapsed_s": 0.0,
        }

    model_spec = f"ane_local:{hf_id}"

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
    env["AOFW_PROVIDER"] = "ane_local"
    env["ANE_META_DIR"] = str(ane_dir_path.resolve())
    env["ANE_HF_MODEL"] = hf_id
    # Ensure project root is on PYTHONPATH so subprocess can find agent_stable_slo
    project_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n{'='*72}")
    print(f"  Model   : {name}")
    print(f"  HF ID   : {hf_id}")
    print(f"  ANE dir : {ane_dir_path}")
    print(f"  Tasks   : {len(tasks)} files")
    print(f"  Power   : {'enabled' if measure_power else 'disabled'}")
    print(f"{'='*72}\n")

    # Optional power monitoring
    power_monitor = None
    if measure_power:
        try:
            from agent_stable_slo.bench.power_monitor import PowerMonitor

            power_monitor = PowerMonitor(interval_ms=100)
            power_monitor.start()
        except ImportError:
            print("  [WARN] PowerMonitor not available -- power measurement disabled.")
            power_monitor = None

    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    elapsed = time.time() - t0

    # Stop power monitor and save summary
    if power_monitor is not None:
        power_summary = power_monitor.stop()
        summary_dict = power_summary.to_dict()
        summary_dict["model"] = name
        summary_dict["hf_id"] = hf_id
        summary_dict["ane_dir"] = str(ane_dir_path)

        # Save power summary into the run sub-directory
        power_dir = out_dir / run_name
        power_dir.mkdir(parents=True, exist_ok=True)
        power_path = power_dir / f"power_summary_{slug}.json"
        with open(power_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)
        print(f"\n  Power summary saved to {power_path}")
        print(
            f"  Mean total: {summary_dict['mean_total_w']:.2f} W  "
            f"ANE: {summary_dict['mean_ane_w']:.2f} W  "
            f"Energy: {summary_dict['energy_j']:.1f} J"
        )

    status = "ok" if result.returncode == 0 else "error"
    print(f"\n  --> {slug}: {status} ({elapsed:.1f}s)")

    return {
        "model": name,
        "hf_id": hf_id,
        "ane_dir": str(ane_dir_path),
        "slug": slug,
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# MLX GPU baseline evaluation
# ---------------------------------------------------------------------------


def run_single_mlx_model(
    name: str,
    mlx_model_id: str,
    tasks: List[str],
    out_dir: Path,
    run_name: str,
    extra_args: List[str],
    measure_power: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate one MLX (GPU) model for head-to-head comparison with ANE results.

    Sets AOFW_PROVIDER=mlx_local and MLX_MODEL so the mlx_local provider
    handles inference via Metal GPU.  Same task suite, same eval harness,
    different compute path.

    Parameters
    ----------
    name:
        Short registry name (e.g. "qwen3.5-0.8b-mlx").
    mlx_model_id:
        MLX model ID string (e.g. "mlx-community/Qwen3.5-0.8B-4bit").
    tasks:
        List of JSONL task file paths.
    out_dir:
        Root output directory.
    run_name:
        Run label used for sub-directory naming.
    extra_args:
        Additional flags forwarded to eval_t_suite.py.
    measure_power:
        If True, wrap the subprocess call with PowerMonitor and save
        a power_summary.json alongside the model results.
    """
    slug = slug_for(name)
    model_spec = f"mlx_local:{mlx_model_id}"

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
    env["MLX_MODEL"] = mlx_model_id
    env["MLX_ENABLE_THINKING"] = "0"
    # Ensure project root is on PYTHONPATH so subprocess can find agent_stable_slo
    project_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n{'='*72}")
    print(f"  MLX Baseline Model")
    print(f"  Model   : {name}")
    print(f"  MLX ID  : {mlx_model_id}")
    print(f"  Tasks   : {len(tasks)} files")
    print(f"  Power   : {'enabled' if measure_power else 'disabled'}")
    print(f"{'='*72}\n")

    # Optional power monitoring
    power_monitor = None
    if measure_power:
        try:
            from agent_stable_slo.bench.power_monitor import PowerMonitor

            power_monitor = PowerMonitor(interval_ms=100)
            power_monitor.start()
        except ImportError:
            print("  [WARN] PowerMonitor not available -- power measurement disabled.")
            power_monitor = None

    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    elapsed = time.time() - t0

    # Stop power monitor and save summary
    if power_monitor is not None:
        power_summary = power_monitor.stop()
        summary_dict = power_summary.to_dict()
        summary_dict["model"] = name
        summary_dict["mlx_model_id"] = mlx_model_id
        summary_dict["backend"] = "mlx_gpu"

        # Save power summary into the run sub-directory
        power_dir = out_dir / run_name
        power_dir.mkdir(parents=True, exist_ok=True)
        power_path = power_dir / f"power_summary_{slug}.json"
        with open(power_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)
        print(f"\n  Power summary saved to {power_path}")
        print(
            f"  Mean total: {summary_dict['mean_total_w']:.2f} W  "
            f"GPU: {summary_dict.get('mean_gpu_w', 0):.2f} W  "
            f"Energy: {summary_dict['energy_j']:.1f} J"
        )

    status = "ok" if result.returncode == 0 else "error"
    print(f"\n  --> {slug}: {status} ({elapsed:.1f}s)")

    return {
        "model": name,
        "mlx_model_id": mlx_model_id,
        "backend": "mlx_gpu",
        "slug": slug,
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run ANE eval suite on Apple Neural Engine CoreML models."
    )

    # Model selection (mutually exclusive)
    model_group = ap.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--models",
        nargs="+",
        help="Model short-names from the registry (e.g. qwen3.5-0.8b qwen3.5-2b).",
    )
    model_group.add_argument(
        "--all",
        action="store_true",
        help=f"Run all {len(MODEL_REGISTRY)} models in the registry.",
    )

    ap.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task JSONL files to evaluate.",
    )
    ap.add_argument(
        "--out-dir",
        default="results/ane_eval",
        help="Root output directory (default: results/ane_eval).",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Run name; defaults to timestamped name.",
    )
    ap.add_argument(
        "--measure-power",
        action="store_true",
        help="Collect CPU/GPU/ANE power via powermetrics (requires sudo -n).",
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
    ap.add_argument(
        "--include-mlx-baseline",
        action="store_true",
        help="Also run MLX GPU baseline models for ANE vs GPU comparison.",
    )

    args = ap.parse_args()

    # Resolve model list
    if args.all:
        selected: List[Tuple[str, Tuple[str, str]]] = list(MODEL_REGISTRY.items())
    else:
        selected = []
        for m in args.models:
            selected.append((m, resolve_model(m)))

    # Validate task files exist
    for t in args.tasks:
        if not Path(t).exists():
            raise SystemExit(f"Task file not found: {t}")

    run_name = args.run_name or f"ane_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect and save hardware info
    hw_info = get_hardware_info()
    hw_path = out_dir / run_name
    hw_path.mkdir(parents=True, exist_ok=True)
    with open(hw_path / "hardware_info.json", "w", encoding="utf-8") as f:
        json.dump(hw_info, f, indent=2)

    print(f"Hardware info saved to {hw_path / 'hardware_info.json'}")
    print(f"  Chip          : {hw_info['chip']}")
    print(f"  Memory        : {hw_info['memory_gb']} GB")
    print(f"  macOS         : {hw_info['macos_version']}")
    print(f"  coremltools   : {hw_info['coremltools_version']}")
    print(f"  anemll        : {hw_info['anemll_version']}")
    print(f"  Compute target: {hw_info['compute_target']}")
    print(f"  Models        : {len(selected)}")
    print(f"  Tasks         : {args.tasks}")
    print(f"  Out dir       : {out_dir / run_name}")

    # Build extra args to forward to eval_t_suite.py
    extra_args: List[str] = []
    if args.capture_detailed:
        extra_args.append("--capture-detailed")
    if args.stability_runs > 1:
        extra_args.extend(["--stability-runs", str(args.stability_runs)])
    if args.slo_budget_ms != 2000.0:
        extra_args.extend(["--slo-budget-ms", str(args.slo_budget_ms)])
    if args.max_records > 0:
        extra_args.extend(["--max-records", str(args.max_records)])

    # Sequential evaluation (one model at a time for memory safety)
    run_log: List[Dict[str, Any]] = []
    mlx_run_log: List[Dict[str, Any]] = []
    overall_t0 = time.time()

    for i, (name, model_info) in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] Starting {name} ({model_info[0]})...")
        summary = run_single_model(
            name=name,
            model_info=model_info,
            tasks=args.tasks,
            out_dir=out_dir,
            run_name=run_name,
            extra_args=extra_args,
            measure_power=args.measure_power,
        )
        run_log.append(summary)

    # MLX GPU baseline evaluation (optional, for ANE vs GPU comparison)
    if args.include_mlx_baseline:
        mlx_items = list(MLX_BASELINE_REGISTRY.items())
        print(f"\n{'='*72}")
        print(f"  MLX GPU Baseline -- {len(mlx_items)} model(s)")
        print(f"{'='*72}")

        for i, (name, mlx_model_id) in enumerate(mlx_items, 1):
            print(f"\n[MLX {i}/{len(mlx_items)}] Starting {name} ({mlx_model_id})...")
            summary = run_single_mlx_model(
                name=name,
                mlx_model_id=mlx_model_id,
                tasks=args.tasks,
                out_dir=out_dir,
                run_name=run_name,
                extra_args=extra_args,
                measure_power=args.measure_power,
            )
            mlx_run_log.append(summary)

    overall_elapsed = time.time() - overall_t0

    # Save run manifest
    manifest = {
        "run_name": run_name,
        "models": {name: {"hf_id": hf_id, "ane_dir": ane_dir} for name, (hf_id, ane_dir) in selected},
        "tasks": args.tasks,
        "hardware": hw_info,
        "results": run_log,
        "mlx_baseline_results": mlx_run_log,
        "include_mlx_baseline": args.include_mlx_baseline,
        "total_elapsed_s": round(overall_elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = hw_path / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Print summary table -- ANE models
    print(f"\n{'='*72}")
    print(f"  ANE Eval Complete -- {len(selected)} ANE models in {overall_elapsed:.0f}s")
    print(f"{'='*72}")
    print(f"  {'Model':<20} {'HF ID':<28} {'Status':<28} {'Time':>8}")
    print(f"  {'-'*20} {'-'*28} {'-'*28} {'-'*8}")
    for entry in run_log:
        print(
            f"  {entry['model']:<20} {entry['hf_id']:<28} "
            f"{entry['status']:<28} {entry['elapsed_s']:>7.1f}s"
        )

    # Print MLX baseline summary (if any)
    if mlx_run_log:
        print(f"\n  {'MLX GPU Baseline':}")
        print(f"  {'Model':<20} {'MLX Model ID':<38} {'Status':<18} {'Time':>8}")
        print(f"  {'-'*20} {'-'*38} {'-'*18} {'-'*8}")
        for entry in mlx_run_log:
            print(
                f"  {entry['model']:<20} {entry['mlx_model_id']:<38} "
                f"{entry['status']:<18} {entry['elapsed_s']:>7.1f}s"
            )

    print(f"\n  Manifest: {manifest_path}")

    # Exit with error if any model failed (skipped is not an error)
    all_results = run_log + mlx_run_log
    failures = [e for e in all_results if e["status"] == "error"]
    if failures:
        print(f"\n  WARNING: {len(failures)} model(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
