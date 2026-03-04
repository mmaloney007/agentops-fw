#!/usr/bin/env python3
"""
P9 ANE Pipeline Demo -- comprehensive end-to-end demonstration of the
heterogeneous ANE+MLX inference and training pipeline.

Walks through the complete P9 pipeline with clear section headers and timing:
  1. Environment check (coremltools, torch, transformers, mlx, ANE)
  2. Model conversion (Qwen3.5-0.8B to CoreML/ANE format)
  3. ANE inference demo (5 sample prompts)
  4. MLX inference comparison (same 5 prompts, head-to-head)
     Default MLX model: Qwen3.5-0.8B-4bit (pre-quantized for Metal GPU)
     This provides a same-architecture comparison: ANE CoreML vs MLX GPU.
  5. Power profile (optional, requires sudo)
  6. Hybrid GRPO training demo (5 steps)
  7. Summary report

Usage:
  python scripts/demo_p9_ane.py
  python scripts/demo_p9_ane.py --skip-convert --skip-grpo
  python scripts/demo_p9_ane.py --measure-power
  python scripts/demo_p9_ane.py --ane-meta-dir models/ane/qwen3.5-0.8b
  python scripts/demo_p9_ane.py --mlx-model mlx-community/Qwen3.5-0.8B-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Section utilities
# ---------------------------------------------------------------------------

_SECTION_NUM = 0


def _section(title: str) -> None:
    """Print a numbered section header."""
    global _SECTION_NUM
    _SECTION_NUM += 1
    print(f"\n{'='*72}")
    print(f"  [{_SECTION_NUM}] {title}")
    print(f"{'='*72}\n")


def _subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n  --- {title} ---\n")


def _kvprint(key: str, value: Any, indent: int = 4) -> None:
    """Print a key-value pair with indentation."""
    print(f"{' ' * indent}{key:<28} {value}")


# ---------------------------------------------------------------------------
# Demo prompts -- 5 diverse tasks with JSON schemas
# ---------------------------------------------------------------------------

DEMO_PROMPTS: List[Dict[str, Any]] = [
    {
        "name": "Intent Classification",
        "prompt": "Book a flight to Paris",
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["intent", "confidence"],
        },
    },
    {
        "name": "Question Answering",
        "prompt": "What is the capital of France?",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
        },
    },
    {
        "name": "Tool Call",
        "prompt": "Search for weather in NYC",
        "schema": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["tool_name", "arguments"],
        },
    },
    {
        "name": "Math Reasoning",
        "prompt": "What is 15 * 23?",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
    },
    {
        "name": "Code Generation",
        "prompt": "Write a hello world in Python",
        "schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "language": {"type": "string"},
            },
            "required": ["code", "language"],
        },
    },
]


# ---------------------------------------------------------------------------
# Section 1: Environment Check
# ---------------------------------------------------------------------------


def section_environment_check() -> Dict[str, Any]:
    """Verify required packages and hardware.  Returns version info dict."""
    _section("Environment Check")

    env_info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "apple_silicon": False,
        "packages": {},
    }

    # Apple Silicon check
    chip = ""
    try:
        chip = (
            subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        pass
    env_info["chip"] = chip
    env_info["apple_silicon"] = "Apple" in chip

    _kvprint("Platform", env_info["platform"])
    _kvprint("Python", env_info["python"])
    _kvprint("macOS", env_info["macos"])
    _kvprint("Chip", chip or "(unknown)")
    _kvprint("Apple Silicon", "Yes" if env_info["apple_silicon"] else "No")

    if not env_info["apple_silicon"]:
        print("\n    WARNING: Apple Silicon not detected. ANE features will not work.")

    # Check packages
    _subsection("Package Versions")

    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "coremltools": "coremltools",
        "mlx": "mlx",
        "mlx_lm": "mlx_lm",
        "numpy": "numpy",
    }

    for label, module_name in packages.items():
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "installed (version unknown)")
            env_info["packages"][label] = version
            _kvprint(label, version)
        except ImportError:
            env_info["packages"][label] = "NOT INSTALLED"
            _kvprint(label, "NOT INSTALLED")

    # Memory
    try:
        mem_bytes = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        mem_gb = round(mem_bytes / (1024**3), 1)
        env_info["memory_gb"] = mem_gb
        _kvprint("Unified Memory", f"{mem_gb} GB")
    except Exception:
        env_info["memory_gb"] = 0

    print()
    return env_info


# ---------------------------------------------------------------------------
# Section 2: Model Conversion
# ---------------------------------------------------------------------------


def section_model_conversion(
    hf_model: str,
    ane_meta_dir: str,
    skip: bool = False,
) -> Dict[str, Any]:
    """Convert HF model to ANE format (or skip if already done / requested)."""
    _section("Model Conversion")

    result: Dict[str, Any] = {"status": "skipped", "elapsed_s": 0.0}

    if skip:
        print("    Skipping model conversion (--skip-convert).")
        return result

    meta_path = Path(ane_meta_dir) / "meta.yaml"
    if meta_path.exists():
        print(f"    ANE model already exists at: {ane_meta_dir}")
        print(f"    meta.yaml found -- skipping conversion.")
        result["status"] = "already_converted"
        return result

    # Check for Anemll converter
    print(f"    HF model    : {hf_model}")
    print(f"    Target dir  : {ane_meta_dir}")
    print()

    # Try to convert via the convert_ane_models.py script
    project_root = Path(__file__).resolve().parents[1]
    convert_script = project_root / "scripts" / "convert_ane_models.py"

    if not convert_script.exists():
        print(f"    ERROR: Conversion script not found: {convert_script}")
        print("    Please convert the model manually:")
        print(f"      python scripts/convert_ane_models.py --model {hf_model}")
        result["status"] = "error"
        return result

    cmd = [
        sys.executable,
        str(convert_script),
        "--model",
        hf_model,
        "--output",
        str(Path(ane_meta_dir).parent),
    ]

    print(f"    Running: {' '.join(cmd)}")
    print()

    t0 = time.time()
    try:
        ret = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)

        if ret.returncode == 0:
            result["status"] = "ok"
            print(f"\n    Conversion completed in {elapsed:.1f}s")
        else:
            result["status"] = "error"
            print(f"\n    Conversion failed (returncode={ret.returncode})")
            print("    You may need to install Anemll first:")
            print("      git clone https://github.com/Anemll/Anemll.git ~/Projects/anemll")
            print("      export ANEMLL_HOME=~/Projects/anemll")
    except Exception as exc:
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        result["status"] = "error"
        print(f"    Exception during conversion: {exc}")

    return result


# ---------------------------------------------------------------------------
# Section 3: ANE Inference Demo
# ---------------------------------------------------------------------------


def section_ane_inference(
    ane_meta_dir: str,
    hf_model: str,
) -> Dict[str, Any]:
    """Run 5 sample prompts through ANE and collect results."""
    _section("ANE Inference Demo")

    results: Dict[str, Any] = {
        "available": False,
        "prompts": [],
        "latencies_ms": [],
        "json_parse_count": 0,
        "mean_latency_ms": 0.0,
    }

    # Check that the ANE model directory exists
    meta_path = Path(ane_meta_dir) / "meta.yaml"
    if not meta_path.exists():
        print(f"    ANE model not found at: {ane_meta_dir}")
        print(f"    Expected meta.yaml at: {meta_path}")
        print()
        print("    To convert the model, run:")
        print(f"      python scripts/convert_ane_models.py --model {hf_model}")
        print()
        print("    Skipping ANE inference demo.")
        return results

    # Configure environment for ane_local
    os.environ["ANE_META_DIR"] = str(Path(ane_meta_dir).resolve())
    os.environ["ANE_HF_MODEL"] = hf_model
    os.environ["ANE_MAX_TOKENS"] = "256"

    try:
        from agent_stable_slo.rollout.providers import ane_local
    except ImportError as exc:
        print(f"    Failed to import ane_local: {exc}")
        print("    Make sure coremltools is installed: pip install coremltools")
        return results

    results["available"] = True

    for i, demo in enumerate(DEMO_PROMPTS, 1):
        _subsection(f"Prompt {i}: {demo['name']}")
        print(f"    Prompt : {demo['prompt']}")
        print(f"    Schema : {json.dumps(demo['schema'], separators=(',', ':'))}")
        print()

        try:
            t0 = time.time()
            raw_text, parsed, latency_ms, ttft_ms, tokens_in, tokens_out = (
                ane_local.generate_raw(
                    prompt=demo["prompt"],
                    schema=demo["schema"],
                    mode="structured",
                    temperature=0.0,
                    max_tokens=256,
                )
            )
            wall_ms = (time.time() - t0) * 1000.0

            json_ok = bool(parsed and isinstance(parsed, dict) and len(parsed) > 0)
            results["latencies_ms"].append(latency_ms)
            results["json_parse_count"] += int(json_ok)

            _kvprint("Raw output", repr(raw_text[:200]))
            _kvprint("Parsed JSON", json.dumps(parsed, ensure_ascii=False)[:200])
            _kvprint("JSON valid", "Yes" if json_ok else "No")
            _kvprint("Latency (provider)", f"{latency_ms:.1f} ms")
            _kvprint("TTFT", f"{ttft_ms:.1f} ms")
            _kvprint("Tokens in", tokens_in)
            _kvprint("Tokens out", tokens_out)

            results["prompts"].append({
                "name": demo["name"],
                "raw": raw_text[:500],
                "parsed": parsed,
                "json_ok": json_ok,
                "latency_ms": round(latency_ms, 1),
                "ttft_ms": round(ttft_ms, 1),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            })

        except Exception as exc:
            print(f"    ERROR: {exc}")
            results["prompts"].append({
                "name": demo["name"],
                "error": str(exc),
            })

    if results["latencies_ms"]:
        results["mean_latency_ms"] = round(
            sum(results["latencies_ms"]) / len(results["latencies_ms"]), 1
        )

    print(f"\n    ANE Summary: {results['json_parse_count']}/{len(DEMO_PROMPTS)} JSON parsed, "
          f"mean latency {results['mean_latency_ms']:.1f} ms")

    return results


# ---------------------------------------------------------------------------
# Section 4: MLX Comparison
# ---------------------------------------------------------------------------


def section_mlx_comparison(mlx_model: str) -> Dict[str, Any]:
    """Run same 5 prompts through MLX for head-to-head comparison."""
    _section("MLX Inference Comparison")

    results: Dict[str, Any] = {
        "available": False,
        "prompts": [],
        "latencies_ms": [],
        "json_parse_count": 0,
        "mean_latency_ms": 0.0,
    }

    # Configure environment for mlx_local
    os.environ["MLX_MODEL"] = mlx_model
    os.environ["MLX_MAX_TOKENS"] = "256"
    os.environ["MLX_ENABLE_THINKING"] = "0"

    try:
        from agent_stable_slo.rollout.providers import mlx_local
    except ImportError as exc:
        print(f"    Failed to import mlx_local: {exc}")
        print("    Make sure mlx and mlx-lm are installed: pip install mlx mlx-lm")
        return results

    results["available"] = True
    print(f"    MLX model: {mlx_model}\n")

    for i, demo in enumerate(DEMO_PROMPTS, 1):
        _subsection(f"Prompt {i}: {demo['name']}")
        print(f"    Prompt : {demo['prompt']}")
        print()

        try:
            t0 = time.time()
            raw_text, parsed, latency_ms, ttft_ms, tokens_in, tokens_out = (
                mlx_local.generate_raw(
                    prompt=demo["prompt"],
                    schema=demo["schema"],
                    mode="structured",
                    temperature=0.0,
                    max_tokens=256,
                )
            )
            wall_ms = (time.time() - t0) * 1000.0

            json_ok = bool(parsed and isinstance(parsed, dict) and len(parsed) > 0)
            results["latencies_ms"].append(latency_ms)
            results["json_parse_count"] += int(json_ok)

            _kvprint("Raw output", repr(raw_text[:200]))
            _kvprint("Parsed JSON", json.dumps(parsed, ensure_ascii=False)[:200])
            _kvprint("JSON valid", "Yes" if json_ok else "No")
            _kvprint("Latency (provider)", f"{latency_ms:.1f} ms")
            _kvprint("Tokens in", tokens_in)
            _kvprint("Tokens out", tokens_out)

            results["prompts"].append({
                "name": demo["name"],
                "raw": raw_text[:500],
                "parsed": parsed,
                "json_ok": json_ok,
                "latency_ms": round(latency_ms, 1),
                "ttft_ms": round(ttft_ms, 1),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            })

        except Exception as exc:
            print(f"    ERROR: {exc}")
            results["prompts"].append({
                "name": demo["name"],
                "error": str(exc),
            })

    if results["latencies_ms"]:
        results["mean_latency_ms"] = round(
            sum(results["latencies_ms"]) / len(results["latencies_ms"]), 1
        )

    print(f"\n    MLX Summary: {results['json_parse_count']}/{len(DEMO_PROMPTS)} JSON parsed, "
          f"mean latency {results['mean_latency_ms']:.1f} ms")

    # Print side-by-side latency table if we have results
    return results


def print_latency_comparison(
    ane_results: Dict[str, Any],
    mlx_results: Dict[str, Any],
) -> None:
    """Print side-by-side latency comparison table."""
    _subsection("Head-to-Head Latency Comparison")

    header = f"    {'Prompt':<25} {'ANE (ms)':>12} {'MLX (ms)':>12} {'Winner':>10}"
    print(header)
    print(f"    {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    for i, demo in enumerate(DEMO_PROMPTS):
        name = demo["name"]

        ane_lat = "--"
        mlx_lat = "--"
        winner = "--"

        if i < len(ane_results.get("prompts", [])):
            ane_entry = ane_results["prompts"][i]
            if "latency_ms" in ane_entry:
                ane_lat = f"{ane_entry['latency_ms']:.1f}"

        if i < len(mlx_results.get("prompts", [])):
            mlx_entry = mlx_results["prompts"][i]
            if "latency_ms" in mlx_entry:
                mlx_lat = f"{mlx_entry['latency_ms']:.1f}"

        if ane_lat != "--" and mlx_lat != "--":
            winner = "ANE" if float(ane_lat) < float(mlx_lat) else "MLX"

        print(f"    {name:<25} {ane_lat:>12} {mlx_lat:>12} {winner:>10}")

    print()


# ---------------------------------------------------------------------------
# Section 5: Power Profile
# ---------------------------------------------------------------------------


def section_power_profile(
    ane_meta_dir: str,
    hf_model: str,
    mlx_model: str,
    measure: bool = False,
) -> Dict[str, Any]:
    """Run power monitoring during ANE and MLX inference batches."""
    _section("Power Profile")

    results: Dict[str, Any] = {
        "measured": False,
        "ane_power": {},
        "mlx_power": {},
    }

    if not measure:
        print("    Skipping power profiling (use --measure-power to enable).")
        print("    Requires: sudo -n powermetrics (passwordless sudo).")
        return results

    try:
        from agent_stable_slo.bench.power_monitor import PowerMonitor
    except ImportError:
        print("    PowerMonitor not available. Skipping power profiling.")
        return results

    # Use a simple schema for repeated inference
    simple_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    simple_prompt = "What is 2 + 2?"

    # ANE power measurement (10 inferences)
    ane_available = Path(ane_meta_dir).joinpath("meta.yaml").exists()
    if ane_available:
        _subsection("ANE Power (10 inferences)")
        os.environ["ANE_META_DIR"] = str(Path(ane_meta_dir).resolve())
        os.environ["ANE_HF_MODEL"] = hf_model
        os.environ["ANE_MAX_TOKENS"] = "128"

        try:
            from agent_stable_slo.rollout.providers import ane_local

            mon = PowerMonitor(interval_ms=100)
            mon.start()

            for i in range(10):
                ane_local.generate_raw(
                    prompt=simple_prompt,
                    schema=simple_schema,
                    temperature=0.0,
                    max_tokens=128,
                )
                print(f"      ANE inference {i+1}/10 done")

            summary = mon.stop()
            results["ane_power"] = summary.to_dict()
            results["measured"] = True

            _kvprint("Mean CPU", f"{summary.mean_cpu_w:.2f} W")
            _kvprint("Mean GPU", f"{summary.mean_gpu_w:.2f} W")
            _kvprint("Mean ANE", f"{summary.mean_ane_w:.2f} W")
            _kvprint("Mean Total", f"{summary.mean_total_w:.2f} W")
            _kvprint("Energy", f"{summary.energy_j:.2f} J")
            _kvprint("Duration", f"{summary.duration_s:.2f} s")

        except Exception as exc:
            print(f"    ANE power measurement error: {exc}")
    else:
        print("    ANE model not available -- skipping ANE power measurement.")

    # MLX power measurement (10 inferences)
    _subsection("MLX Power (10 inferences)")
    os.environ["MLX_MODEL"] = mlx_model
    os.environ["MLX_MAX_TOKENS"] = "128"
    os.environ["MLX_ENABLE_THINKING"] = "0"

    try:
        from agent_stable_slo.rollout.providers import mlx_local

        mon = PowerMonitor(interval_ms=100)
        mon.start()

        for i in range(10):
            mlx_local.generate_raw(
                prompt=simple_prompt,
                schema=simple_schema,
                temperature=0.0,
                max_tokens=128,
            )
            print(f"      MLX inference {i+1}/10 done")

        summary = mon.stop()
        results["mlx_power"] = summary.to_dict()
        results["measured"] = True

        _kvprint("Mean CPU", f"{summary.mean_cpu_w:.2f} W")
        _kvprint("Mean GPU", f"{summary.mean_gpu_w:.2f} W")
        _kvprint("Mean ANE", f"{summary.mean_ane_w:.2f} W")
        _kvprint("Mean Total", f"{summary.mean_total_w:.2f} W")
        _kvprint("Energy", f"{summary.energy_j:.2f} J")
        _kvprint("Duration", f"{summary.duration_s:.2f} s")

    except Exception as exc:
        print(f"    MLX power measurement error: {exc}")

    # Power comparison table
    if results["ane_power"] and results["mlx_power"]:
        _subsection("Power Comparison")
        ane_p = results["ane_power"]
        mlx_p = results["mlx_power"]
        header = f"    {'Metric':<25} {'ANE':>12} {'MLX':>12}"
        print(header)
        print(f"    {'-'*25} {'-'*12} {'-'*12}")
        for metric in ["mean_cpu_w", "mean_gpu_w", "mean_ane_w", "mean_total_w", "energy_j"]:
            label = metric.replace("mean_", "").replace("_", " ").title()
            ane_val = ane_p.get(metric, 0.0)
            mlx_val = mlx_p.get(metric, 0.0)
            print(f"    {label:<25} {ane_val:>12.2f} {mlx_val:>12.2f}")

    return results


# ---------------------------------------------------------------------------
# Section 6: Hybrid GRPO Demo
# ---------------------------------------------------------------------------


def section_hybrid_grpo(
    ane_meta_dir: str,
    hf_model: str,
    skip: bool = False,
) -> Dict[str, Any]:
    """Demonstrate hybrid ANE+MLX GRPO training for 5 steps."""
    _section("Hybrid GRPO Training Demo")

    results: Dict[str, Any] = {
        "available": False,
        "steps": [],
        "mean_step_ms": 0.0,
        "final_reward": 0.0,
    }

    if skip:
        print("    Skipping GRPO demo (--skip-grpo).")
        return results

    # Check prerequisites
    meta_path = Path(ane_meta_dir) / "meta.yaml"
    if not meta_path.exists():
        print(f"    ANE model not found at: {ane_meta_dir}")
        print("    Hybrid GRPO requires a converted ANE model.")
        print(f"    Convert first: python scripts/convert_ane_models.py --model {hf_model}")
        return results

    # Check for task files
    project_root = Path(__file__).resolve().parents[1]
    task_candidates = [
        project_root / "tasks" / "clinc_en.jsonl",
        project_root / "tasks" / "public_gsm8k.jsonl",
        project_root / "tasks" / "hotpot_dev.jsonl",
    ]
    task_file = None
    for candidate in task_candidates:
        if candidate.exists():
            task_file = str(candidate)
            break

    if task_file is None:
        print("    No task files found in tasks/ directory.")
        print("    GRPO demo requires at least one JSONL task file.")
        return results

    try:
        from agent_stable_slo.train.ane_grpo_adapter import (
            HybridGRPOConfig,
            HybridGRPOTrainer,
        )
    except ImportError as exc:
        print(f"    Failed to import HybridGRPOTrainer: {exc}")
        print("    Install: pip install mlx mlx-lm")
        return results

    print(f"    ANE model dir : {ane_meta_dir}")
    print(f"    Base model    : {hf_model}")
    print(f"    Task file     : {task_file}")
    print(f"    Steps         : 5")
    print(f"    Group size    : 2")
    print()

    try:
        cfg = HybridGRPOConfig(
            base_model=hf_model,
            ane_meta_dir=str(Path(ane_meta_dir).resolve()),
            tasks=[task_file],
            num_steps=5,
            group_size=2,
            max_tokens=128,
            temperature=0.7,
            lora_rank=4,
            lora_layers=4,
            checkpoint_every=0,  # No checkpointing for demo
            seed=42,
        )

        # We do a manual training loop so we can capture per-step timings
        # rather than calling trainer.run() which writes to its own log.
        trainer = HybridGRPOTrainer(cfg)
        results["available"] = True

        # The trainer.run() method does everything internally and logs to JSONL.
        # For the demo, we call run() and then parse its log file.
        print("    Starting hybrid GRPO training...\n")

        t0 = time.time()
        adapter_path = trainer.run()
        total_ms = (time.time() - t0) * 1000.0

        # Parse the training log for per-step breakdowns
        log_path = trainer.log_path
        if log_path.exists():
            with open(log_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        step_info = {
                            "step": rec.get("step", 0),
                            "reward": rec.get("mean_reward", 0.0),
                            "loss": rec.get("loss", 0.0),
                            "ane_rollout_ms": rec.get("ane_rollout_ms", 0.0),
                            "mlx_gradient_ms": rec.get("mlx_gradient_ms", 0.0),
                            "weight_sync_ms": rec.get("weight_sync_ms", 0.0),
                            "step_ms": rec.get("step_ms", 0.0),
                        }
                        results["steps"].append(step_info)
                    except json.JSONDecodeError:
                        continue

        # Print per-step timing breakdown
        if results["steps"]:
            _subsection("Per-Step Timing Breakdown")
            header = (
                f"    {'Step':>4}  {'ANE Rollout':>14}  {'MLX Grad':>12}  "
                f"{'Wt Sync':>10}  {'Total':>10}  {'Reward':>8}"
            )
            print(header)
            print(f"    {'----':>4}  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")

            for s in results["steps"]:
                print(
                    f"    {s['step']:>4}  "
                    f"{s['ane_rollout_ms']:>12.1f}ms  "
                    f"{s['mlx_gradient_ms']:>10.1f}ms  "
                    f"{s['weight_sync_ms']:>8.1f}ms  "
                    f"{s['step_ms']:>8.1f}ms  "
                    f"{s['reward']:>8.3f}"
                )

            step_times = [s["step_ms"] for s in results["steps"]]
            results["mean_step_ms"] = round(
                sum(step_times) / len(step_times), 1
            )
            results["final_reward"] = round(results["steps"][-1]["reward"], 4)

            _subsection("Reward Progression")
            for s in results["steps"]:
                bar_len = max(0, int(s["reward"] * 20))
                bar = "#" * bar_len
                print(f"    Step {s['step']:>2}: {s['reward']:>7.3f}  {bar}")

        print(f"\n    Adapter saved to: {adapter_path}")
        print(f"    Training log: {log_path}")
        print(f"    Total training time: {total_ms:.0f} ms")

    except Exception as exc:
        print(f"    GRPO training error: {exc}")
        import traceback
        traceback.print_exc()

    return results


# ---------------------------------------------------------------------------
# Section 7: Summary Report
# ---------------------------------------------------------------------------


def section_summary(
    env_info: Dict[str, Any],
    conversion_result: Dict[str, Any],
    ane_results: Dict[str, Any],
    mlx_results: Dict[str, Any],
    power_results: Dict[str, Any],
    grpo_results: Dict[str, Any],
) -> None:
    """Print the final formatted summary report."""
    _section("Summary Report")

    # Compute speedup
    ane_mean = ane_results.get("mean_latency_ms", 0.0)
    mlx_mean = mlx_results.get("mean_latency_ms", 0.0)
    ane_parsed = ane_results.get("json_parse_count", 0)
    mlx_parsed = mlx_results.get("json_parse_count", 0)

    if ane_mean > 0 and mlx_mean > 0:
        speedup = mlx_mean / ane_mean
        speedup_str = f"{speedup:.2f}x"
        faster = "ANE" if speedup > 1.0 else "MLX"
        speedup_label = f"{speedup_str} ({faster} faster)"
    else:
        speedup_label = "N/A (incomplete data)"

    grpo_step_ms = grpo_results.get("mean_step_ms", 0.0)
    grpo_reward = grpo_results.get("final_reward", 0.0)

    # Box drawing
    w = 50
    print(f"    +{'='*w}+")
    print(f"    |{'P9 ANE Pipeline Demo Results':^{w}}|")
    print(f"    +{'='*w}+")

    # Environment
    print(f"    | {'Environment':<{w-1}}|")
    print(f"    |   {'Chip:':<20} {env_info.get('chip', 'N/A'):<{w-23}}|")
    print(f"    |   {'Memory:':<20} {str(env_info.get('memory_gb', 'N/A')) + ' GB':<{w-23}}|")
    print(f"    +{'-'*w}+")

    # Conversion
    conv_status = conversion_result.get("status", "N/A")
    print(f"    | {'Model Conversion':<{w-1}}|")
    print(f"    |   {'Status:':<20} {conv_status:<{w-23}}|")
    print(f"    +{'-'*w}+")

    # ANE Inference
    print(f"    | {'ANE Inference (5 prompts)':<{w-1}}|")
    if ane_results.get("available"):
        ane_lat_str = f"{ane_mean:.1f} ms"
        ane_parse_str = f"{ane_parsed}/5"
        print(f"    |   {'Mean latency:':<20} {ane_lat_str:<{w-23}}|")
        print(f"    |   {'JSON parse rate:':<20} {ane_parse_str:<{w-23}}|")
    else:
        print(f"    |   {'Status:':<20} {'Not available':<{w-23}}|")
    print(f"    +{'-'*w}+")

    # MLX Inference
    print(f"    | {'MLX Inference (5 prompts)':<{w-1}}|")
    if mlx_results.get("available"):
        mlx_lat_str = f"{mlx_mean:.1f} ms"
        mlx_parse_str = f"{mlx_parsed}/5"
        print(f"    |   {'Mean latency:':<20} {mlx_lat_str:<{w-23}}|")
        print(f"    |   {'JSON parse rate:':<20} {mlx_parse_str:<{w-23}}|")
    else:
        print(f"    |   {'Status:':<20} {'Not available':<{w-23}}|")
    print(f"    +{'-'*w}+")

    # Speedup
    print(f"    | {'ANE vs MLX Speedup':<{w-1}}|")
    print(f"    |   {'Speedup:':<20} {speedup_label:<{w-23}}|")
    print(f"    +{'-'*w}+")

    # Power (if measured)
    if power_results.get("measured"):
        print(f"    | {'Power Profile':<{w-1}}|")
        ane_p = power_results.get("ane_power", {})
        mlx_p = power_results.get("mlx_power", {})
        if ane_p:
            ane_pw = f"{ane_p.get('mean_total_w', 0):.2f} W"
            print(f"    |   {'ANE mean power:':<20} {ane_pw:<{w-23}}|")
        if mlx_p:
            mlx_pw = f"{mlx_p.get('mean_total_w', 0):.2f} W"
            print(f"    |   {'MLX mean power:':<20} {mlx_pw:<{w-23}}|")
        if ane_p and mlx_p:
            ane_e = ane_p.get("energy_j", 0)
            mlx_e = mlx_p.get("energy_j", 0)
            if ane_e > 0 and mlx_e > 0:
                eff = f"{mlx_e / ane_e:.2f}x (ANE more efficient)"
                print(f"    |   {'Efficiency:':<20} {eff:<{w-23}}|")
        print(f"    +{'-'*w}+")

    # GRPO Training
    print(f"    | {'Hybrid GRPO (5 steps)':<{w-1}}|")
    if grpo_results.get("available"):
        step_str = f"{grpo_step_ms:.1f} ms"
        reward_str = f"{grpo_reward:.4f}"
        print(f"    |   {'Mean step time:':<20} {step_str:<{w-23}}|")
        print(f"    |   {'Final reward:':<20} {reward_str:<{w-23}}|")
    else:
        status = "Skipped" if not grpo_results.get("steps") else "Error"
        print(f"    |   {'Status:':<20} {status:<{w-23}}|")
    print(f"    +{'='*w}+")

    print()


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="P9 ANE Pipeline Demo -- end-to-end demonstration of "
                    "heterogeneous ANE+MLX inference and training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--ane-meta-dir",
        default="models/ane/qwen2.5-0.5b",
        help="Path to converted ANE model directory (default: models/ane/qwen2.5-0.5b).",
    )
    ap.add_argument(
        "--hf-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID for tokenizer and GRPO base (default: Qwen/Qwen2.5-0.5B-Instruct).",
    )
    ap.add_argument(
        "--mlx-model",
        default="mlx-community/Qwen3.5-0.8B-4bit",
        help="MLX model ID for comparison inference (default: mlx-community/Qwen3.5-0.8B-4bit).",
    )
    ap.add_argument(
        "--measure-power",
        action="store_true",
        help="Enable power profiling via powermetrics (requires sudo -n).",
    )
    ap.add_argument(
        "--skip-grpo",
        action="store_true",
        help="Skip the hybrid GRPO training demo.",
    )
    ap.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip the model conversion step.",
    )

    args = ap.parse_args()

    # Ensure project root is on sys.path
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    overall_t0 = time.time()

    print()
    print("=" * 72)
    print("       P9 ANE Pipeline Demo")
    print("       Heterogeneous Apple Neural Engine + MLX Inference & Training")
    print("=" * 72)

    # Accumulate results from each section; each section is wrapped in
    # try/except so failures are non-fatal.

    # Section 1: Environment Check
    try:
        env_info = section_environment_check()
    except Exception as exc:
        print(f"    Environment check failed: {exc}")
        env_info = {}

    # Section 2: Model Conversion
    try:
        conversion_result = section_model_conversion(
            hf_model=args.hf_model,
            ane_meta_dir=args.ane_meta_dir,
            skip=args.skip_convert,
        )
    except Exception as exc:
        print(f"    Model conversion failed: {exc}")
        conversion_result = {"status": "error"}

    # Section 3: ANE Inference Demo
    try:
        ane_results = section_ane_inference(
            ane_meta_dir=args.ane_meta_dir,
            hf_model=args.hf_model,
        )
    except Exception as exc:
        print(f"    ANE inference failed: {exc}")
        ane_results = {"available": False}

    # Section 4: MLX Comparison
    try:
        mlx_results = section_mlx_comparison(mlx_model=args.mlx_model)
    except Exception as exc:
        print(f"    MLX comparison failed: {exc}")
        mlx_results = {"available": False}

    # Print latency comparison if both are available
    if ane_results.get("available") and mlx_results.get("available"):
        try:
            print_latency_comparison(ane_results, mlx_results)
        except Exception:
            pass

    # Section 5: Power Profile
    try:
        power_results = section_power_profile(
            ane_meta_dir=args.ane_meta_dir,
            hf_model=args.hf_model,
            mlx_model=args.mlx_model,
            measure=args.measure_power,
        )
    except Exception as exc:
        print(f"    Power profiling failed: {exc}")
        power_results = {"measured": False}

    # Section 6: Hybrid GRPO Demo
    try:
        grpo_results = section_hybrid_grpo(
            ane_meta_dir=args.ane_meta_dir,
            hf_model=args.hf_model,
            skip=args.skip_grpo,
        )
    except Exception as exc:
        print(f"    GRPO demo failed: {exc}")
        grpo_results = {"available": False}

    # Section 7: Summary Report (always runs)
    try:
        section_summary(
            env_info=env_info,
            conversion_result=conversion_result,
            ane_results=ane_results,
            mlx_results=mlx_results,
            power_results=power_results,
            grpo_results=grpo_results,
        )
    except Exception as exc:
        print(f"    Summary report generation failed: {exc}")

    overall_elapsed = time.time() - overall_t0
    print(f"    Total demo time: {overall_elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
