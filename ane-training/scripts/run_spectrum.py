#!/usr/bin/env python3
"""
Run the full spectrum experiment: public vs private ANE GRPO training.

Launches both grpo_public and grpo_private binaries for each model,
collects JSONL results, and generates comparison.

Usage:
    python scripts/run_spectrum.py --models stories110m qwen2.5-0.5b \
                                    --steps 5 --out-dir out/spectrum
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def find_binary(name: str) -> str:
    """Find compiled binary in ane-training/."""
    script_dir = Path(__file__).parent.parent
    binary = script_dir / name
    if binary.exists():
        return str(binary)
    raise FileNotFoundError(f"Binary not found: {binary}. Run 'make {name}' first.")


def find_weights(model: str, weights_dir: str) -> dict:
    """Find weight and tokenizer files for a model."""
    model_dir = Path(weights_dir) / model

    # Find safetensors file
    safetensors = list(model_dir.glob("*.safetensors"))
    if not safetensors:
        # Try model.safetensors specifically
        safetensors = [model_dir / "model.safetensors"]

    tokenizer = model_dir / "tokenizer.json"

    return {
        "weights": str(safetensors[0]) if safetensors else None,
        "tokenizer": str(tokenizer) if tokenizer.exists() else None,
    }


def run_experiment(binary: str, model: str, backend: str,
                   weights: str, tokenizer: str, tasks: str,
                   steps: int, out_dir: str) -> dict:
    """Run one cell of the experiment matrix."""
    cell_dir = os.path.join(out_dir, f"{model}_{backend}")
    os.makedirs(cell_dir, exist_ok=True)

    cmd = [
        binary,
        "--model", model,
        "--weights", weights,
        "--tokenizer", tokenizer,
        "--tasks", tasks,
        "--steps", str(steps),
        "--out-dir", cell_dir,
    ]

    print(f"\n{'='*60}")
    print(f"Running: {model} / {backend}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"FAILED (exit {result.returncode})")
            print(f"stderr: {result.stderr[:500]}")
            return {"status": "failed", "error": result.stderr[:500], "elapsed_s": elapsed}

        print(f"Completed in {elapsed:.1f}s")

        # Read JSONL log
        log_path = os.path.join(cell_dir, "training.jsonl")
        steps_data = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    if line.strip():
                        steps_data.append(json.loads(line))

        return {
            "status": "success",
            "elapsed_s": elapsed,
            "steps": steps_data,
            "model": model,
            "backend": backend,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_s": 3600}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run public vs private ANE GRPO spectrum")
    parser.add_argument("--models", nargs="+", default=["stories110m", "qwen2.5-0.5b"],
                        choices=["stories110m", "qwen2.5-0.5b"])
    parser.add_argument("--backends", nargs="+", default=["public", "private"],
                        choices=["public", "private"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--tasks", required=True, help="Path to tasks.jsonl")
    parser.add_argument("--weights-dir", default="weights",
                        help="Directory containing model weights")
    parser.add_argument("--out-dir", default="out/spectrum")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Find binaries
    binaries = {
        "public": find_binary("grpo_public"),
        "private": find_binary("grpo_private"),
    }

    # Run all cells
    results = {}
    for model in args.models:
        files = find_weights(model, args.weights_dir)
        if not files["weights"]:
            print(f"WARNING: No weights found for {model} in {args.weights_dir}/{model}/")
            print(f"Run: python scripts/download_weights.py --model {model}")
            continue

        for backend in args.backends:
            key = f"{model}_{backend}"
            results[key] = run_experiment(
                binary=binaries[backend],
                model=model,
                backend=backend,
                weights=files["weights"],
                tokenizer=files["tokenizer"],
                tasks=args.tasks,
                steps=args.steps,
                out_dir=args.out_dir,
            )

    # Save summary
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Model':<20} {'Backend':<10} {'Status':<10} {'Elapsed':<10} {'Mean Reward':<12} {'JSON Valid':<10}")
    print(f"{'-'*80}")
    for key, r in results.items():
        model, backend = key.rsplit("_", 1)
        status = r.get("status", "?")
        elapsed = f"{r.get('elapsed_s', 0):.1f}s"

        if r.get("steps"):
            last_step = r["steps"][-1]
            mean_r = f"{last_step.get('mean_reward', 0):.3f}"
            valid = f"{last_step.get('json_valid_pct', 0):.0f}%"
        else:
            mean_r = "N/A"
            valid = "N/A"

        print(f"{model:<20} {backend:<10} {status:<10} {elapsed:<10} {mean_r:<12} {valid:<10}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
