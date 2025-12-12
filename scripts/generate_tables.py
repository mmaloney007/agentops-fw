#!/usr/bin/env python3
"""
One-shot helper to generate paper-style tables:
- Summarize train/eval logs into CSV/markdown
- (Optionally) sweep checkpoints and append per-ckpt metrics

Usage:
  python scripts/generate_tables.py \
    --run-dir out/my_run \
    --tasks tasks/fc_tasks.jsonl \
    --steps 100 \
    --samples 1 \
    --out table.csv \
    --sweep-checkpoints

Notes:
- "run" = training output dir (passed via --out), containing train_log.jsonl/eval.jsonl and checkpoints/.
- If --sweep-checkpoints is set, we evaluate all checkpoints under run_dir/checkpoints on the provided tasks.
"""
import argparse, os, json
from pathlib import Path
from typing import List

import pandas as pd

from scripts.summarize_logs import summarize_file
from scripts.eval_checkpoints import eval_checkpoint, _load_tasks
from agent_stable_slo.utils.repro import env_snapshot, atomic_write_json
from agent_stable_slo.utils.hardware import detect_hardware, recommended_defaults


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory (e.g., out/my_run)")
    ap.add_argument("--tasks", type=str, default=None, help="Tasks JSONL for checkpoint sweep (e.g., tasks/fc_tasks.jsonl)")
    ap.add_argument("--steps", type=int, default=100, help="Steps per checkpoint sweep (if enabled)")
    ap.add_argument("--samples", type=int, default=1, help="Samples per task during checkpoint sweep")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--sweep-checkpoints", action="store_true", help="If set, evaluate all checkpoints in the run_dir")
    ap.add_argument("--max-ckpts", type=int, default=0, help="Limit checkpoints evaluated during sweep (0 = all).")
    ap.add_argument("--out", type=str, default="table.csv", help="CSV output path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    # Summarize logs
    inputs: List[Path] = []
    for cand in ["train_log.jsonl", "eval.jsonl"]:
        p = run_dir / cand
        if p.exists():
            inputs.append(p)
    if not inputs:
        raise SystemExit(f"No train_log.jsonl or eval.jsonl found under {run_dir}")

    summaries = [summarize_file(p) for p in inputs]

    # Optional checkpoint sweep
    sweep_summaries = []
    if args.sweep_checkpoints:
        tasks_path = args.tasks
        if tasks_path is None:
            # try manifest
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
                tasks_path = manifest.get("config", {}).get("tasks") or manifest.get("tasks")
        if tasks_path is None:
            raise SystemExit("tasks file required for checkpoint sweep (pass --tasks or ensure manifest has tasks)")
        tasks = _load_tasks(tasks_path)
        hw = detect_hardware()
        hw_cfg = hw.as_dict()
        hw_cfg["recommended"] = recommended_defaults(hw)
        base_model = None
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
            base_model = manifest.get("config", {}).get("base_model") or manifest.get("base_model")
        if base_model is None:
            raise SystemExit("base_model not found in manifest; please set base_model manually in manifest or pass different run_dir")

        ckpts = sorted((run_dir / "checkpoints").glob("step_*"))
        if args.max_ckpts and len(ckpts) > args.max_ckpts:
            ckpts = ckpts[-args.max_ckpts :]
        if not ckpts:
            raise SystemExit(f"No checkpoints found under {run_dir}/checkpoints")
        for ck in ckpts:
            summary = eval_checkpoint(
                ck,
                base_model,
                tasks,
                args.steps,
                args.max_new_tokens,
                hw_cfg.get("recommended", {}).get("torch_dtype", "float16"),
                False,
                args.temperature,
                args.top_p,
                args.deterministic,
                args.samples,
                float(os.getenv("LAMBDA_LATENCY", "0.0")),
                float(os.getenv("MU_COST", "0.0")),
                float(os.getenv("GAMMA_STABILITY", "0.0")),
            )
            sweep_summaries.append(summary)

    df = pd.DataFrame(summaries)
    if sweep_summaries:
        df_ck = pd.DataFrame(sweep_summaries)
        df = pd.concat([df, df_ck], ignore_index=True, sort=False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_markdown(index=False))
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
