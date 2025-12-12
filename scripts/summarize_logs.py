#!/usr/bin/env python3
"""
Summarize train/eval JSONL logs into a compact table (reward mean/CI, latency p95/p99, json_valid rate).
Usage examples:
  # Summarize a single run directory (auto-picks train_log.jsonl)
  python scripts/summarize_logs.py --inputs out/my_run --out table.csv

  # Summarize specific files (train + eval)
  python scripts/summarize_logs.py --inputs out/my_run/train_log.jsonl out/my_run/eval.jsonl --out table.csv

Here, "run" means a directory produced by the trainer (e.g., --out out/my_run) containing train_log.jsonl or eval.jsonl.
"""
import argparse, json, math
from pathlib import Path
from typing import List, Dict

import pandas as pd


def _bootstrap_ci(values: List[float], iters: int = 200, alpha: float = 0.05) -> Dict[str, float]:
    if not values:
        return {}
    import random

    rng = random.Random(17)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / float(n))
    means.sort()
    lo_idx = int(alpha / 2 * (iters - 1))
    hi_idx = int((1 - alpha / 2) * (iters - 1))
    return {"mean": sum(values) / float(n), "ci_lower": means[lo_idx], "ci_upper": means[hi_idx]}


def summarize_file(path: Path) -> Dict[str, float]:
    rewards: List[float] = []
    lats: List[float] = []
    ttfts: List[float] = []
    json_valids: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "reward" in rec:
            rewards.append(float(rec["reward"]))
        if "latency_ms" in rec:
            lats.append(float(rec["latency_ms"]))
        if "ttft_ms" in rec:
            ttfts.append(float(rec["ttft_ms"]))
        if "json_valid" in rec:
            json_valids.append(int(rec["json_valid"]))
    summary = {"file": str(path)}
    if rewards:
        ci = _bootstrap_ci(rewards)
        summary.update(
            {
                "reward_mean": ci["mean"],
                "reward_ci_lower": ci["ci_lower"],
                "reward_ci_upper": ci["ci_upper"],
            }
        )
    if lats:
        s = sorted(lats)
        summary["latency_p95_ms"] = s[int(0.95 * (len(s) - 1))]
        summary["latency_p99_ms"] = s[int(0.99 * (len(s) - 1))]
    if ttfts:
        summary["ttft_avg_ms"] = sum(ttfts) / len(ttfts)
    if json_valids:
        summary["json_valid_rate"] = sum(json_valids) / len(json_valids)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="JSONL files to summarize (train_log.jsonl or eval.jsonl).")
    ap.add_argument("--out", type=str, default=None, help="Optional CSV output path.")
    args = ap.parse_args()

    records = []
    for path_str in args.inputs:
        p = Path(path_str)
        if p.is_dir():
            # pick common filenames
            for cand in ["train_log.jsonl", "eval.jsonl"]:
                if (p / cand).exists():
                    p = p / cand
                    break
        if not p.exists():
            raise SystemExit(f"input file not found: {p}")
        records.append(summarize_file(p))

    df = pd.DataFrame(records)
    df = df[sorted(df.columns)]
    print(df.to_markdown(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
