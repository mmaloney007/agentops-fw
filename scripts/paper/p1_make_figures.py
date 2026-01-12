#!/usr/bin/env python3
"""
Generate Paper 1 figures from W&B episodes artifacts or local episodes.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _load_episodes(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _download_wandb_artifact(artifact: str, target_dir: Path) -> Path:
    import wandb

    api = wandb.Api()
    art = api.artifact(artifact)
    target_dir.mkdir(parents=True, exist_ok=True)
    art.download(root=str(target_dir))
    return target_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="", help="Path to local episodes.jsonl (optional).")
    ap.add_argument("--episodes-dir", default="", help="Directory containing one or more episodes.jsonl files.")
    ap.add_argument("--artifact", default="", help="W&B artifact name (entity/project/name:alias).")
    ap.add_argument("--out-dir", default="papers/p1/figs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes: List[Dict[str, Any]] = []
    if args.episodes_dir:
        for path in Path(args.episodes_dir).rglob("episodes.jsonl"):
            episodes.extend(_load_episodes(path))
    elif args.episodes:
        episodes = _load_episodes(Path(args.episodes))
    elif args.artifact:
        download_dir = _download_wandb_artifact(args.artifact, Path("out/wandb_artifacts"))
        candidates = list(download_dir.rglob("episodes.jsonl"))
        if not candidates:
            raise SystemExit("episodes.jsonl not found in artifact")
        episodes = _load_episodes(candidates[0])
    else:
        raise SystemExit("Provide --episodes, --episodes-dir, or --artifact.")
    if not episodes:
        raise SystemExit("No episodes loaded.")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required for figures: {exc}") from exc

    rows = []
    for ep in episodes:
        row = {
            "model": ep.get("model"),
            "decode_mode": ep.get("decode_mode"),
            "latency_ms": ep.get("latency_ms"),
        }
        for k, v in (ep.get("metrics") or {}).items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)

    # Figure 1: latency histogram
    plt.figure(figsize=(6, 4))
    df["latency_ms"].dropna().plot.hist(bins=30, color="#4c78a8", alpha=0.85)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Latency Histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "latency_hist.png", dpi=160)
    plt.close()

    # Figure 2: p95 latency by decode mode (proxy for QPS vs p95)
    p95_by_mode = df.groupby("decode_mode")["latency_ms"].quantile(0.95).reset_index()
    plt.figure(figsize=(6, 4))
    plt.bar(p95_by_mode["decode_mode"], p95_by_mode["latency_ms"], color="#f58518")
    plt.xlabel("Decode Mode")
    plt.ylabel("p95 Latency (ms)")
    plt.title("p95 Latency by Decode Mode")
    plt.tight_layout()
    plt.savefig(out_dir / "qps_vs_p95.png", dpi=160)
    plt.close()

    # Figure 3: success@SLO by decode mode
    if "success_at_slo" in df.columns:
        success = df.groupby("decode_mode")["success_at_slo"].mean().reset_index()
        plt.figure(figsize=(6, 4))
        plt.bar(success["decode_mode"], success["success_at_slo"], color="#54a24b")
        plt.ylim(0, 1)
        plt.xlabel("Decode Mode")
        plt.ylabel("Success@SLO")
        plt.title("Success@SLO by Decode Mode")
        plt.tight_layout()
        plt.savefig(out_dir / "success_under_budget.png", dpi=160)
        plt.close()

    # Figure 4: simple pareto (p95 vs faithfulness)
    if "hotpot_faithfulness" in df.columns:
        pareto = df.groupby("decode_mode").agg(
            p95_latency=("latency_ms", lambda x: x.quantile(0.95)),
            faithfulness=("hotpot_faithfulness", "mean"),
        )
        plt.figure(figsize=(6, 4))
        plt.scatter(pareto["p95_latency"], pareto["faithfulness"], color="#e45756")
        for mode, row in pareto.iterrows():
            plt.text(row["p95_latency"], row["faithfulness"], str(mode))
        plt.xlabel("p95 Latency (ms)")
        plt.ylabel("Faithfulness")
        plt.title("Pareto: Faithfulness vs p95 Latency")
        plt.tight_layout()
        plt.savefig(out_dir / "pareto.png", dpi=160)
        plt.close()

    print(f"[P1] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
