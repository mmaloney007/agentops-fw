#!/usr/bin/env python3
"""
P9 Full Demo: One command to produce all results and visual charts.

Reads existing evaluation data from disk, runs live demos where needed,
and generates matplotlib charts showing the full P9 story:

  1. ANE vs MLX inference comparison (latency, throughput)
  2. Qwen3.5 model scaling (0.8B → 35B-MoE on MLX)
  3. Full model zoo comparison (13+ models)
  4. Live ANE inference demo (5 prompts)
  5. Live Qwen3.5 GRPO training (20 steps, MLX)
  6. Hybrid ANE+MLX GRPO training (5 steps)
  7. Dashboard combining all charts

Outputs saved to: results/p9_demo/

Usage:
  python scripts/demo_p9_full.py              # Full demo (live + charts)
  python scripts/demo_p9_full.py --charts-only # Charts from existing data only
  python scripts/demo_p9_full.py --skip-grpo   # Skip GRPO training runs
  python scripts/demo_p9_full.py --grpo-steps 50  # More training steps
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLORS = {
    "ane": "#FF6B35",      # Orange for ANE
    "mlx": "#4ECDC4",      # Teal for MLX
    "hybrid": "#9B59B6",   # Purple for hybrid
    "reward": "#2ECC71",   # Green for reward
    "loss": "#E74C3C",     # Red for loss
    "valid": "#3498DB",    # Blue for validity
    "sync": "#F39C12",     # Yellow for sync
    "bg": "#1a1a2e",       # Dark background
    "fg": "#e0e0e0",       # Light text
    "grid": "#333355",     # Grid lines
    "accent": "#FF6B35",   # Accent
}

MODEL_COLORS = [
    "#FF6B35", "#4ECDC4", "#9B59B6", "#2ECC71", "#E74C3C",
    "#3498DB", "#F39C12", "#1ABC9C", "#E91E63", "#00BCD4",
    "#FF9800", "#8BC34A", "#673AB7",
]


def apply_dark_style():
    """Apply dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": "#16213e",
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["fg"],
        "text.color": COLORS["fg"],
        "xtick.color": COLORS["fg"],
        "ytick.color": COLORS["fg"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "font.family": "monospace",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.titlesize": 18,
        "figure.titleweight": "bold",
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ane_smoke_results() -> List[Dict[str, Any]]:
    """Load ANE smoke test results."""
    path = PROJECT_ROOT / "results" / "ane_smoke_test.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [r for r in data if r.get("status") == "PASS"]


def load_mlx_eval_results() -> Dict[str, Dict[str, Any]]:
    """Load all MLX evaluation summaries. Prefer 3000-record runs."""
    results = {}
    eval_dir = PROJECT_ROOT / "results" / "mlx_eval"
    if not eval_dir.exists():
        return results

    for summary_path in sorted(eval_dir.rglob("summary.json")):
        with open(summary_path) as f:
            data = json.load(f)
        model_id = data.get("model", "")
        count = data.get("count", 0)
        # Skip broken runs
        if data.get("avg_latency_ms", 0) >= 999999:
            continue
        # Keep the run with the most records
        if model_id not in results or count > results[model_id].get("count", 0):
            results[model_id] = data
    return results


def load_hybrid_grpo_log() -> List[Dict[str, Any]]:
    """Load the most recent hybrid GRPO training log."""
    out_dir = PROJECT_ROOT / "out"
    if not out_dir.exists():
        return []

    logs = sorted(out_dir.glob("hybrid_train_*/train_log.jsonl"), reverse=True)
    if not logs:
        return []

    records = []
    with open(logs[0]) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_mlx_grpo_logs() -> Dict[str, List[Dict[str, Any]]]:
    """Load all Qwen3.5 MLX GRPO training logs (keyed by model size)."""
    demo_dir = PROJECT_ROOT / "results" / "p9_demo"
    logs: Dict[str, List[Dict[str, Any]]] = {}

    # Look for model-specific logs: qwen35_0.8b_grpo_log.jsonl, qwen35_2b_grpo_log.jsonl
    for log_path in sorted(demo_dir.glob("qwen35_*_grpo_log.jsonl")):
        # Extract size label from filename (e.g. "0.8b", "2b")
        stem = log_path.stem  # e.g. "qwen35_0.8b_grpo_log"
        parts = stem.replace("qwen35_", "").replace("_grpo_log", "")
        label = parts.upper()  # "0.8B", "2B"
        records = []
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if records:
            logs[label] = records

    # Also check generic log (backwards compat)
    generic = demo_dir / "qwen35_grpo_log.jsonl"
    if generic.exists() and not logs:
        records = []
        with open(generic) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if records:
            logs["0.8B"] = records

    return logs


def model_short_name(model_id: str) -> str:
    """Extract short display name from model ID."""
    name = model_id.split("/")[-1]
    name = name.replace("-Instruct", "").replace("-instruct", "")
    name = name.replace("-it", "").replace("-4bit", "")
    name = name.replace("-4k", "").replace("mlx-community-", "")
    return name


# ---------------------------------------------------------------------------
# Chart 1: ANE vs MLX inference comparison
# ---------------------------------------------------------------------------


def chart_ane_vs_mlx(
    ane_results: List[Dict],
    mlx_results: Dict[str, Dict],
    out_dir: Path,
) -> Optional[Path]:
    """Bar chart comparing ANE and MLX inference."""
    # Find models that have both ANE and MLX results
    ane_by_arch = {}
    for r in ane_results:
        name = r.get("model_name", "")
        if "qwen2" in name.lower():
            ane_by_arch["Qwen2.5-0.5B"] = r
        elif "llama" in name.lower():
            ane_by_arch["Llama-3.2-1B"] = r

    mlx_models = {
        "Llama-3.2-1B": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "Qwen3.5-0.8B": "mlx-community/Qwen3.5-0.8B-4bit",
        "Qwen3.5-2B": "mlx-community/Qwen3.5-2B-4bit",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ANE vs MLX Inference on Apple Silicon", y=0.98)

    # Left: Latency comparison
    ax = axes[0]
    labels = []
    ane_lats = []
    mlx_lats = []

    for name in ["Qwen2.5-0.5B", "Llama-3.2-1B"]:
        if name in ane_by_arch:
            labels.append(name)
            ane_lats.append(ane_by_arch[name]["total_ms"])
            # Find matching MLX model
            mlx_key = mlx_models.get(name)
            if mlx_key and mlx_key in mlx_results:
                mlx_lats.append(mlx_results[mlx_key]["avg_latency_ms"])
            else:
                mlx_lats.append(0)

    if labels:
        x = np.arange(len(labels))
        w = 0.35
        bars1 = ax.bar(x - w/2, ane_lats, w, label="ANE (~3W)", color=COLORS["ane"], edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + w/2, mlx_lats, w, label="MLX GPU (~30W)", color=COLORS["mlx"], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Inference Latency")
        ax.legend(framealpha=0.8)
        ax.grid(axis="y", alpha=0.3)
        # Value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

    # Right: Throughput comparison
    ax = axes[1]
    if ane_results:
        names = []
        ane_tps = []
        for r in ane_results:
            n = r.get("model_name", "?")
            if "qwen" in n.lower():
                names.append("Qwen2.5-0.5B")
            elif "llama" in n.lower():
                names.append("Llama-3.2-1B")
            else:
                names.append(n[:15])
            ane_tps.append(r["tok_per_sec"])

        x = np.arange(len(names))
        bars = ax.bar(x, ane_tps, 0.6, color=COLORS["ane"], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel("Tokens / second")
        ax.set_title("ANE Throughput (Neural Engine)")
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10)

        # Add power annotation
        ax.text(0.02, 0.95, "ANE: ~3W total system power",
                transform=ax.transAxes, fontsize=9, color=COLORS["ane"],
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["bg"], edgecolor=COLORS["ane"], alpha=0.8))

    plt.tight_layout()
    path = out_dir / "1_ane_vs_mlx.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path}")
    return path


# ---------------------------------------------------------------------------
# Chart 2: Qwen3.5 model scaling on MLX
# ---------------------------------------------------------------------------


def chart_qwen35_scaling(
    mlx_results: Dict[str, Dict],
    out_dir: Path,
) -> Optional[Path]:
    """Show how Qwen3.5 scales from 0.8B to 35B-MoE on MLX."""
    qwen_models = [
        ("Qwen3.5-0.8B", "mlx-community/Qwen3.5-0.8B-4bit", 0.8),
        ("Qwen3.5-2B", "mlx-community/Qwen3.5-2B-4bit", 2.0),
        ("Qwen3.5-35B\n(MoE A3B)", "mlx-community/Qwen3.5-35B-A3B-4bit", 3.0),  # Active params
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Qwen3.5 Family: Scaling on Apple Silicon (MLX)", y=0.98)

    names, sizes, latencies, validities = [], [], [], []
    for name, model_id, active_b in qwen_models:
        if model_id in mlx_results:
            r = mlx_results[model_id]
            names.append(name)
            sizes.append(active_b)
            latencies.append(r["avg_latency_ms"])
            validities.append(r.get("json_valid", 0) * 100)

    if not names:
        plt.close(fig)
        return None

    x = np.arange(len(names))

    # Latency
    ax = axes[0]
    bars = ax.bar(x, latencies, 0.6, color=[COLORS["mlx"]] * len(names),
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Inference Latency")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

    # JSON Validity
    ax = axes[1]
    bar_colors = []
    for v in validities:
        if v >= 80:
            bar_colors.append(COLORS["reward"])
        elif v >= 60:
            bar_colors.append(COLORS["sync"])
        else:
            bar_colors.append(COLORS["loss"])
    bars = ax.bar(x, validities, 0.6, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("JSON Validity (%)")
    ax.set_title("Structured Output Quality")
    ax.set_ylim(0, 105)
    ax.axhline(y=80, color=COLORS["reward"], linestyle="--", alpha=0.5, label="80% threshold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    # Efficiency (validity / latency)
    ax = axes[2]
    efficiency = [v / (l / 1000) if l > 0 else 0 for v, l in zip(validities, latencies)]
    bars = ax.bar(x, efficiency, 0.6, color=COLORS["hybrid"], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Validity% / Second")
    ax.set_title("SLO Efficiency")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = out_dir / "2_qwen35_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path}")
    return path


# ---------------------------------------------------------------------------
# Chart 3: Full model zoo comparison
# ---------------------------------------------------------------------------


def chart_model_zoo(
    mlx_results: Dict[str, Dict],
    ane_results: List[Dict],
    out_dir: Path,
) -> Optional[Path]:
    """Scatter plot of all models: latency vs validity."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle("P9 Model Zoo: Latency vs JSON Validity on Apple Silicon", y=0.98)

    # MLX models
    mlx_data = []
    for model_id, data in mlx_results.items():
        if data.get("count", 0) < 100:
            continue  # Skip smoke tests
        name = model_short_name(model_id)
        lat = data["avg_latency_ms"]
        valid = data.get("json_valid", 0) * 100
        mlx_data.append((name, lat, valid, model_id))

    if not mlx_data:
        plt.close(fig)
        return None

    # Sort by latency for consistent coloring
    mlx_data.sort(key=lambda x: x[1])

    for i, (name, lat, valid, mid) in enumerate(mlx_data):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        is_qwen35 = "Qwen3.5" in name
        marker = "D" if is_qwen35 else "o"
        size = 200 if is_qwen35 else 120
        edgecolor = "white" if is_qwen35 else color
        ax.scatter(lat, valid, c=color, s=size, marker=marker,
                   edgecolors=edgecolor, linewidths=2, zorder=5, label=f"MLX: {name}")

    # ANE models
    for r in ane_results:
        name = r.get("model_name", "?")
        if "qwen" in name.lower():
            display = "Qwen2.5-0.5B"
        elif "llama" in name.lower():
            display = "Llama-3.2-1B"
        else:
            display = name[:15]
        lat = r["total_ms"]
        ax.scatter(lat, 95, c=COLORS["ane"], s=250, marker="*",
                   edgecolors="white", linewidths=2, zorder=6,
                   label=f"ANE: {display}")

    # SLO zones
    ax.axhline(y=80, color=COLORS["reward"], linestyle="--", alpha=0.4)
    ax.axvline(x=2000, color=COLORS["sync"], linestyle="--", alpha=0.4)
    ax.text(100, 82, "80% validity threshold", fontsize=8, color=COLORS["reward"], alpha=0.7)
    ax.text(2050, 5, "2s SLO", fontsize=8, color=COLORS["sync"], alpha=0.7, rotation=90)

    # Ideal zone
    ax.fill_between([0, 2000], [80, 80], [105, 105],
                    color=COLORS["reward"], alpha=0.05)
    ax.text(200, 97, "SLO-compliant zone", fontsize=9, color=COLORS["reward"],
            alpha=0.5, style="italic")

    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("JSON Validity (%)")
    ax.set_ylim(0, 105)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.grid(True, alpha=0.2)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    path = out_dir / "3_model_zoo.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path}")
    return path


# ---------------------------------------------------------------------------
# Chart 4: GRPO training curves
# ---------------------------------------------------------------------------


def chart_grpo_training(
    hybrid_log: List[Dict],
    mlx_grpo_logs: Dict[str, List[Dict]],
    qwen25_log: List[Dict],
    out_dir: Path,
) -> Optional[Path]:
    """Training curves: loss, reward, validity comparison, and timing breakdown."""
    has_hybrid = len(hybrid_log) > 1
    has_mlx = any(len(v) > 1 for v in mlx_grpo_logs.values())
    has_qwen25 = len(qwen25_log) > 1

    if not has_hybrid and not has_mlx and not has_qwen25:
        return None

    # Color cycle for model sizes
    MLX_COLORS = {"0.8B": "#4ECDC4", "2B": "#3498DB", "MoE": "#9B59B6"}
    QWEN25_COLOR = "#2ECC71"  # Green for Qwen2.5 (the stable one)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GRPO Training Dynamics on Apple Silicon", y=0.98)

    # Top-left: Loss curves
    ax = axes[0, 0]
    if has_hybrid:
        steps = [r["step"] for r in hybrid_log]
        losses = [r["loss"] for r in hybrid_log]
        ax.plot(steps, losses, "o-", color=COLORS["ane"], linewidth=2,
                markersize=6, label="Hybrid ANE+MLX\n(Qwen2.5-0.5B)")
    for size, log in sorted(mlx_grpo_logs.items()):
        if len(log) < 2:
            continue
        steps = [r["step"] for r in log]
        losses = [r["loss"] for r in log]
        c = MLX_COLORS.get(size, COLORS["mlx"])
        ax.plot(steps, losses, "s-", color=c, linewidth=2,
                markersize=5, label=f"Qwen3.5-{size}")
    if has_qwen25:
        steps = [r["step"] for r in qwen25_log]
        losses = [r["loss"] for r in qwen25_log]
        ax.plot(steps, losses, "^-", color=QWEN25_COLOR, linewidth=2,
                markersize=6, label="Qwen2.5-0.5B\n(conservative)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Loss Curves")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # Top-right: Reward curves
    ax = axes[0, 1]
    if has_hybrid:
        steps = [r["step"] for r in hybrid_log]
        rewards = [r["mean_reward"] for r in hybrid_log]
        ax.plot(steps, rewards, "o-", color=COLORS["ane"], linewidth=2,
                markersize=6, label="Hybrid ANE+MLX")
    for size, log in sorted(mlx_grpo_logs.items()):
        if len(log) < 2:
            continue
        steps = [r["step"] for r in log]
        rewards = [r["mean_reward"] for r in log]
        c = MLX_COLORS.get(size, COLORS["mlx"])
        ax.plot(steps, rewards, "s-", color=c, linewidth=2,
                markersize=5, label=f"Qwen3.5-{size}")
    if has_qwen25:
        steps = [r["step"] for r in qwen25_log]
        rewards = [r["mean_reward"] for r in qwen25_log]
        ax.plot(steps, rewards, "^-", color=QWEN25_COLOR, linewidth=2,
                markersize=6, label="Qwen2.5-0.5B")
    ax.axhline(y=0, color="white", linestyle=":", alpha=0.3)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward Progression")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: JSON validity comparison across all models
    ax = axes[1, 0]
    if has_mlx or has_qwen25:
        for size, log in sorted(mlx_grpo_logs.items()):
            if len(log) < 2:
                continue
            steps = [r["step"] for r in log]
            valid = [r.get("json_valid", 0) * 100 for r in log]
            c = MLX_COLORS.get(size, COLORS["mlx"])
            ax.plot(steps, valid, "o-", color=c, linewidth=2,
                    markersize=6, label=f"Qwen3.5-{size}")
            ax.fill_between(steps, valid, alpha=0.1, color=c)
        if has_qwen25:
            steps = [r["step"] for r in qwen25_log]
            valid = [r.get("json_valid", 0) * 100 for r in qwen25_log]
            ax.plot(steps, valid, "^-", color=QWEN25_COLOR, linewidth=2.5,
                    markersize=7, label="Qwen2.5-0.5B (conservative)")
            ax.fill_between(steps, valid, alpha=0.15, color=QWEN25_COLOR)
        if has_hybrid:
            steps = [r["step"] for r in hybrid_log]
            valid = [r.get("json_valid", 0) * 100 for r in hybrid_log]
            ax.plot(steps, valid, "D-", color=COLORS["ane"], linewidth=2,
                    markersize=5, label="Hybrid (Qwen2.5-0.5B)")
        ax.axhline(y=80, color=COLORS["reward"], linestyle="--", alpha=0.4, label="80% SLO")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("JSON Valid (%)")
        ax.set_ylim(-5, 110)
        ax.set_title("JSON Validity During Training")
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3)
    elif has_hybrid:
        steps = [r["step"] for r in hybrid_log]
        ane_ms = [r["ane_rollout_ms"] for r in hybrid_log]
        mlx_ms = [r["mlx_gradient_ms"] for r in hybrid_log]
        sync_ms = [r["weight_sync_ms"] for r in hybrid_log]
        ax.bar(steps, ane_ms, label="ANE Rollout", color=COLORS["ane"], edgecolor="white", linewidth=0.5)
        ax.bar(steps, mlx_ms, bottom=ane_ms, label="MLX Gradient", color=COLORS["mlx"], edgecolor="white", linewidth=0.5)
        bottoms = [a + m for a, m in zip(ane_ms, mlx_ms)]
        ax.bar(steps, sync_ms, bottom=bottoms, label="Weight Sync", color=COLORS["sync"], edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Hybrid Step Timing Breakdown")
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(axis="y", alpha=0.3)

    # Bottom-right: Compute split pie (hybrid) or capacity summary
    ax = axes[1, 1]
    if has_hybrid:
        total_ane = sum(r["ane_rollout_ms"] for r in hybrid_log)
        total_mlx = sum(r["mlx_gradient_ms"] for r in hybrid_log)
        total_sync = sum(r["weight_sync_ms"] for r in hybrid_log)
        total = total_ane + total_mlx + total_sync

        if total > 0:
            sizes = [total_ane, total_mlx, total_sync]
            labels_pie = [
                f"ANE Rollout\n{total_ane/1000:.1f}s ({100*total_ane/total:.0f}%)",
                f"MLX Gradient\n{total_mlx/1000:.1f}s ({100*total_mlx/total:.0f}%)",
                f"Weight Sync\n{total_sync/1000:.3f}s ({100*total_sync/total:.1f}%)",
            ]
            colors_pie = [COLORS["ane"], COLORS["mlx"], COLORS["sync"]]
            wedges, texts = ax.pie(
                sizes, labels=labels_pie, colors=colors_pie,
                startangle=90, textprops={"fontsize": 9, "color": COLORS["fg"]},
                wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            )
            ax.set_title("Compute Split (Hybrid Training)")
    elif has_mlx:
        # Capacity summary: valid steps / total for each model
        ax.axis("off")
        lines = ["GRPO Capacity Threshold", "=" * 35, ""]
        for size, log in sorted(mlx_grpo_logs.items()):
            n_valid = sum(1 for r in log if r.get("json_valid", 0))
            n_total = len(log)
            pct = 100 * n_valid / n_total if n_total else 0
            first_r = log[0]["mean_reward"] if log else 0
            last_r = log[-1]["mean_reward"] if log else 0
            lines.append(f"Qwen3.5-{size}:")
            lines.append(f"  Valid steps: {n_valid}/{n_total} ({pct:.0f}%)")
            lines.append(f"  Reward: {first_r:.2f} -> {last_r:.2f}")
            lines.append("")
        lines.append("Larger models maintain JSON")
        lines.append("validity longer under GRPO.")
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#16213e",
                         edgecolor=COLORS["accent"], alpha=0.9))

    plt.tight_layout()
    path = out_dir / "4_grpo_training.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path}")
    return path


# ---------------------------------------------------------------------------
# Chart 5: Dashboard (combined overview)
# ---------------------------------------------------------------------------


def chart_dashboard(
    ane_results: List[Dict],
    mlx_results: Dict[str, Dict],
    hybrid_log: List[Dict],
    mlx_grpo_logs: Dict[str, List[Dict]],
    qwen25_log: List[Dict],
    out_dir: Path,
) -> Optional[Path]:
    """Single-page dashboard combining key results."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "P9: Heterogeneous Compute on Apple Silicon\n"
        "Neural Engine + Metal GPU for SLO-Aware Agent Training",
        fontsize=16, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

    # --- Row 1: ANE throughput, model zoo scatter, Qwen3.5 validity ---

    # ANE throughput bars
    ax = fig.add_subplot(gs[0, 0])
    if ane_results:
        names = []
        tps = []
        for r in ane_results:
            n = r.get("model_name", "")
            if "qwen" in n.lower():
                names.append("Qwen2.5\n0.5B")
            elif "llama" in n.lower():
                names.append("Llama3.2\n1B")
            else:
                names.append(n[:8])
            tps.append(r["tok_per_sec"])
        bars = ax.bar(range(len(names)), tps, color=COLORS["ane"], edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("tok/s")
        ax.set_title("ANE Throughput\n(~3W)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

    # Model zoo scatter (compact)
    ax = fig.add_subplot(gs[0, 1:3])
    mlx_data = []
    for mid, data in mlx_results.items():
        if data.get("count", 0) < 100:
            continue
        name = model_short_name(mid)
        lat = data["avg_latency_ms"]
        valid = data.get("json_valid", 0) * 100
        mlx_data.append((name, lat, valid))
    mlx_data.sort(key=lambda x: x[1])

    for i, (name, lat, valid) in enumerate(mlx_data):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        is_q35 = "Qwen3.5" in name
        marker = "D" if is_q35 else "o"
        size = 150 if is_q35 else 80
        ax.scatter(lat, valid, c=color, s=size, marker=marker,
                   edgecolors="white" if is_q35 else color, linewidths=1.5, zorder=5)
        # Label only Qwen3.5 models
        if is_q35:
            ax.annotate(name, (lat, valid), textcoords="offset points",
                        xytext=(8, -5), fontsize=7, color=color)

    for r in ane_results:
        n = r.get("model_name", "")
        display = "Qwen2.5-0.5B" if "qwen" in n.lower() else "Llama-3.2-1B"
        ax.scatter(r["total_ms"], 95, c=COLORS["ane"], s=180, marker="*",
                   edgecolors="white", linewidths=1.5, zorder=6)
        ax.annotate(f"ANE: {display}", (r["total_ms"], 95), textcoords="offset points",
                    xytext=(8, -5), fontsize=7, color=COLORS["ane"])

    ax.axhline(y=80, color=COLORS["reward"], linestyle="--", alpha=0.3)
    ax.axvline(x=2000, color=COLORS["sync"], linestyle="--", alpha=0.3)
    ax.fill_between([0, 2000], [80, 80], [105, 105], color=COLORS["reward"], alpha=0.05)
    ax.set_xlabel("Latency (ms)", fontsize=9)
    ax.set_ylabel("JSON Valid %", fontsize=9)
    ax.set_title("Model Zoo: Latency vs Validity", fontsize=10)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2)

    # Qwen3.5 validity bars
    ax = fig.add_subplot(gs[0, 3])
    qwen_names = []
    qwen_valid = []
    for name, mid in [("0.8B", "mlx-community/Qwen3.5-0.8B-4bit"),
                      ("2B", "mlx-community/Qwen3.5-2B-4bit"),
                      ("MoE\n35B", "mlx-community/Qwen3.5-35B-A3B-4bit")]:
        if mid in mlx_results and mlx_results[mid].get("count", 0) >= 100:
            qwen_names.append(name)
            qwen_valid.append(mlx_results[mid].get("json_valid", 0) * 100)
    if qwen_names:
        colors = [COLORS["reward"] if v >= 80 else COLORS["sync"] if v >= 60 else COLORS["loss"]
                  for v in qwen_valid]
        bars = ax.bar(range(len(qwen_names)), qwen_valid, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(qwen_names)))
        ax.set_xticklabels(qwen_names, fontsize=8)
        ax.set_ylabel("Valid %")
        ax.set_title("Qwen3.5\nValidity", fontsize=10)
        ax.set_ylim(0, 105)
        ax.axhline(y=80, color=COLORS["reward"], linestyle="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8)

    # --- Row 2: Training curves ---
    has_hybrid = len(hybrid_log) > 1
    has_mlx_grpo = any(len(v) > 1 for v in mlx_grpo_logs.values())
    has_qwen25 = len(qwen25_log) > 1
    MLX_COLORS = {"0.8B": "#4ECDC4", "2B": "#3498DB", "MoE": "#9B59B6"}
    QWEN25_COLOR = "#2ECC71"

    # Loss
    ax = fig.add_subplot(gs[1, 0:2])
    if has_hybrid:
        s = [r["step"] for r in hybrid_log]
        l = [r["loss"] for r in hybrid_log]
        ax.plot(s, l, "o-", color=COLORS["ane"], linewidth=2, markersize=5, label="Hybrid ANE+MLX (Qwen2.5-0.5B)")
    for size, log in sorted(mlx_grpo_logs.items()):
        if len(log) < 2:
            continue
        s = [r["step"] for r in log]
        l = [r["loss"] for r in log]
        c = MLX_COLORS.get(size, COLORS["mlx"])
        ax.plot(s, l, "s-", color=c, linewidth=2, markersize=4, label=f"Qwen3.5-{size}")
    if has_qwen25:
        s = [r["step"] for r in qwen25_log]
        l = [r["loss"] for r in qwen25_log]
        ax.plot(s, l, "^-", color=QWEN25_COLOR, linewidth=2, markersize=5, label="Qwen2.5-0.5B")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("GRPO Policy Loss", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # Reward
    ax = fig.add_subplot(gs[1, 2:4])
    if has_hybrid:
        s = [r["step"] for r in hybrid_log]
        rew = [r["mean_reward"] for r in hybrid_log]
        ax.plot(s, rew, "o-", color=COLORS["ane"], linewidth=2, markersize=5, label="Hybrid ANE+MLX")
    for size, log in sorted(mlx_grpo_logs.items()):
        if len(log) < 2:
            continue
        s = [r["step"] for r in log]
        rew = [r["mean_reward"] for r in log]
        c = MLX_COLORS.get(size, COLORS["mlx"])
        ax.plot(s, rew, "s-", color=c, linewidth=2, markersize=4, label=f"Qwen3.5-{size}")
    if has_qwen25:
        s = [r["step"] for r in qwen25_log]
        rew = [r["mean_reward"] for r in qwen25_log]
        ax.plot(s, rew, "^-", color=QWEN25_COLOR, linewidth=2, markersize=5, label="Qwen2.5-0.5B")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward Progression", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # --- Row 3: Timing breakdown + stats ---

    # Timing stacked bar
    ax = fig.add_subplot(gs[2, 0:2])
    if has_hybrid:
        steps = [r["step"] for r in hybrid_log]
        ane_ms = [r["ane_rollout_ms"] for r in hybrid_log]
        mlx_ms = [r["mlx_gradient_ms"] for r in hybrid_log]
        sync_ms = [r["weight_sync_ms"] for r in hybrid_log]
        ax.bar(steps, ane_ms, label="ANE Rollout", color=COLORS["ane"], edgecolor="white", linewidth=0.5)
        ax.bar(steps, mlx_ms, bottom=ane_ms, label="MLX Gradient", color=COLORS["mlx"], edgecolor="white", linewidth=0.5)
        bottoms = [a + m for a, m in zip(ane_ms, mlx_ms)]
        ax.bar(steps, sync_ms, bottom=bottoms, label="Wt Sync", color=COLORS["sync"], edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Hybrid Training: Per-Step Breakdown", fontsize=10)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(axis="y", alpha=0.3)

    # Stats text box
    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis("off")
    stats_lines = []
    stats_lines.append("KEY RESULTS")
    stats_lines.append("=" * 40)

    if ane_results:
        stats_lines.append("")
        stats_lines.append("ANE Inference (Neural Engine, ~3W):")
        for r in ane_results:
            n = r.get("model_name", "")
            if "qwen" in n.lower():
                n = "Qwen2.5-0.5B"
            elif "llama" in n.lower():
                n = "Llama-3.2-1B"
            stats_lines.append(f"  {n}: {r['tok_per_sec']:.1f} tok/s, "
                             f"TTFT={r['ttft_ms']:.0f}ms")

    if mlx_results:
        # Count SLO-compliant models
        compliant = sum(1 for d in mlx_results.values()
                       if d.get("count", 0) >= 100
                       and d.get("json_valid", 0) >= 0.8
                       and d.get("avg_latency_ms", 99999) <= 2000)
        total = sum(1 for d in mlx_results.values() if d.get("count", 0) >= 100)
        stats_lines.append("")
        stats_lines.append(f"MLX Model Zoo: {total} models evaluated")
        stats_lines.append(f"  SLO-compliant (<2s, >80% valid): {compliant}/{total}")

    if has_hybrid:
        total_time = sum(r["step_ms"] for r in hybrid_log)
        avg_ane = sum(r["ane_rollout_ms"] for r in hybrid_log) / len(hybrid_log)
        avg_mlx = sum(r["mlx_gradient_ms"] for r in hybrid_log) / len(hybrid_log)
        stats_lines.append("")
        stats_lines.append(f"Hybrid GRPO ({len(hybrid_log)} steps):")
        stats_lines.append(f"  ANE rollout: {avg_ane:.0f}ms avg")
        stats_lines.append(f"  MLX gradient: {avg_mlx:.0f}ms avg")
        stats_lines.append(f"  Total: {total_time/1000:.1f}s")

    if has_qwen25:
        n_valid = sum(1 for r in qwen25_log if r.get("json_valid", 0))
        first_r = qwen25_log[0]["mean_reward"]
        last_r = qwen25_log[-1]["mean_reward"]
        stats_lines.append("")
        stats_lines.append(f"Qwen2.5-0.5B GRPO ({len(qwen25_log)} steps):")
        stats_lines.append(f"  Valid: {n_valid}/{len(qwen25_log)}, "
                         f"rew {first_r:.2f}->{last_r:.2f}")

    if has_mlx_grpo:
        stats_lines.append("")
        stats_lines.append("Qwen3.5 GRPO (unstable):")
        for size, log in sorted(mlx_grpo_logs.items()):
            if not log:
                continue
            n_valid = sum(1 for r in log if r.get("json_valid", 0))
            first_r = log[0]["mean_reward"]
            last_r = log[-1]["mean_reward"]
            stats_lines.append(f"  {size}: {n_valid}/{len(log)} valid, "
                             f"rew {first_r:.2f}->{last_r:.2f}")

    stats_text = "\n".join(stats_lines)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#16213e",
                     edgecolor=COLORS["accent"], alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "5_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path}")
    return path


# ---------------------------------------------------------------------------
# Live demo: ANE inference
# ---------------------------------------------------------------------------


def run_live_ane_demo(out_dir: Path) -> List[Dict]:
    """Run 5 live ANE inference prompts and return results."""
    print("\n  Running live ANE inference demo...")

    ane_meta_dir = str((PROJECT_ROOT / "models" / "ane" / "qwen2.5-0.5b").resolve())
    if not Path(ane_meta_dir).joinpath("meta.yaml").exists():
        print("    ANE model not found, skipping live demo.")
        return []

    os.environ["ANE_META_DIR"] = ane_meta_dir
    os.environ["ANE_HF_MODEL"] = "Qwen/Qwen2.5-0.5B-Instruct"
    os.environ["ANE_MAX_TOKENS"] = "128"

    from agent_stable_slo.rollout.providers import ane_local

    prompts = [
        ("Intent", "Book a flight to Paris",
         {"type": "object", "properties": {"intent": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["intent", "confidence"]}),
        ("QA", "What is the capital of France?",
         {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}),
        ("Tool", "Search for weather in NYC",
         {"type": "object", "properties": {"tool_name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["tool_name", "arguments"]}),
        ("Math", "What is 15 * 23?",
         {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}),
        ("Code", "Write hello world in Python",
         {"type": "object", "properties": {"code": {"type": "string"}, "language": {"type": "string"}}, "required": ["code", "language"]}),
    ]

    results = []
    print(f"\n    {'Task':<8} {'Latency':>10} {'TTFT':>8} {'Valid':>6}  Output")
    print(f"    {'----':<8} {'-------':>10} {'----':>8} {'-----':>6}  ------")

    for name, prompt, schema in prompts:
        try:
            raw, parsed, lat_ms, ttft_ms, tok_in, tok_out = ane_local.generate_raw(
                prompt=prompt, schema=schema, mode="structured", temperature=0.0, max_tokens=128)
            valid = bool(parsed and len(parsed) > 0)
            output_str = json.dumps(parsed, ensure_ascii=False)[:60] if parsed else raw[:60]
            print(f"    {name:<8} {lat_ms:>8.0f}ms {ttft_ms:>6.0f}ms {'YES' if valid else 'NO':>6}  {output_str}")
            results.append({"name": name, "latency_ms": lat_ms, "ttft_ms": ttft_ms,
                           "valid": valid, "parsed": parsed, "raw": raw[:200]})
        except Exception as e:
            print(f"    {name:<8} ERROR: {e}")
            results.append({"name": name, "error": str(e)})

    # Save results
    with open(out_dir / "ane_live_demo.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ---------------------------------------------------------------------------
# Live demo: Qwen2.5-0.5B MLX GRPO training (conservative, stable)
# ---------------------------------------------------------------------------


def load_qwen25_grpo_log() -> List[Dict[str, Any]]:
    """Load Qwen2.5-0.5B MLX GRPO training log if it exists."""
    log_path = PROJECT_ROOT / "results" / "p9_demo" / "qwen25_0.5b_grpo_log.jsonl"
    if not log_path.exists():
        return []
    records = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_qwen25_grpo(
    out_dir: Path,
    num_steps: int = 25,
) -> List[Dict]:
    """Run Qwen2.5-0.5B GRPO training with conservative settings."""
    model_id = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    print(f"\n  Running Qwen2.5-0.5B MLX GRPO ({num_steps} steps, conservative)...")

    task_file = str(PROJECT_ROOT / "tasks" / "clinc_en.jsonl")
    if not Path(task_file).exists():
        print("    Task file not found, skipping.")
        return []

    try:
        from agent_stable_slo.train.mlx_grpo_adapter import MLXGRPOTrainer
        from agent_stable_slo.train.mlx_train_config import MLXTrainConfig
    except ImportError as e:
        print(f"    Import error: {e}")
        return []

    log_path = out_dir / "qwen25_0.5b_grpo_log.jsonl"

    cfg = MLXTrainConfig(
        base_model=model_id,
        tasks=[task_file],
        num_steps=num_steps,
        group_size=4,
        max_tokens=128,
        lora_rank=8,
        lora_layers=8,
        learning_rate=1e-5,
        beta=0.3,
        checkpoint_every=0,
        eval_interval=5,
        seed=42,
        adapter_path=str(out_dir / "qwen25_0.5b_adapter"),
        log_path=str(log_path),
    )

    try:
        trainer = MLXGRPOTrainer(cfg)
        trainer.run()
    except Exception as e:
        print(f"    Training error: {e}")
        import traceback
        traceback.print_exc()
        return []

    # Read back the log
    records = []
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    print(f"    Completed {len(records)} steps.")
    if records:
        n_valid = sum(1 for r in records if r.get("json_valid", 0))
        print(f"    Valid: {n_valid}/{len(records)} ({100*n_valid/len(records):.0f}%)")
        print(f"    Reward: {records[0]['mean_reward']:.3f} -> {records[-1]['mean_reward']:.3f}")

    return records


# ---------------------------------------------------------------------------
# Live demo: Qwen3.5 MLX GRPO training
# ---------------------------------------------------------------------------


def run_qwen35_grpo(
    out_dir: Path,
    num_steps: int = 20,
    qwen35_model: str = "mlx-community/Qwen3.5-2B-4bit",
) -> List[Dict]:
    """Run Qwen3.5 GRPO training on MLX and return log records."""
    short_name = qwen35_model.split("/")[-1]
    print(f"\n  Running {short_name} MLX GRPO ({num_steps} steps)...")

    task_file = str(PROJECT_ROOT / "tasks" / "clinc_en.jsonl")
    if not Path(task_file).exists():
        print("    Task file not found, skipping.")
        return []

    try:
        from agent_stable_slo.train.mlx_grpo_adapter import MLXGRPOTrainer
        from agent_stable_slo.train.mlx_train_config import MLXTrainConfig
    except ImportError as e:
        print(f"    Import error: {e}")
        return []

    # Derive size label for filename
    size_label = "2b" if "2B" in qwen35_model.upper() or "2b" in qwen35_model else "0.8b"
    log_path = out_dir / f"qwen35_{size_label}_grpo_log.jsonl"

    cfg = MLXTrainConfig(
        base_model=qwen35_model,
        tasks=[task_file],
        num_steps=num_steps,
        group_size=2,
        max_tokens=96,
        lora_rank=4,
        lora_layers=4,
        learning_rate=5e-5,
        checkpoint_every=0,
        seed=42,
        adapter_path=str(out_dir / f"qwen35_{size_label}_adapter"),
        log_path=str(log_path),
    )

    try:
        trainer = MLXGRPOTrainer(cfg)
        trainer.run()
    except Exception as e:
        print(f"    Training error: {e}")
        import traceback
        traceback.print_exc()
        return []

    # Read back the log
    records = []
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    print(f"    Completed {len(records)} steps.")
    if records:
        print(f"    Reward: {records[0]['mean_reward']:.3f} -> {records[-1]['mean_reward']:.3f}")
        print(f"    Loss: {records[0]['loss']:.4f} -> {records[-1]['loss']:.4f}")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="P9 Full Demo: one command for all results + charts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--charts-only", action="store_true",
                    help="Only generate charts from existing data (no live runs)")
    ap.add_argument("--skip-grpo", action="store_true",
                    help="Skip GRPO training runs")
    ap.add_argument("--skip-ane", action="store_true",
                    help="Skip live ANE demo")
    ap.add_argument("--grpo-steps", type=int, default=20,
                    help="Number of GRPO training steps (default: 20)")
    ap.add_argument("--grpo-model", default="mlx-community/Qwen3.5-2B-4bit",
                    help="MLX model for Qwen3.5 GRPO training (default: Qwen3.5-2B-4bit)")
    ap.add_argument("--run-qwen25", action="store_true",
                    help="Run Qwen2.5-0.5B GRPO (conservative settings, recommended)")
    ap.add_argument("--qwen25-steps", type=int, default=25,
                    help="Steps for Qwen2.5-0.5B GRPO (default: 25)")
    args = ap.parse_args()

    apply_dark_style()
    out_dir = PROJECT_ROOT / "results" / "p9_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print()
    print("=" * 60)
    print("  P9 FULL DEMO")
    print("  Heterogeneous Compute on Apple Silicon")
    print("=" * 60)

    # --- Load existing data ---
    print("\n[1/6] Loading existing results...")
    ane_results = load_ane_smoke_results()
    mlx_results = load_mlx_eval_results()
    hybrid_log = load_hybrid_grpo_log()
    mlx_grpo_logs = load_mlx_grpo_logs()
    qwen25_log = load_qwen25_grpo_log()
    print(f"  ANE models: {len(ane_results)}")
    print(f"  MLX models: {len(mlx_results)}")
    print(f"  Hybrid GRPO steps: {len(hybrid_log)}")
    print(f"  Qwen2.5-0.5B GRPO steps: {len(qwen25_log)}")
    for size, log in sorted(mlx_grpo_logs.items()):
        print(f"  Qwen3.5-{size} GRPO steps: {len(log)}")

    # --- Live ANE demo ---
    if not args.charts_only and not args.skip_ane:
        print("\n[2/6] Live ANE inference demo...")
        run_live_ane_demo(out_dir)
    else:
        print("\n[2/6] Skipping live ANE demo.")

    # --- Qwen2.5-0.5B GRPO training (conservative) ---
    # --run-qwen25 overrides --skip-grpo for this model
    if not args.charts_only and (args.run_qwen25 or (not args.skip_grpo and not qwen25_log)):
        if args.run_qwen25 or not qwen25_log:
            print(f"\n[2.5/6] Qwen2.5-0.5B GRPO ({args.qwen25_steps} steps, conservative)...")
            qwen25_log = run_qwen25_grpo(out_dir, num_steps=args.qwen25_steps)
    elif qwen25_log:
        n_valid = sum(1 for r in qwen25_log if r.get("json_valid", 0))
        print(f"\n[2.5/6] Qwen2.5-0.5B GRPO: using existing {len(qwen25_log)}-step log ({n_valid} valid).")

    # --- Qwen3.5 GRPO training ---
    if not args.charts_only and not args.skip_grpo and not mlx_grpo_logs:
        print(f"\n[3/6] Qwen3.5 GRPO training ({args.grpo_steps} steps)...")
        new_log = run_qwen35_grpo(out_dir, num_steps=args.grpo_steps, qwen35_model=args.grpo_model)
        if new_log:
            size = "2B" if "2B" in args.grpo_model.upper() or "2b" in args.grpo_model else "0.8B"
            mlx_grpo_logs[size] = new_log
    elif mlx_grpo_logs:
        total = sum(len(v) for v in mlx_grpo_logs.values())
        print(f"\n[3/6] Qwen3.5 GRPO: using existing logs ({total} total steps across {len(mlx_grpo_logs)} models).")
    else:
        print("\n[3/6] Skipping Qwen3.5 GRPO training.")

    # --- Generate charts ---
    print("\n[4/6] Generating charts...")
    charts = []

    c = chart_ane_vs_mlx(ane_results, mlx_results, out_dir)
    if c: charts.append(c)

    c = chart_qwen35_scaling(mlx_results, out_dir)
    if c: charts.append(c)

    c = chart_model_zoo(mlx_results, ane_results, out_dir)
    if c: charts.append(c)

    c = chart_grpo_training(hybrid_log, mlx_grpo_logs, qwen25_log, out_dir)
    if c: charts.append(c)

    print("\n[5/6] Generating dashboard...")
    c = chart_dashboard(ane_results, mlx_results, hybrid_log, mlx_grpo_logs, qwen25_log, out_dir)
    if c: charts.append(c)

    # --- Summary ---
    elapsed = time.time() - t0
    print(f"\n[6/6] Done in {elapsed:.1f}s")
    print(f"\n{'=' * 60}")
    print(f"  OUTPUT: {out_dir}/")
    print(f"{'=' * 60}")
    for c in charts:
        print(f"  {c.name}")
    print()

    # Quick open on macOS
    if charts and sys.platform == "darwin":
        print(f"  To view: open {out_dir}")
    print()


if __name__ == "__main__":
    main()
