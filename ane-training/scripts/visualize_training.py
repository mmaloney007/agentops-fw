#!/usr/bin/env python3
"""
Visualization script for GRPO training logs.

Reads JSONL log files produced by grpo_public/grpo_private and generates:
1. Training Progress: reward + JSON valid % over steps
2. Step Timing Breakdown: stacked bar chart
3. Power Profile: CPU/GPU/ANE watts per step
4. Component Activity: horizontal bar showing compute distribution
5. Comparison: side-by-side public vs private

Usage:
    python visualize_training.py results/exp_hard_public/grpo_log.jsonl --output figs/
    python visualize_training.py log1.jsonl log2.jsonl --output figs/ --compare
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def load_jsonl(path):
    """Load a JSONL log file into a list of dicts."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def extract_series(entries):
    """Extract time series from log entries."""
    steps = [e['step'] for e in entries]
    rewards = [e.get('mean_reward', 0) for e in entries]
    json_valid = [e.get('json_valid_pct', 0) for e in entries]

    # Timing breakdown
    timing = [e.get('timing', {}) for e in entries]
    rollout_ms = [t.get('rollout_ms', 0) for t in timing]
    reward_ms = [t.get('reward_ms', 0) for t in timing]
    gradient_ms = [t.get('gradient_ms', 0) for t in timing]
    sync_ms = [t.get('sync_ms', 0) for t in timing]
    total_ms = [t.get('total_ms', 0) for t in timing]
    ane_ms = [t.get('ane_ms', 0) for t in timing]
    cpu_attn_ms = [t.get('cpu_attn_ms', 0) for t in timing]
    cpu_proj_ms = [t.get('cpu_proj_ms', 0) for t in timing]

    # Power data
    power = [e.get('power', {}) for e in entries]
    cpu_w = [p.get('cpu_w', 0) for p in power]
    gpu_w = [p.get('gpu_w', 0) for p in power]
    ane_w = [p.get('ane_w', 0) for p in power]
    total_w = [p.get('total_w', e.get('power_w', 0)) for p, e in zip(power, entries)]
    cpu_pct = [p.get('cpu_pct', 0) for p in power]

    backend = entries[0].get('backend', 'unknown') if entries else 'unknown'
    model = entries[0].get('model', 'unknown') if entries else 'unknown'

    return {
        'steps': steps, 'rewards': rewards, 'json_valid': json_valid,
        'rollout_ms': rollout_ms, 'reward_ms': reward_ms,
        'gradient_ms': gradient_ms, 'sync_ms': sync_ms,
        'total_ms': total_ms, 'ane_ms': ane_ms,
        'cpu_attn_ms': cpu_attn_ms, 'cpu_proj_ms': cpu_proj_ms,
        'cpu_w': cpu_w, 'gpu_w': gpu_w, 'ane_w': ane_w,
        'total_w': total_w, 'cpu_pct': cpu_pct,
        'backend': backend, 'model': model,
    }


def plot_training_progress(ax, series, label=None):
    """Plot reward and JSON valid % over steps."""
    steps = series['steps']
    color1 = '#2196F3'
    color2 = '#FF9800'

    lbl = label or series['backend']

    ax.plot(steps, series['rewards'], color=color1, marker='o', markersize=4,
            label=f'{lbl} reward', linewidth=1.5)

    ax2 = ax.twinx()
    ax2.plot(steps, series['json_valid'], color=color2, marker='s', markersize=4,
             label=f'{lbl} JSON valid %', linewidth=1.5, linestyle='--')

    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Reward', color=color1)
    ax2.set_ylabel('JSON Valid %', color=color2)
    ax.set_title(f'Training Progress ({series["model"]})')
    ax.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)


def plot_timing_breakdown(ax, series):
    """Stacked bar chart of step timing components."""
    steps = series['steps']
    n = len(steps)
    x = np.arange(n)

    # Compute "other" as total - known components
    other_ms = []
    for i in range(n):
        known = (series['rollout_ms'][i] + series['gradient_ms'][i] + series['sync_ms'][i])
        other_ms.append(max(0, series['total_ms'][i] - known))

    width = 0.7
    bottom = np.zeros(n)

    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
    labels = ['Rollout', 'Gradient', 'Sync', 'ANE (within grad)', 'Other']
    data = [
        series['rollout_ms'],
        series['gradient_ms'],
        series['sync_ms'],
        series['ane_ms'],
        other_ms,
    ]

    for vals, color, label in zip(data, colors, labels):
        ax.bar(x, vals, width, bottom=bottom, color=color, label=label, alpha=0.85)
        bottom += np.array(vals)

    ax.set_xlabel('Step')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Step Timing Breakdown ({series["backend"]})')
    ax.set_xticks(x)
    ax.set_xticklabels(steps)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')


def plot_power_profile(ax, series):
    """Line chart of CPU/GPU/ANE watts per step."""
    steps = series['steps']

    ax.plot(steps, series['cpu_w'], color='#2196F3', marker='o', markersize=3,
            label='CPU', linewidth=1.5)
    ax.plot(steps, series['gpu_w'], color='#4CAF50', marker='s', markersize=3,
            label='GPU', linewidth=1.5)
    ax.plot(steps, series['ane_w'], color='#FF9800', marker='^', markersize=3,
            label='ANE', linewidth=1.5)
    ax.plot(steps, series['total_w'], color='#666666', marker='d', markersize=3,
            label='Total', linewidth=1.5, linestyle='--')

    ax.set_xlabel('Step')
    ax.set_ylabel('Power (W)')
    ax.set_title(f'Power Profile ({series["backend"]})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_component_activity(ax, series):
    """Horizontal bar showing compute distribution per step."""
    steps = series['steps']
    n = len(steps)

    # Normalize timing components within gradient phase
    categories = ['ANE', 'CPU Attention', 'CPU Projection', 'Other']
    colors = ['#FF9800', '#2196F3', '#4CAF50', '#9E9E9E']

    data = np.zeros((n, 4))
    for i in range(n):
        grad_total = max(series['gradient_ms'][i], 1e-6)
        data[i, 0] = series['ane_ms'][i] / grad_total * 100
        data[i, 1] = series['cpu_attn_ms'][i] / grad_total * 100
        data[i, 2] = series['cpu_proj_ms'][i] / grad_total * 100
        data[i, 3] = max(0, 100 - data[i, 0] - data[i, 1] - data[i, 2])

    y = np.arange(n)
    left = np.zeros(n)

    for j, (cat, color) in enumerate(zip(categories, colors)):
        ax.barh(y, data[:, j], left=left, color=color, label=cat, height=0.6)
        left += data[:, j]

    ax.set_yticks(y)
    ax.set_yticklabels([f'Step {s}' for s in steps])
    ax.set_xlabel('% of Gradient Phase')
    ax.set_title(f'Component Activity ({series["backend"]})')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')


def plot_comparison(ax_reward, ax_timing, series_list, labels):
    """Side-by-side comparison of multiple runs."""
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']

    for i, (s, lbl) in enumerate(zip(series_list, labels)):
        c = colors[i % len(colors)]
        ax_reward.plot(s['steps'], s['rewards'], color=c, marker='o', markersize=4,
                       label=lbl, linewidth=1.5)
        ax_timing.plot(s['steps'], s['total_ms'], color=c, marker='o', markersize=4,
                       label=lbl, linewidth=1.5)

    ax_reward.set_xlabel('Step')
    ax_reward.set_ylabel('Mean Reward')
    ax_reward.set_title('Reward Comparison')
    ax_reward.legend(fontsize=8)
    ax_reward.grid(True, alpha=0.3)

    ax_timing.set_xlabel('Step')
    ax_timing.set_ylabel('Total Time (ms)')
    ax_timing.set_title('Timing Comparison')
    ax_timing.legend(fontsize=8)
    ax_timing.grid(True, alpha=0.3)


def export_pgfplots(series, output_dir):
    """Export .dat files for pgfplots inclusion in LaTeX."""
    out = Path(output_dir)

    # Training progress
    with open(out / 'training_progress.dat', 'w') as f:
        f.write('step\treward\tjson_valid_pct\n')
        for i, s in enumerate(series['steps']):
            f.write(f'{s}\t{series["rewards"][i]:.4f}\t{series["json_valid"][i]:.1f}\n')

    # Timing breakdown
    with open(out / 'timing_breakdown.dat', 'w') as f:
        f.write('step\trollout_ms\tgradient_ms\tsync_ms\tane_ms\ttotal_ms\n')
        for i, s in enumerate(series['steps']):
            f.write(f'{s}\t{series["rollout_ms"][i]:.1f}\t{series["gradient_ms"][i]:.1f}\t'
                    f'{series["sync_ms"][i]:.1f}\t{series["ane_ms"][i]:.1f}\t'
                    f'{series["total_ms"][i]:.1f}\n')

    # Power profile
    with open(out / 'power_profile.dat', 'w') as f:
        f.write('step\tcpu_w\tgpu_w\tane_w\ttotal_w\tcpu_pct\n')
        for i, s in enumerate(series['steps']):
            f.write(f'{s}\t{series["cpu_w"][i]:.2f}\t{series["gpu_w"][i]:.2f}\t'
                    f'{series["ane_w"][i]:.2f}\t{series["total_w"][i]:.2f}\t'
                    f'{series["cpu_pct"][i]:.1f}\n')


def main():
    parser = argparse.ArgumentParser(description='Visualize GRPO training logs')
    parser.add_argument('logs', nargs='+', help='JSONL log file(s)')
    parser.add_argument('--output', '-o', default='figs/', help='Output directory')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison plots for multiple logs')
    parser.add_argument('--format', choices=['png', 'pdf', 'both'], default='png',
                        help='Output format (default: png)')
    parser.add_argument('--pgfplots', action='store_true',
                        help='Also export .dat files for pgfplots')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load all log files
    all_series = []
    all_labels = []
    for path in args.logs:
        entries = load_jsonl(path)
        if not entries:
            print(f'Warning: no entries in {path}', file=sys.stderr)
            continue
        s = extract_series(entries)
        all_series.append(s)
        label = f'{s["backend"]}_{s["model"]}'
        all_labels.append(label)
        print(f'Loaded {path}: {len(entries)} steps, backend={s["backend"]}, model={s["model"]}')

    if not all_series:
        print('No valid log entries found.', file=sys.stderr)
        return 1

    formats = []
    if args.format in ('png', 'both'):
        formats.append('png')
    if args.format in ('pdf', 'both'):
        formats.append('pdf')

    # Generate per-run plots
    for s, label in zip(all_series, all_labels):
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        plot_training_progress(ax1, s)

        ax2 = fig.add_subplot(gs[0, 1])
        plot_timing_breakdown(ax2, s)

        ax3 = fig.add_subplot(gs[1, 0])
        plot_power_profile(ax3, s)

        ax4 = fig.add_subplot(gs[1, 1])
        plot_component_activity(ax4, s)

        fig.suptitle(f'GRPO Training: {label}', fontsize=14, fontweight='bold')

        for fmt in formats:
            outpath = os.path.join(args.output, f'{label}.{fmt}')
            fig.savefig(outpath, dpi=150, bbox_inches='tight')
            print(f'Saved {outpath}')
        plt.close(fig)

        if args.pgfplots:
            pgf_dir = os.path.join(args.output, f'{label}_pgfplots')
            os.makedirs(pgf_dir, exist_ok=True)
            export_pgfplots(s, pgf_dir)
            print(f'Exported pgfplots data to {pgf_dir}/')

    # Generate comparison plot if multiple logs
    if args.compare and len(all_series) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        plot_comparison(ax1, ax2, all_series, all_labels)
        fig.suptitle('Multi-Run Comparison', fontsize=14, fontweight='bold')

        for fmt in formats:
            outpath = os.path.join(args.output, f'comparison.{fmt}')
            fig.savefig(outpath, dpi=150, bbox_inches='tight')
            print(f'Saved {outpath}')
        plt.close(fig)

    return 0


if __name__ == '__main__':
    sys.exit(main())
