#!/usr/bin/env python3
"""
Reconcile P2 training results and generate consistent paper tables.

This script:
1. Reads results/p2_training_results.csv
2. Filters to 500-step runs only (excluding 250-step and old runs)
3. Computes mean/std across 3 seeds
4. Outputs LaTeX table for paper

Usage:
    python scripts/reconcile_p2_results.py
    python scripts/reconcile_p2_results.py --check  # Verify against paper values
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


def load_results(csv_path: Path) -> list:
    """Load CSV results."""
    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def filter_500_step_runs(results: list) -> dict:
    """Filter to only 500-step runs with valid seeds (42, 123, 456)."""
    filtered = defaultdict(list)

    valid_seeds = {'42', '123', '456'}

    for row in results:
        steps = row.get('steps', '')
        seed = row.get('seed', '')
        model = row.get('model', '')

        # Skip non-500-step runs
        if steps != '500':
            continue

        # Skip old runs with unknown seeds
        if seed not in valid_seeds:
            continue

        filtered[model].append({
            'seed': seed,
            'valid_pct': float(row.get('valid_pct', 0)),
            'last50_pct': float(row.get('last50_pct', 0)),
            'avg_reward': float(row.get('avg_reward', 0)),
        })

    return filtered


def compute_aggregates(filtered: dict) -> dict:
    """Compute mean and std for each model across seeds."""
    aggregates = {}

    for model, runs in filtered.items():
        if len(runs) < 3:
            print(f"  WARNING: {model} has only {len(runs)} seeds, expected 3")

        valid_pcts = [r['valid_pct'] for r in runs]
        last50_pcts = [r['last50_pct'] for r in runs]
        rewards = [r['avg_reward'] for r in runs]

        aggregates[model] = {
            'n_seeds': len(runs),
            'valid_pct_mean': statistics.mean(valid_pcts) if valid_pcts else 0,
            'valid_pct_std': statistics.stdev(valid_pcts) if len(valid_pcts) > 1 else 0,
            'last50_pct_mean': statistics.mean(last50_pcts) if last50_pcts else 0,
            'last50_pct_std': statistics.stdev(last50_pcts) if len(last50_pcts) > 1 else 0,
            'avg_reward_mean': statistics.mean(rewards) if rewards else 0,
            'avg_reward_std': statistics.stdev(rewards) if len(rewards) > 1 else 0,
            'seeds': [r['seed'] for r in runs],
            'last50_values': last50_pcts,
        }

    return aggregates


def generate_latex_table(aggregates: dict) -> str:
    """Generate LaTeX table for paper."""
    # Sort by model size (approximate)
    size_order = {
        'llama-3.2-1b': 1,
        'llama-3.2-3b': 3,
        'qwen2.5-3b': 3,
        'phi-3-mini': 4,
        'qwen3-4b': 4,
        'yi-1.5-6b': 6,
        'mistral-7b': 7,
        'ministral-8b': 8,
        'llama-3.1-8b': 8,
        'gemma-2-9b': 9,
        'gemma-3-12b': 12,
    }

    sorted_models = sorted(aggregates.keys(), key=lambda m: size_order.get(m, 100))

    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Model & Size & Valid \\% & Last-50 \\% & Learning? \\\\",
        "\\midrule",
    ]

    for model in sorted_models:
        agg = aggregates[model]

        # Determine if model shows learning
        # Learning = Last-50 > 30% consistently
        learning = "Yes" if agg['last50_pct_mean'] >= 30 else "No"
        if 0 < agg['last50_pct_mean'] < 30:
            learning = "Partial"

        # Get size from model name
        size = size_order.get(model, '?')

        # Format with std dev
        valid_str = f"{agg['valid_pct_mean']:.1f} ± {agg['valid_pct_std']:.1f}"
        last50_str = f"{agg['last50_pct_mean']:.1f} ± {agg['last50_pct_std']:.1f}"

        lines.append(f"{model} & {size}B & {valid_str} & {last50_str} & {learning} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])

    return '\n'.join(lines)


def generate_markdown_table(aggregates: dict) -> str:
    """Generate markdown table for verification."""
    size_order = {
        'llama-3.2-1b': 1, 'llama-3.2-3b': 3, 'qwen2.5-3b': 3,
        'phi-3-mini': 4, 'qwen3-4b': 4, 'yi-1.5-6b': 6,
        'mistral-7b': 7, 'ministral-8b': 8, 'llama-3.1-8b': 8,
        'gemma-2-9b': 9, 'gemma-3-12b': 12,
    }

    sorted_models = sorted(aggregates.keys(), key=lambda m: size_order.get(m, 100))

    lines = [
        "| Model | Size | Seeds | Valid % (mean±std) | Last-50 % (mean±std) | Learning? |",
        "|-------|------|-------|-------------------|---------------------|-----------|",
    ]

    for model in sorted_models:
        agg = aggregates[model]

        learning = "Yes" if agg['last50_pct_mean'] >= 30 else "No"
        if 0 < agg['last50_pct_mean'] < 30:
            learning = "Partial"

        size = size_order.get(model, '?')
        valid_str = f"{agg['valid_pct_mean']:.1f} ± {agg['valid_pct_std']:.1f}"
        last50_str = f"{agg['last50_pct_mean']:.1f} ± {agg['last50_pct_std']:.1f}"

        lines.append(f"| {model} | {size}B | {agg['n_seeds']} | {valid_str} | {last50_str} | {learning} |")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Reconcile P2 results')
    parser.add_argument('--csv', default='results/p2_training_results.csv',
                        help='Path to results CSV')
    parser.add_argument('--check', action='store_true',
                        help='Show detailed breakdown for verification')
    parser.add_argument('--latex', action='store_true',
                        help='Output LaTeX table')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return 1

    print("=" * 70)
    print("  P2 RESULTS RECONCILIATION")
    print("=" * 70)
    print()

    # Load and filter
    print(f"Loading {csv_path}...")
    results = load_results(csv_path)
    print(f"  Total rows: {len(results)}")

    filtered = filter_500_step_runs(results)
    print(f"  Filtered to 500-step runs: {sum(len(v) for v in filtered.values())} runs across {len(filtered)} models")
    print()

    # Compute aggregates
    aggregates = compute_aggregates(filtered)

    # Show detailed breakdown if requested
    if args.check:
        print("Detailed breakdown (500-step runs only):")
        print("-" * 70)
        for model, agg in sorted(aggregates.items()):
            print(f"\n{model}:")
            print(f"  Seeds: {agg['seeds']}")
            print(f"  Last-50 values: {agg['last50_values']}")
            print(f"  Mean Last-50: {agg['last50_pct_mean']:.1f}% ± {agg['last50_pct_std']:.1f}%")
        print()

    # Generate tables
    print("\n" + "=" * 70)
    print("  RECONCILED RESULTS (500-step, 3 seeds)")
    print("=" * 70)
    print()
    print(generate_markdown_table(aggregates))

    if args.latex:
        print()
        print("LaTeX table:")
        print("-" * 70)
        print(generate_latex_table(aggregates))

    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    # Count learning models
    learning = [m for m, a in aggregates.items() if a['last50_pct_mean'] >= 30]
    partial = [m for m, a in aggregates.items() if 0 < a['last50_pct_mean'] < 30]
    no_learning = [m for m, a in aggregates.items() if a['last50_pct_mean'] == 0]

    print(f"Models showing sustained learning (Last-50 >= 30%): {len(learning)}")
    for m in learning:
        print(f"  - {m}: {aggregates[m]['last50_pct_mean']:.1f}%")

    print(f"\nModels showing partial learning (0 < Last-50 < 30%): {len(partial)}")
    for m in partial:
        print(f"  - {m}: {aggregates[m]['last50_pct_mean']:.1f}%")

    print(f"\nModels showing no learning (Last-50 = 0%): {len(no_learning)}")
    for m in no_learning:
        print(f"  - {m}")

    return 0


if __name__ == '__main__':
    exit(main())
