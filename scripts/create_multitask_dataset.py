#!/usr/bin/env python3
"""
Create a balanced multi-task training dataset (T1-T5) for Paper 2.

The Problem:
- T1 has 10 examples, T2 has 6 examples
- T3 has 500, T4 has 500, T5 has 300
- Direct concatenation would be dominated by T3/T4/T5

Solution:
- Oversample T1/T2 to match the target count per task
- Create a shuffled, balanced dataset

Usage:
    python scripts/create_multitask_dataset.py --output tasks/t1t5_balanced.jsonl
    python scripts/create_multitask_dataset.py --target-per-task 100 --output tasks/t1t5_100each.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of records."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def oversample(records: List[Dict[str, Any]], target: int, seed: int) -> List[Dict[str, Any]]:
    """Oversample records to reach target count, preserving IDs uniqueness."""
    if not records:
        return []

    rng = random.Random(seed)
    result = []

    # First, include all original records
    result.extend(records)

    # Then sample with replacement to reach target
    while len(result) < target:
        sample = rng.choice(records).copy()
        # Make ID unique to avoid collisions
        sample['id'] = f"{sample['id']}_dup{len(result)}"
        result.append(sample)

    return result[:target]


def undersample(records: List[Dict[str, Any]], target: int, seed: int) -> List[Dict[str, Any]]:
    """Undersample records to reach target count."""
    if len(records) <= target:
        return records

    rng = random.Random(seed)
    return rng.sample(records, target)


def create_balanced_dataset(
    task_files: Dict[str, Path],
    target_per_task: int,
    strategy: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """Create balanced dataset from multiple task files."""

    all_records = []
    stats = {}

    for task_name, path in task_files.items():
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {task_name}")
            continue

        records = load_jsonl(path)
        original_count = len(records)

        if strategy == 'oversample':
            # Oversample small tasks, undersample large tasks
            balanced = oversample(records, target_per_task, seed) if len(records) < target_per_task else undersample(records, target_per_task, seed)
        elif strategy == 'undersample':
            # Undersample everything to min task size
            balanced = undersample(records, target_per_task, seed)
        else:  # 'natural'
            balanced = records

        stats[task_name] = {
            'original': original_count,
            'balanced': len(balanced),
        }

        all_records.extend(balanced)

    # Shuffle
    rng = random.Random(seed)
    rng.shuffle(all_records)

    return all_records, stats


def main():
    parser = argparse.ArgumentParser(description='Create balanced multi-task dataset')
    parser.add_argument('--output', '-o', type=str, default='tasks/t1t5_balanced.jsonl',
                        help='Output JSONL file path')
    parser.add_argument('--target-per-task', '-t', type=int, default=100,
                        help='Target number of examples per task type (default: 100)')
    parser.add_argument('--strategy', '-s', choices=['oversample', 'undersample', 'natural'],
                        default='oversample',
                        help='Balancing strategy (default: oversample)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--tasks-dir', type=str, default='tasks',
                        help='Directory containing task JSONL files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print stats without writing output')
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)

    # Define task files
    # Use expanded T2 if available (100+ examples), otherwise fall back to original (6 examples)
    t2_file = tasks_dir / 't2_expanded.jsonl'
    if not t2_file.exists():
        t2_file = tasks_dir / 't2_grounded.jsonl'
        print(f"  Note: Using original T2 file ({t2_file}). Run expand_t2_tasks.py for more T2 examples.")

    task_files = {
        'T1': tasks_dir / 't1_structured.jsonl',
        'T2': t2_file,
        'T3': tasks_dir / 't3_tools.jsonl',
        'T4': tasks_dir / 't4_bfcl.jsonl',
        'T5': tasks_dir / 't5_swebench.jsonl',
    }

    print("=" * 60)
    print("  Creating Balanced Multi-Task Dataset (T1-T5)")
    print("=" * 60)
    print(f"\n  Target per task: {args.target_per_task}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {args.output}")
    print()

    # Create balanced dataset
    records, stats = create_balanced_dataset(
        task_files=task_files,
        target_per_task=args.target_per_task,
        strategy=args.strategy,
        seed=args.seed,
    )

    # Print stats
    print("Task Statistics:")
    print("-" * 50)
    print(f"  {'Task':<6} {'Original':<12} {'Balanced':<12} {'Change':<12}")
    print("-" * 50)

    for task, s in stats.items():
        change = s['balanced'] - s['original']
        change_str = f"+{change}" if change > 0 else str(change)
        print(f"  {task:<6} {s['original']:<12} {s['balanced']:<12} {change_str:<12}")

    print("-" * 50)
    print(f"  {'TOTAL':<6} {sum(s['original'] for s in stats.values()):<12} {len(records):<12}")
    print()

    # Check task distribution
    task_counts = Counter(r.get('task_type', 'unknown') for r in records)
    print("Final Task Type Distribution:")
    for task_type, count in sorted(task_counts.items()):
        print(f"  {task_type}: {count}")
    print()

    if args.dry_run:
        print("DRY RUN - No file written")
        return 0

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Wrote {len(records)} records to {output_path}")
    print()

    # Print training recommendations
    print("Training Recommendations:")
    print("-" * 50)
    total_examples = len(records)
    print(f"  Total examples: {total_examples}")
    print(f"  Recommended steps for 1 epoch: {total_examples}")
    print(f"  Recommended steps for 3 epochs: {total_examples * 3}")
    print(f"  Conservative (500 steps): Covers {500 / total_examples:.1%} of dataset per epoch")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
