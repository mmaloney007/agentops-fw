#!/usr/bin/env python3
"""
λ (Lambda) Ablation Experiment for P4: Testing Latency Sensitivity in GRPO

This script orchestrates the λ ablation experiment that tests the latency sensitivity
mechanism in GRPO across three models exhibiting different failure modes:

1. Qwen3-4B on T5 (transient latency spikes)
2. Yi-1.5-6B on Mixed (catastrophic SLO violations)
3. Phi-3-mini on T2 (high latency penalties due to small batch size)

Experiment Design:
- 3 models × 4 λ values [0.0, 0.05, 0.1, 0.2] × 3 seeds [42, 123, 456]
- Total: 36 training runs
- Steps: 1000 per run (50 minutes each on RTX 4090)
- Estimated compute: ~30 GPU-hours total

Hypothesis:
- λ=0.0: No latency penalty. Models may exhibit baseline failure modes (transient spikes,
  catastrophic violations, high variance).
- λ=0.05-0.1: Balanced latency control. Models should improve SLO adherence without
  excessive token overhead.
- λ=0.2: Strong latency penalty. Models may be overly conservative, reducing output quality
  to minimize latency.

Environment Variables:
- LAMBDA_LATENCY: Latency penalty weight (swept parameter)
- MU_COST: Cost penalty weight (fixed at 0.05)
- GAMMA_STABILITY: Stability penalty weight (fixed at 0.1)

The script supports:
- --dry-run: Print commands without executing
- --resume: Skip completed runs (check for manifest.json in output dir)
- Organized output: out/p4_ablation/{model}_{task}_lambda{λ}_seed{seed}/
"""

import argparse
import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Task name -> task file mapping
TASK_FILES = {
    'T1': 'tasks/t1_expanded.jsonl',
    'T2': 'tasks/t2_expanded.jsonl',
    'T3': 'tasks/t3_tools.jsonl',
    'T4': 'tasks/t4_bfcl.jsonl',
    'T5': 'tasks/t5_swebench.jsonl',
    'Mixed': 'tasks/t1t5_balanced.jsonl',
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    'models': [
        {
            'name': 'qwen3_4b',
            'config_preset': 'p4_qwen3_4b_ablation',
            'task': 'T5',
            'description': 'Transient latency spikes'
        },
        {
            'name': 'yi_6b',
            'config_preset': 'p2_yi_6b',
            'task': 'Mixed',
            'description': 'Catastrophic SLO violations'
        },
        {
            'name': 'phi3_mini',
            'config_preset': 'p2_phi3_mini',
            'task': 'T2',
            'description': 'High latency tax (small batch)'
        }
    ],
    'lambda_values': [0.0, 0.05, 0.1, 0.2],
    'seeds': [42, 123, 456],
    'steps': 1000,
    'fixed_weights': {
        'MU_COST': 0.05,
        'GAMMA_STABILITY': 0.1
    }
}


def build_experiment_matrix() -> List[Dict]:
    """
    Build the full experiment matrix.

    Returns:
        List of dicts with keys: model_name, config_preset, task, lambda_val, seed, description
    """
    matrix = []
    for model_info in EXPERIMENT_CONFIG['models']:
        for lambda_val in EXPERIMENT_CONFIG['lambda_values']:
            for seed in EXPERIMENT_CONFIG['seeds']:
                matrix.append({
                    'model_name': model_info['name'],
                    'config_preset': model_info['config_preset'],
                    'task': model_info['task'],
                    'lambda_val': lambda_val,
                    'seed': seed,
                    'description': model_info['description']
                })
    return matrix


def build_output_dir(base_dir: str, run_config: Dict) -> str:
    """
    Build the output directory path for a run.

    Format: out/p4_ablation/{model}_{task}_lambda{λ}_seed{seed}/
    """
    model = run_config['model_name']
    task = run_config['task']
    lambda_val = run_config['lambda_val']
    seed = run_config['seed']

    dir_name = f"{model}_{task}_lambda{lambda_val}_seed{seed}"
    return os.path.join(base_dir, 'p4_ablation', dir_name)


def build_command(
    run_config: Dict,
    output_dir: str,
    steps: int = 1000
) -> str:
    """
    Build the training command for a single run.

    Args:
        run_config: Single experiment config from matrix
        output_dir: Output directory for this run
        steps: Number of training steps

    Returns:
        Full command string with environment variables
    """
    # Build environment variables for reward weights
    env_vars = {
        'LAMBDA_LATENCY': str(run_config['lambda_val']),
        'MU_COST': str(EXPERIMENT_CONFIG['fixed_weights']['MU_COST']),
        'GAMMA_STABILITY': str(EXPERIMENT_CONFIG['fixed_weights']['GAMMA_STABILITY']),
    }

    env_str = ' '.join([f"{k}={v}" for k, v in env_vars.items()])

    # Resolve task file from task name
    task_file = TASK_FILES[run_config['task']]

    # Build the training command — pass seed, lam-latency, and tasks via CLI args
    # (env vars are also set for LAMBDA_LATENCY as belt-and-suspenders)
    cmd = (
        f"{env_str} python -m agent_stable_slo.train.grpo_train_loop "
        f"--config-preset {run_config['config_preset']} "
        f"--tasks {task_file} "
        f"--steps {steps} "
        f"--seed {run_config['seed']} "
        f"--lam-latency {run_config['lambda_val']} "
        f"--out {output_dir}"
    )

    return cmd


def run_experiment(
    matrix: List[Dict],
    base_output_dir: str,
    dry_run: bool = False,
    resume: bool = False
) -> None:
    """
    Execute the experiment matrix.

    Args:
        matrix: List of experiment configurations
        base_output_dir: Base output directory
        dry_run: If True, print commands without executing
        resume: If True, skip runs with existing manifest.json
    """
    total_runs = len(matrix)
    completed_runs = 0
    skipped_runs = 0

    logger.info(f"Starting λ ablation experiment: {total_runs} runs total")
    logger.info(f"Output directory: {base_output_dir}")

    if dry_run:
        logger.warning("DRY RUN MODE: Commands will be printed but not executed")

    for idx, run_config in enumerate(matrix, 1):
        output_dir = build_output_dir(base_output_dir, run_config)

        # Check if run is already completed (if resume mode)
        if resume:
            manifest_path = os.path.join(output_dir, 'manifest.json')
            if os.path.exists(manifest_path):
                logger.info(
                    f"[{idx}/{total_runs}] SKIPPED (already completed): "
                    f"{run_config['model_name']} λ={run_config['lambda_val']} "
                    f"seed={run_config['seed']}"
                )
                skipped_runs += 1
                continue

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = build_command(run_config, output_dir, EXPERIMENT_CONFIG['steps'])

        # Log run info
        logger.info(
            f"[{idx}/{total_runs}] Running: {run_config['model_name']} "
            f"λ={run_config['lambda_val']} seed={run_config['seed']} "
            f"on {run_config['task']} ({run_config['description']})"
        )

        if dry_run:
            logger.info(f"  Command: {cmd}")
            logger.info(f"  Output: {output_dir}")
        else:
            try:
                # Execute the training command
                logger.info(f"  Command: {cmd}")
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=False
                )

                if result.returncode == 0:
                    logger.info("  ✓ Run completed successfully")
                    completed_runs += 1
                else:
                    logger.error(
                        f"  ✗ Run failed with return code {result.returncode}"
                    )
            except Exception as e:
                logger.error(f"  ✗ Run failed with exception: {e}")

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("Experiment Summary:")
    logger.info(f"  Total runs:      {total_runs}")
    logger.info(f"  Completed:       {completed_runs}")
    logger.info(f"  Skipped:         {skipped_runs}")
    logger.info(f"  Failed:          {total_runs - completed_runs - skipped_runs}")
    logger.info("="*70)


def print_experiment_matrix(matrix: List[Dict]) -> None:
    """
    Print a formatted table of the experiment matrix.

    Args:
        matrix: List of experiment configurations
    """
    print("\n" + "="*90)
    print("P4 λ Ablation Experiment Matrix")
    print("="*90)

    # Group by model for cleaner display
    models = {}
    for run in matrix:
        model = run['model_name']
        if model not in models:
            models[model] = {'task': run['task'], 'desc': run['description'], 'runs': []}
        models[model]['runs'].append(run)

    for model_name, model_data in models.items():
        print(f"\n{model_name.upper()} (Task: {model_data['task']}, {model_data['desc']})")
        print("-" * 90)
        print(f"{'λ (Lambda)':<15} {'Seed':<10} {'Output Directory':<65}")
        print("-" * 90)

        for run in model_data['runs']:
            output_dir = build_output_dir('out', run)
            lambda_str = f"{run['lambda_val']:.2f}"
            seed_str = str(run['seed'])
            print(f"{lambda_str:<15} {seed_str:<10} {output_dir:<65}")

    print("\n" + "="*90)
    print(f"Total runs: {len(matrix)} ({len(models)} models × "
          f"{len(EXPERIMENT_CONFIG['lambda_values'])} λ values × "
          f"{len(EXPERIMENT_CONFIG['seeds'])} seeds)")
    print(f"Estimated compute: ~{len(matrix) * 50 / 60:.1f} GPU-hours on RTX 4090")
    print("="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run P4 λ (lambda) ablation experiment for latency sensitivity testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print experiment matrix without running
  python run_lambda_ablation.py --dry-run

  # Run full experiment
  python run_lambda_ablation.py

  # Resume previously interrupted experiment (skip completed runs)
  python run_lambda_ablation.py --resume

  # Save experiment matrix to JSON
  python run_lambda_ablation.py --save-matrix experiment_matrix.json --dry-run
        """
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='out',
        help='Base output directory (default: out)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already-completed runs (checks for manifest.json)'
    )
    parser.add_argument(
        '--save-matrix',
        type=str,
        help='Save experiment matrix to JSON file'
    )
    parser.add_argument(
        '--no-matrix-print',
        action='store_true',
        help='Skip printing experiment matrix'
    )

    args = parser.parse_args()

    # Build experiment matrix
    matrix = build_experiment_matrix()

    # Print matrix unless suppressed
    if not args.no_matrix_print:
        print_experiment_matrix(matrix)

    # Save matrix if requested
    if args.save_matrix:
        with open(args.save_matrix, 'w') as f:
            json.dump(matrix, f, indent=2)
        logger.info(f"Experiment matrix saved to {args.save_matrix}")

    # Run experiment (or print commands)
    if args.dry_run or args.save_matrix:
        # In dry-run mode, still print the commands
        if args.dry_run:
            logger.info("\nDry run - printing commands for first 5 runs:")
            for run_config in matrix[:5]:
                output_dir = build_output_dir(args.output_dir, run_config)
                cmd = build_command(run_config, output_dir)
                print(f"\n{cmd}")
            if len(matrix) > 5:
                logger.info(f"\n... and {len(matrix) - 5} more runs")
    else:
        # Actually run the experiment
        run_experiment(matrix, args.output_dir, dry_run=False, resume=args.resume)


if __name__ == '__main__':
    main()
