#!/usr/bin/env python3
"""
Enhanced smoke test for P2 training on MacBook Pro M2 Max (64GB).

This script validates the FULL training pipeline:
1. MPS backend and memory
2. Model loading with LoRA
3. Actual training steps (gradient computation)
4. Time estimation for full training runs

Use this BEFORE starting long training runs to catch issues early.

Usage:
    python scripts/smoke_test_p2_training.py
    python scripts/smoke_test_p2_training.py --model Qwen/Qwen2.5-3B-Instruct --steps 5
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_status(test: str, passed: bool, details: str = ""):
    status = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"  [{color}{status}{reset}] {test}")
    if details:
        for line in details.split('\n'):
            print(f"        {line}")


def test_environment() -> Tuple[bool, dict]:
    """Test basic environment setup."""
    results = {}

    # Python version
    version = sys.version_info
    results['python'] = version >= (3, 10)

    # Memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        results['memory_gb'] = mem.total / (1024**3)
        results['memory_ok'] = results['memory_gb'] >= 32
    except ImportError:
        results['memory_ok'] = True  # Assume OK if can't check

    # MPS
    try:
        import torch
        results['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        results['cuda_available'] = torch.cuda.is_available()

        if results['mps_available']:
            results['device'] = 'mps'
        elif results['cuda_available']:
            results['device'] = 'cuda'
        else:
            results['device'] = 'cpu'
    except ImportError:
        results['mps_available'] = False
        results['cuda_available'] = False
        results['device'] = 'cpu'

    return all([results.get('python'), results.get('mps_available') or results.get('cuda_available')]), results


def test_model_loading(model_id: str, device: str) -> Tuple[bool, dict]:
    """Test model loading with LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    results = {}

    try:
        print("        Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"        Loading model to {device}...")
        start = time.time()

        # Use float16 for MPS, bfloat16 for CUDA
        dtype = torch.float16 if device == 'mps' else torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device if device != 'mps' else None,  # MPS doesn't support device_map
            trust_remote_code=True,
        )

        if device == 'mps':
            model = model.to(device)

        results['load_time'] = time.time() - start

        # Attach LoRA
        print("        Attaching LoRA adapter...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()

        results['trainable_params'] = trainable
        results['total_params'] = total
        results['trainable_pct'] = 100 * trainable / total
        results['model'] = model
        results['tokenizer'] = tokenizer
        results['dtype'] = str(dtype)

        return True, results

    except Exception as e:
        results['error'] = str(e)
        return False, results


def test_training_step(
    model,
    tokenizer,
    device: str,
    num_steps: int = 5,
) -> Tuple[bool, dict]:
    """Test actual training steps with gradient computation."""
    import torch

    results = {
        'step_times': [],
        'memory_usage': [],
    }

    # Sample prompts (mini version of T1)
    prompts = [
        "Given an incident, classify severity. Incident: API latency spike. Return JSON: {\"severity\": \"high\"}",
        "Given an incident, classify severity. Incident: Disk usage at 50%. Return JSON: {\"severity\": \"low\"}",
        "Given an incident, classify severity. Incident: Database connection failed. Return JSON: {\"severity\": \"critical\"}",
    ]

    try:
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        model.train()

        print(f"        Running {num_steps} training steps...")

        for step in range(num_steps):
            prompt = prompts[step % len(prompts)]

            step_start = time.time()

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_time = time.time() - step_start
            results['step_times'].append(step_time)

            # Memory tracking
            if device == 'mps':
                # MPS doesn't have memory_allocated, estimate from system
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    results['memory_usage'].append(mem.used / (1024**3))
                except Exception:
                    pass
            elif device == 'cuda':
                results['memory_usage'].append(torch.cuda.memory_allocated() / (1024**3))

            print(f"          Step {step+1}/{num_steps}: loss={loss.item():.4f}, time={step_time:.2f}s")

        # Clear cache
        if device == 'mps':
            torch.mps.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()

        results['avg_step_time'] = sum(results['step_times']) / len(results['step_times'])
        results['total_time'] = sum(results['step_times'])

        return True, results

    except Exception as e:
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()
        return False, results


def estimate_training_time(
    step_time: float,
    target_steps: int,
    num_seeds: int = 3,
) -> dict:
    """Estimate total training time for different configurations."""
    estimates = {}

    # Single run
    single_run_seconds = step_time * target_steps
    single_run_hours = single_run_seconds / 3600

    # Multiple seeds
    full_run_hours = single_run_hours * num_seeds

    estimates['step_time_seconds'] = step_time
    estimates['steps_per_minute'] = 60 / step_time
    estimates['single_run'] = {
        'steps': target_steps,
        'hours': single_run_hours,
        'human': f"{single_run_hours:.1f}h" if single_run_hours >= 1 else f"{single_run_hours*60:.0f}m"
    }
    estimates['full_study'] = {
        'seeds': num_seeds,
        'total_runs': num_seeds,
        'hours': full_run_hours,
        'human': f"{full_run_hours:.1f}h" if full_run_hours >= 1 else f"{full_run_hours*60:.0f}m"
    }

    return estimates


def main():
    parser = argparse.ArgumentParser(description='P2 Training Smoke Test')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Model to test (default: Qwen2.5-0.5B for speed)')
    parser.add_argument('--steps', type=int, default=5,
                        help='Number of training steps to test')
    parser.add_argument('--target-steps', type=int, default=500,
                        help='Target steps for time estimation')
    args = parser.parse_args()

    print_header("P2 TRAINING SMOKE TEST")
    print(f"  Model: {args.model}")
    print(f"  Test steps: {args.steps}")
    print(f"  Target steps for estimation: {args.target_steps}")

    all_passed = True

    # 1. Environment
    print_header("1. Environment Check")
    passed, env_results = test_environment()
    all_passed = all_passed and passed

    print_status("Python version >= 3.10", env_results.get('python', False))
    mem_gb = env_results.get('memory_gb')
    mem_str = f"{mem_gb:.1f}GB available" if mem_gb else "unknown"
    print_status(
        "Memory >= 32GB",
        env_results.get('memory_ok', True),
        mem_str
    )
    print_status(
        "Compute device available",
        env_results.get('mps_available') or env_results.get('cuda_available'),
        f"Device: {env_results.get('device', 'none')}"
    )

    device = env_results.get('device', 'cpu')
    if device == 'cpu':
        print("\n  WARNING: No GPU detected. Training will be very slow.")

    # 2. Model Loading
    print_header("2. Model Loading + LoRA")
    passed, model_results = test_model_loading(args.model, device)
    all_passed = all_passed and passed

    if passed:
        print_status(
            "Model loaded",
            True,
            f"Loaded in {model_results['load_time']:.1f}s"
        )
        print_status(
            "LoRA attached",
            True,
            f"Trainable: {model_results['trainable_params']:,} / {model_results['total_params']:,} ({model_results['trainable_pct']:.2f}%)"
        )
    else:
        print_status("Model loading", False, model_results.get('error', 'Unknown error'))
        print("\n  FATAL: Cannot continue without model. Check error above.")
        return 1

    # 3. Training Steps
    print_header("3. Training Step Validation")
    passed, train_results = test_training_step(
        model_results['model'],
        model_results['tokenizer'],
        device,
        args.steps,
    )
    all_passed = all_passed and passed

    if passed:
        print_status(
            "Training steps completed",
            True,
            f"Avg step time: {train_results['avg_step_time']:.2f}s"
        )
    else:
        print_status("Training steps", False, train_results.get('error', 'Unknown error'))
        if 'traceback' in train_results:
            print(f"\n  Traceback:\n{train_results['traceback']}")
        return 1

    # 4. Time Estimation
    print_header("4. Time Estimation")

    # Scale step time based on model size ratio
    # Qwen2.5-0.5B is our test model, estimate for target model
    test_params = model_results['total_params']
    step_time = train_results['avg_step_time']

    # Typical model sizes for estimation
    model_sizes = {
        'Qwen2.5-0.5B': 500_000_000,
        'Qwen2.5-3B': 3_000_000_000,
        'Qwen3-4B': 4_000_000_000,
        'Yi-1.5-6B': 6_000_000_000,
        'Mistral-7B': 7_000_000_000,
        'Llama-3.1-8B': 8_000_000_000,
        'Gemma-2-9B': 9_000_000_000,
        'Gemma-3-12B': 12_000_000_000,
    }

    print(f"\n  Measured step time for {args.model}: {step_time:.2f}s")
    print(f"\n  Estimated training times for {args.target_steps} steps (single seed):")
    print(f"  {'-'*60}")
    print(f"  {'Model':<20} {'Est. Step Time':<15} {'Total Time':<15} {'3 Seeds':<15}")
    print(f"  {'-'*60}")

    for model_name, params in model_sizes.items():
        # Rough scaling: step time proportional to sqrt(params) for LoRA
        # This is an approximation - actual scaling depends on many factors
        scale = (params / test_params) ** 0.6  # Sub-linear scaling due to LoRA
        est_step = step_time * scale
        est_total = est_step * args.target_steps / 3600
        est_3seeds = est_total * 3

        time_str = f"{est_total:.1f}h" if est_total >= 1 else f"{est_total*60:.0f}m"
        seeds_str = f"{est_3seeds:.1f}h" if est_3seeds >= 1 else f"{est_3seeds*60:.0f}m"

        print(f"  {model_name:<20} {est_step:.2f}s{'':<10} {time_str:<15} {seeds_str:<15}")

    print(f"  {'-'*60}")

    # 5. Multi-task training estimate
    print_header("5. Multi-Task Training Estimate (T1-T5)")

    # For balanced 500 examples (100 per task)
    total_examples = 500
    epochs_for_500_steps = 500 / total_examples

    print("\n  With balanced T1-T5 dataset (100 examples/task = 500 total):")
    print(f"  - 500 steps = {epochs_for_500_steps:.1f} epochs over dataset")
    print(f"  - For 3 epochs coverage: {total_examples * 3} steps recommended")
    print()

    # Full study estimate for Gemma-3-12B (the hardest model)
    gemma_scale = (12_000_000_000 / test_params) ** 0.6
    gemma_step = step_time * gemma_scale
    gemma_500 = gemma_step * 500 / 3600
    gemma_1500 = gemma_step * 1500 / 3600
    gemma_full = gemma_1500 * 3  # 3 seeds

    print("  Gemma-3-12B (largest model) estimates:")
    print(f"    - 500 steps × 1 seed:  {gemma_500:.1f}h")
    print(f"    - 1500 steps × 1 seed: {gemma_1500:.1f}h")
    print(f"    - 1500 steps × 3 seeds: {gemma_full:.1f}h ({gemma_full/24:.1f} days)")

    # Summary
    print_header("SUMMARY")

    if all_passed:
        print("\n  \033[92mALL TESTS PASSED\033[0m")
        print("\n  Your system is ready for P2 training!")
        print("\n  Recommended next steps:")
        print("  1. Create balanced dataset:")
        print("     python scripts/create_multitask_dataset.py -t 100 -o tasks/t1t5_balanced.jsonl")
        print()
        print("  2. Start training (smallest model first):")
        print("     python -m agent_stable_slo.train.grpo_train_loop \\")
        print("         --model Qwen/Qwen2.5-3B-Instruct \\")
        print("         --tasks tasks/t1t5_balanced.jsonl \\")
        print("         --steps 500 --seed 42")
    else:
        print("\n  \033[91mSOME TESTS FAILED\033[0m")
        print("\n  Please resolve the issues above before starting training.")

    # Cleanup
    if 'model' in model_results:
        del model_results['model']
        del model_results['tokenizer']
        if device == 'mps':
            import torch
            torch.mps.empty_cache()

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
