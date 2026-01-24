#!/usr/bin/env python3
"""
Smoke test for MacBook Pro M2 Max (64GB RAM) with MPS backend.

This script validates:
1. MPS backend availability
2. PyTorch MPS support
3. Transformers + PEFT + BitsAndBytes compatibility
4. Small model loading (Qwen2.5-3B as test case)
5. Basic inference
6. LoRA adapter attachment

Run this BEFORE starting any training to catch issues early.

Usage:
    python scripts/smoke_test_mps.py
"""

import sys
import os
import time

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_status(test: str, passed: bool, details: str = ""):
    status = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"  [{color}{status}{reset}] {test}")
    if details:
        print(f"        {details}")

def test_python_version():
    """Check Python version >= 3.10"""
    version = sys.version_info
    passed = version >= (3, 10)
    details = f"Python {version.major}.{version.minor}.{version.micro}"
    print_status("Python version >= 3.10", passed, details)
    return passed

def test_mps_availability():
    """Check if MPS (Metal Performance Shaders) is available"""
    try:
        import torch
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        details = f"torch.backends.mps.is_available() = {has_mps}"
        print_status("MPS backend available", has_mps, details)
        return has_mps
    except ImportError:
        print_status("MPS backend available", False, "PyTorch not installed")
        return False

def test_torch_mps_device():
    """Check if we can create tensors on MPS device"""
    try:
        import torch
        if not torch.backends.mps.is_available():
            print_status("MPS tensor creation", False, "MPS not available")
            return False

        # Create a small tensor on MPS
        x = torch.randn(100, 100, device='mps')
        y = torch.randn(100, 100, device='mps')
        z = torch.matmul(x, y)

        passed = z.device.type == 'mps'
        details = f"Created 100x100 tensor, matmul on MPS, device={z.device}"
        print_status("MPS tensor creation", passed, details)
        return passed
    except Exception as e:
        print_status("MPS tensor creation", False, str(e))
        return False

def test_transformers_import():
    """Check transformers library"""
    try:
        import transformers
        version = transformers.__version__
        print_status("Transformers import", True, f"version {version}")
        return True
    except ImportError as e:
        print_status("Transformers import", False, str(e))
        return False

def test_peft_import():
    """Check PEFT (LoRA) library"""
    try:
        import peft
        version = peft.__version__
        print_status("PEFT (LoRA) import", True, f"version {version}")
        return True
    except ImportError as e:
        print_status("PEFT (LoRA) import", False, str(e))
        return False

def test_bitsandbytes_import():
    """Check bitsandbytes library (may not work on MPS)"""
    try:
        import bitsandbytes as bnb
        version = bnb.__version__
        print_status("BitsAndBytes import", True, f"version {version}")
        return True
    except ImportError:
        # BitsAndBytes often doesn't work on MPS - this is expected
        print_status("BitsAndBytes import", False, "Not available (expected on MPS)")
        return False

def test_memory_info():
    """Report system memory"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        passed = total_gb >= 32  # Need at least 32GB for training
        details = f"Total: {total_gb:.1f}GB, Available: {available_gb:.1f}GB"
        print_status(f"RAM >= 32GB", passed, details)
        return passed
    except ImportError:
        print_status("RAM check", False, "psutil not installed")
        return False

def test_small_model_load():
    """Test loading a small model (Qwen2.5-0.5B as quick test)"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Use smallest Qwen model for quick test
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        print(f"        Loading {model_id}...")
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Try MPS first, fall back to CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

        elapsed = time.time() - start
        details = f"Loaded in {elapsed:.1f}s on {device}"
        print_status("Small model load (Qwen2.5-0.5B)", True, details)

        # Quick inference test
        inputs = tokenizer("Hello, I am", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print_status("Basic inference", True, f"Output: '{decoded[:50]}...'")

        # Cleanup
        del model
        del tokenizer
        if device == "mps":
            torch.mps.empty_cache()

        return True

    except Exception as e:
        print_status("Small model load", False, str(e)[:100])
        return False

def test_lora_attachment():
    """Test LoRA adapter creation"""
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"        Loading model for LoRA test...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

        # Create LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Attach LoRA
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()

        pct = 100 * trainable / total
        details = f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)"
        print_status("LoRA adapter attachment", True, details)

        # Cleanup
        del model
        if device == "mps":
            torch.mps.empty_cache()

        return True

    except Exception as e:
        print_status("LoRA adapter attachment", False, str(e)[:100])
        return False

def main():
    print_header("SMOKE TEST: MacBook Pro M2 Max (MPS Backend)")
    print(f"  Testing training feasibility on Apple Silicon")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Basic checks
    print_header("1. Environment Checks")
    results['python'] = test_python_version()
    results['memory'] = test_memory_info()

    # MPS checks
    print_header("2. MPS Backend Checks")
    results['mps_available'] = test_mps_availability()
    results['mps_tensor'] = test_torch_mps_device()

    # Library checks
    print_header("3. Library Checks")
    results['transformers'] = test_transformers_import()
    results['peft'] = test_peft_import()
    results['bnb'] = test_bitsandbytes_import()  # Expected to fail on MPS

    # Model checks (slower)
    print_header("4. Model Loading Checks")
    results['model_load'] = test_small_model_load()
    results['lora'] = test_lora_attachment()

    # Summary
    print_header("SUMMARY")

    critical_tests = ['python', 'mps_available', 'mps_tensor', 'transformers', 'peft', 'model_load', 'lora']
    critical_passed = all(results.get(t, False) for t in critical_tests)

    if critical_passed:
        print("\n  \033[92mALL CRITICAL TESTS PASSED\033[0m")
        print("\n  Your MacBook Pro M2 Max is ready for P2 training!")
        print("\n  Recommended models for MPS training:")
        print("    - Qwen2.5-3B (3B params, ~6GB)")
        print("    - Qwen3-4B (4B params, ~8GB)")
        print("    - Yi-1.5-6B (6B params, ~12GB)")
        print("    - Mistral-7B (7B params, ~14GB)")
        print("    - Gemma-2-9B (9B params, ~18GB) - may be tight")
        print("    - Gemma-3-12B (12B params, ~24GB) - needs float16, no LoRA")
        print("\n  Note: BitsAndBytes 4-bit quantization doesn't work on MPS.")
        print("  Use float16 instead. Models up to ~9B should fit in 64GB RAM.")
    else:
        print("\n  \033[91mSOME CRITICAL TESTS FAILED\033[0m")
        print("\n  Failed tests:")
        for t in critical_tests:
            if not results.get(t, False):
                print(f"    - {t}")
        print("\n  Please resolve these issues before training.")

    return 0 if critical_passed else 1

if __name__ == "__main__":
    sys.exit(main())
