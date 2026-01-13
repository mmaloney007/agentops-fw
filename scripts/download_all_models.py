#!/usr/bin/env python3
"""
Download all 4 models for P2 training experiments.
Author: Mike Maloney <mike.maloney@unh.edu>
"""

import os
import sys
from datetime import datetime

def download_model(repo_id: str, local_dir: str, token: str = None):
    """Download a model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"Target: {local_dir}")
    print(f"Started: {datetime.now().isoformat()}")
    print('='*60)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=['*.gguf', '*.bin', '*.ot'],
            token=token
        )
        print(f"✓ Completed: {repo_id}")
        return True
    except Exception as e:
        print(f"✗ Failed: {repo_id}")
        print(f"  Error: {e}")
        return False


def main():
    # Model configurations: (HuggingFace repo, local directory)
    models = [
        ("Qwen/Qwen3-4B", "./models/qwen3-4b"),
        ("mistralai/Ministral-8B-Instruct-2410", "./models/ministral-3b-instruct"),  # Using 8B as 3B may not exist
        ("google/gemma-3-12b-it", "./models/gemma-3-12b-it"),
        # GPT-OSS-20B may need special handling if it's a private/custom model
    ]

    # Check for HF token (needed for gated models like Gemma)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Warning: No HF_TOKEN found. Gated models (like Gemma) may fail.")
        print("Set HF_TOKEN environment variable if needed.")

    print(f"\n{'#'*60}")
    print("# Model Download Script for P2 Training")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"# Models to download: {len(models)}")
    print(f"{'#'*60}")

    results = {}
    for repo_id, local_dir in models:
        results[repo_id] = download_model(repo_id, local_dir, token)

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print('='*60)
    for repo_id, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {status}: {repo_id}")

    print(f"\nCompleted: {datetime.now().isoformat()}")

    # Return non-zero if any failed
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
