#!/usr/bin/env python3
"""
Build T5 task suite from SWE-bench Lite.

SWE-bench Lite is a 300-task subset of SWE-bench testing the ability to
resolve real GitHub issues. This represents end-to-end software engineering
capability - the ultimate test for SLO-aware agents.

Source: https://github.com/SWE-bench/SWE-bench
Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite
Paper: Jimenez et al., "SWE-bench: Can Language Models Resolve Real-world Github Issues?"
"""

import argparse
import json
from pathlib import Path


def load_swebench_lite(cache_dir: Path) -> list[dict]:
    """Load SWE-bench Lite from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets package required. Install with: pip install datasets")
        return []

    cache_file = cache_dir / "swebench_lite.json"
    if cache_file.exists():
        print("  Using cached SWE-bench Lite")
        with open(cache_file) as f:
            return json.load(f)

    print("  Downloading SWE-bench Lite from HuggingFace...")
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        items = [dict(item) for item in dataset]

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(items, f)

        return items
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return []


def convert_swebench_to_task(item: dict) -> dict | None:
    """Convert a SWE-bench item to our task format."""
    try:
        instance_id = item.get("instance_id", "")
        repo = item.get("repo", "")
        problem = item.get("problem_statement", "")
        hints = item.get("hints_text", "")
        base_commit = item.get("base_commit", "")
        fail_to_pass = item.get("FAIL_TO_PASS", "[]")

        if not problem:
            return None

        # Parse target tests
        try:
            target_tests = json.loads(fail_to_pass)
        except Exception:
            target_tests = []

        # Build prompt (structured for SLO-aware agent)
        prompt = f"""Resolve the following GitHub issue by generating a patch.

Repository: {repo}
Base commit: {base_commit}

## Issue

{problem}
"""
        if hints and hints.strip():
            prompt += f"""
## Discussion/Hints

{hints}
"""

        if target_tests:
            prompt += f"""
## Target Tests (must pass after fix)

{chr(10).join("- " + t for t in target_tests[:5])}
"""

        prompt += """
## Output Format

Provide your solution as a unified diff patch that resolves the issue.
"""

        return {
            "id": f"t5_{instance_id}",
            "task_type": "t5",
            "source": "swebench_lite",
            "repo": repo,
            "instance_id": instance_id,
            "base_commit": base_commit,
            "prompt": prompt,
            "schema_path": "tasks/schemas/t5_patch_schema.json",
            "gold": {
                "patch": item.get("patch", ""),
                "test_patch": item.get("test_patch", ""),
                "fail_to_pass": target_tests,
            },
        }
    except Exception as e:
        print(f"  Warning: Failed to convert {item.get('instance_id', 'unknown')}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Build T5 SWE-bench Lite task suite")
    parser.add_argument("--out", default="tasks/t5_swebench.jsonl", help="Output file")
    parser.add_argument(
        "--count", type=int, default=300, help="Max tasks (300 = full Lite)"
    )
    parser.add_argument(
        "--cache-dir", default=".cache/swebench", help="Cache directory"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    print("Building T5 (SWE-bench Lite) task suite...")

    items = load_swebench_lite(cache_dir)
    if not items:
        print("Failed to load SWE-bench Lite. Exiting.")
        return

    print(f"  Loaded {len(items)} items")

    tasks = []
    for item in items[: args.count]:
        task = convert_swebench_to_task(item)
        if task:
            tasks.append(task)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    print(f"Wrote {len(tasks)} tasks to {out_path}")

    # Summary by repo
    by_repo = {}
    for t in tasks:
        repo = t.get("repo", "unknown")
        by_repo[repo] = by_repo.get(repo, 0) + 1
    print("By repository:", dict(sorted(by_repo.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
