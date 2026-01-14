#!/usr/bin/env python3
"""
Build T4 task suite from Berkeley Function Calling Leaderboard (BFCL) v4.

BFCL is the industry-standard benchmark for function/tool calling evaluation.
We use the "simple" categories for single-function calls that align with our
SLO-aware structured output thesis.

Source: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
Paper: https://openreview.net/forum?id=2GmDdhBdDk
Install: pip install bfcl-eval
"""

import argparse
import json
from pathlib import Path


def get_bfcl_data_dir() -> Path | None:
    """Find BFCL data directory from installed package."""
    try:
        import bfcl_eval
        pkg_path = Path(bfcl_eval.__file__).parent
        data_dir = pkg_path / "data"
        if data_dir.exists():
            return data_dir
    except ImportError:
        pass
    return None


def load_bfcl_file(data_dir: Path, filename: str) -> list[dict]:
    """Load a BFCL JSON file (one JSON object per line)."""
    filepath = data_dir / filename
    if not filepath.exists():
        print(f"  Warning: {filename} not found")
        return []

    items = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_ground_truth(data_dir: Path, filename: str) -> dict[str, list]:
    """Load ground truth answers, keyed by id."""
    answer_dir = data_dir / "possible_answer"
    filepath = answer_dir / filename

    if not filepath.exists():
        return {}

    answers = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                item_id = item.get("id", "")
                answers[item_id] = item.get("ground_truth", [])
    return answers


def extract_user_question(question_field) -> str:
    """Extract the user question from BFCL's nested format."""
    if isinstance(question_field, str):
        return question_field

    # Handle nested list format: [[{role, content}, ...]]
    if isinstance(question_field, list):
        for turn in question_field:
            if isinstance(turn, list):
                for msg in turn:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return msg.get("content", "")
            elif isinstance(turn, dict) and turn.get("role") == "user":
                return turn.get("content", "")

    return ""


def convert_bfcl_to_task(item: dict, ground_truth: list | None, category: str) -> dict | None:
    """Convert a BFCL item to our task format."""
    try:
        item_id = item.get("id", "")
        question = extract_user_question(item.get("question"))
        functions = item.get("function", [])

        if not question or not functions:
            return None

        # Format function signatures for prompt
        func_docs = []
        for func in functions:
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            param_props = params.get("properties", {})
            required = params.get("required", [])

            param_strs = []
            for pname, pinfo in param_props.items():
                ptype = pinfo.get("type", "any")
                req_marker = " (required)" if pname in required else ""
                param_strs.append(f"  - {pname}: {ptype}{req_marker}")

            func_doc = f"### {name}\n{desc}"
            if param_strs:
                func_doc += "\nParameters:\n" + "\n".join(param_strs)
            func_docs.append(func_doc)

        # Build prompt
        prompt = (
            "Select the correct function and provide arguments as JSON.\n\n"
            "## Available Functions\n\n"
            + "\n\n".join(func_docs)
            + f"\n\n## User Request\n\n{question}\n\n"
            "## Response Format\n\n"
            'Respond with JSON: {"name": "function_name", "arguments": {...}}'
        )

        return {
            "id": f"t4_{item_id}",
            "task_type": "t4",
            "source": "bfcl_v4",
            "category": category,
            "prompt": prompt,
            "schema_path": "tasks/schemas/t4_function_call_schema.json",
            "gold": ground_truth[0] if ground_truth else None,
            "functions": functions,
        }
    except Exception as e:
        print(f"  Warning: Failed to convert {item.get('id', 'unknown')}: {e}")
        return None


CATEGORY_FILES = {
    "simple_python": "BFCL_v4_simple_python.json",
    "simple_java": "BFCL_v4_simple_java.json",
    "simple_js": "BFCL_v4_simple_javascript.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
    "live_simple": "BFCL_v4_live_simple.json",
    "live_multiple": "BFCL_v4_live_multiple.json",
}


def main():
    parser = argparse.ArgumentParser(description="Build T4 BFCL task suite")
    parser.add_argument("--out", default="tasks/t4_bfcl.jsonl", help="Output file")
    parser.add_argument("--count", type=int, default=500, help="Max tasks to include")
    parser.add_argument("--categories", nargs="+", default=["simple_python", "multiple"],
                        choices=list(CATEGORY_FILES.keys()),
                        help="BFCL categories to include")
    args = parser.parse_args()

    print("Building T4 (BFCL v4) task suite...")

    data_dir = get_bfcl_data_dir()
    if not data_dir:
        print("Error: bfcl-eval package not installed. Run: pip install bfcl-eval")
        return

    print(f"  Using data from: {data_dir}")

    tasks = []

    for category in args.categories:
        filename = CATEGORY_FILES.get(category)
        if not filename:
            continue

        print(f"Processing {category} ({filename})...")

        items = load_bfcl_file(data_dir, filename)
        ground_truths = load_ground_truth(data_dir, filename)

        print(f"  Loaded {len(items)} items, {len(ground_truths)} ground truths")

        for item in items:
            if len(tasks) >= args.count:
                break

            item_id = item.get("id", "")
            gt = ground_truths.get(item_id)

            task = convert_bfcl_to_task(item, gt, category)
            if task:
                tasks.append(task)

        if len(tasks) >= args.count:
            break

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    print(f"\nWrote {len(tasks)} tasks to {out_path}")

    # Summary
    by_cat = {}
    for t in tasks:
        cat = t.get("category", "unknown")
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print("By category:", by_cat)

    with_gt = sum(1 for t in tasks if t.get("gold"))
    print(f"With ground truth: {with_gt}/{len(tasks)}")


if __name__ == "__main__":
    main()
