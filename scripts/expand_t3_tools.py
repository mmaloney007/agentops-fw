#!/usr/bin/env python3
"""
Expand T3 tool tasks deterministically to a target count.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def generate_templates(rng: random.Random, count: int) -> List[dict]:
    metrics = [
        "checkout_latency_ms",
        "auth_failures_pct",
        "billing_error_rate",
        "email_queue_depth",
        "search_error_rate",
        "payments_retry_pct",
    ]
    windows = ["last_15m", "last_hour", "last_day"]
    aggregates = ["p50", "p90", "p95", "p99"]
    fields_pool = [
        ["email", "plan_tier"],
        ["region", "active_subscriptions"],
        ["organization", "active_seats"],
        ["support_tier", "plan_tier"],
    ]
    components = [
        "checkout-api",
        "billing-webhook",
        "auth-service",
        "search-service",
        "data-pipeline",
    ]
    focuses = ["root cause", "follow-up actions", "remediation steps", "customer impact"]

    rows: List[dict] = []
    for i in range(count):
        tool_type = rng.choice(["lookup_customer", "fetch_metric", "open_incident", "summarize_report"])
        if tool_type == "lookup_customer":
            cust_id = str(rng.randint(1000, 99999))
            fields = rng.choice(fields_pool)
            ask = f"Get {', '.join(fields)} for customer {cust_id} to decide next steps."
            arguments = {"customer_id": cust_id, "fields": fields}
        elif tool_type == "fetch_metric":
            metric = rng.choice(metrics)
            window = rng.choice(windows)
            aggregate = rng.choice(aggregates)
            ask = f"Fetch {aggregate} {metric} over {window} for the status check."
            arguments = {"metric": metric, "window": window, "aggregate": aggregate}
        elif tool_type == "open_incident":
            severity = rng.choice(["low", "medium", "high", "critical"])
            component = rng.choice(components)
            summary = f"{component} issue detected during monitoring run {rng.randint(100,999)}"
            ask = f"Open an incident for {component} with {severity} severity and note the issue."
            arguments = {"severity": severity, "component": component, "summary": summary}
        else:
            report_id = f"RPT-{rng.randint(1, 999):03d}"
            focus = rng.choice(focuses)
            ask = f"Summarize report {report_id} focusing on {focus} for the exec brief."
            arguments = {"report_id": report_id, "focus": focus}
        rows.append(
            {
                "id": f"t3_generated_{i}",
                "task_type": "t3",
                "prompt": "",
                "schema_path": "tasks/schemas/t3_tool_call_schema.json",
                "gold": {"tool": tool_type, "arguments": arguments},
                "ask": ask,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="tasks/t3_tools.jsonl")
    ap.add_argument("--out", dest="out_path", default="tasks/t3_tools.jsonl")
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not args.no_backup and in_path == out_path:
        bak = in_path.with_suffix(".jsonl.bak")
        if not bak.exists():
            in_path.replace(bak)
            in_path = bak

    base = load_jsonl(in_path)
    rng = random.Random(args.seed)

    # Normalize prompts for existing rows.
    tool_hint = (
        "Choose the single best tool and fill arguments precisely. "
        "Tools: lookup_customer(customer_id, fields[]), "
        "fetch_metric(metric, window=last_15m|last_hour|last_day, aggregate=p50|p90|p95|p99), "
        "open_incident(severity, component, summary), "
        "summarize_report(report_id, focus)."
    )
    for rec in base:
        ask = rec.get("ask") or rec.get("prompt", "")
        rec["prompt"] = f"{tool_hint}\n\nRequest: {ask}"
        rec["task_type"] = rec.get("task_type") or "t3"
        rec["schema_path"] = "tasks/schemas/t3_tool_call_schema.json"

    rows = list(base)
    if len(rows) < args.target:
        extra = generate_templates(rng, args.target - len(rows))
        for rec in extra:
            ask = rec.get("ask", "")
            rec["prompt"] = f"{tool_hint}\n\nRequest: {ask}"
        rows.extend(extra)

    rows = rows[: args.target]
    write_jsonl(out_path, rows)
    print(f"[done] wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
