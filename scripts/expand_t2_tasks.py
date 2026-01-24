#!/usr/bin/env python3
"""
Generate additional T2 (summary) task examples for Paper 2 training.

T2 is a context summarization task - given an operational context note, produce:
- short_summary (one sentence)
- key_points (3-8 bullets)
- primary_risk (string)
- recommended_action (string)

This script generates variations by:
1. Using templates for common operational scenarios
2. Creating realistic context notes with different metrics/issues
3. Producing ground truth summaries

Usage:
    python scripts/expand_t2_tasks.py --output tasks/t2_expanded.jsonl --count 100
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Templates for context generation
SCENARIO_TEMPLATES = [
    # Performance scenarios
    {
        "template": "{service} latency p{percentile} increased to {latency}ms after {event}. {impact}. {action_taken}.",
        "fields": {
            "service": ["API gateway", "checkout-service", "auth-service", "search-api", "recommendation-engine", "payment-processor", "notification-service", "inventory-api"],
            "percentile": ["50", "90", "95", "99"],
            "latency": ["150", "250", "500", "800", "1200", "1800", "2500"],
            "event": ["code deployment", "config change", "traffic spike", "database migration", "cache invalidation", "dependency update", "feature flag rollout"],
            "impact": ["Error rate spiked to 2%", "User complaints increased", "Conversion dropped 5%", "Timeout errors doubled", "Cart abandonment increased"],
            "action_taken": ["Team is investigating", "Rollback in progress", "Scaling up instances", "Cache warmed up manually"],
        },
        "summary_prefix": "Latency degradation in",
        "risk_focus": "performance impact",
        "action_focus": "investigate root cause"
    },
    # Cost/FinOps scenarios
    {
        "template": "{resource} costs increased {percentage}% over {timeframe} due to {cause}. {detail}.",
        "fields": {
            "resource": ["Compute", "GPU cluster", "Database", "Storage", "CDN bandwidth", "API calls", "ML inference", "Data transfer"],
            "percentage": ["12", "18", "25", "35", "50", "75"],
            "timeframe": ["the past week", "this month", "since the feature launch", "after scaling up"],
            "cause": ["unbounded batch jobs", "cache miss storm", "inefficient queries", "log volume spike", "unused reserved instances", "burst capacity usage"],
            "detail": ["Budget alerts triggered", "Approaching quarterly limit", "Cost anomaly detected", "Finance flagged for review"],
        },
        "summary_prefix": "Cost escalation in",
        "risk_focus": "budget overrun",
        "action_focus": "optimize resource usage"
    },
    # Incident/Outage scenarios
    {
        "template": "Incident: {component} experienced {issue} affecting {scope}. Duration: {duration}. {outcome}.",
        "fields": {
            "component": ["Database primary", "Load balancer", "Auth provider", "Payment gateway", "CDN edge", "Message queue", "Search cluster", "Redis cache"],
            "issue": ["connection failures", "timeout spikes", "partial outage", "data inconsistency", "replication lag", "certificate expiry"],
            "scope": ["15% of users", "EU region", "mobile clients", "enterprise tier", "all checkout requests", "new user signups"],
            "duration": ["12 minutes", "35 minutes", "2 hours", "45 minutes", "90 minutes"],
            "outcome": ["Service restored", "Failover successful", "Manual intervention required", "Auto-recovery triggered"],
        },
        "summary_prefix": "Service disruption in",
        "risk_focus": "customer impact",
        "action_focus": "prevent recurrence"
    },
    # Security scenarios
    {
        "template": "Security alert: {finding} detected in {location}. {severity}. {status}.",
        "fields": {
            "finding": ["Unusual login patterns", "Elevated API errors from single IP", "Outdated dependencies with CVEs", "Misconfigured secrets", "Unauthorized access attempt", "Credential leak in logs"],
            "location": ["auth-service", "admin portal", "CI/CD pipeline", "staging environment", "third-party integration", "customer data export"],
            "severity": ["High priority", "Medium risk", "Requires immediate attention", "Flagged for security review"],
            "status": ["Under investigation", "Access revoked", "Patch applied", "Monitoring enhanced"],
        },
        "summary_prefix": "Security concern identified in",
        "risk_focus": "data exposure",
        "action_focus": "strengthen security posture"
    },
    # Data/Pipeline scenarios
    {
        "template": "Data pipeline: {pipeline} {issue}. {impact}. {metrics}.",
        "fields": {
            "pipeline": ["ETL daily sync", "Event streaming", "ML feature pipeline", "Analytics aggregation", "Log ingestion", "Customer export", "Recommendation update"],
            "issue": ["failed at transformation step", "experiencing 6-hour lag", "produced duplicate records", "missed SLA window", "dropped 3% of events"],
            "impact": ["Dashboards showing stale data", "ML predictions degraded", "Reports delayed", "Alerting gaps observed"],
            "metrics": ["Backfill queued", "Recovery ETA 2 hours", "Manual verification needed", "Partial data available"],
        },
        "summary_prefix": "Data integrity issue in",
        "risk_focus": "decision-making accuracy",
        "action_focus": "restore data freshness"
    },
    # Capacity/Scaling scenarios
    {
        "template": "Capacity alert: {resource} at {utilization}% utilization. {trend}. {constraint}.",
        "fields": {
            "resource": ["Database connections", "Memory pool", "CPU cluster", "Disk IOPS", "Network bandwidth", "Worker threads", "Queue depth"],
            "utilization": ["85", "90", "92", "95", "98"],
            "trend": ["Trending up 5% daily", "Spike during peak hours", "Gradual increase over 2 weeks", "Sudden jump after deployment"],
            "constraint": ["Approaching hard limit", "Auto-scaling delayed", "Reserved capacity exhausted", "Throttling imminent"],
        },
        "summary_prefix": "Capacity pressure on",
        "risk_focus": "service degradation",
        "action_focus": "scale proactively"
    },
    # Release/Deployment scenarios
    {
        "template": "Release {version}: {status}. {observation}. {metric_change}.",
        "fields": {
            "version": ["4.2.1", "v23.1", "2024.01", "hotfix-892", "canary-3"],
            "status": ["Deployed to 10% of traffic", "Rolled back after errors", "Fully deployed", "Paused for validation"],
            "observation": ["New feature adoption at 8%", "Error rate baseline unchanged", "Memory footprint increased 15%", "Latency improved 20ms"],
            "metric_change": ["Conversion steady", "No anomalies detected", "Support tickets unchanged", "Performance within SLO"],
        },
        "summary_prefix": "Release status for",
        "risk_focus": "stability",
        "action_focus": "monitor and validate"
    },
]


def generate_context(template_data: Dict, rng: random.Random) -> str:
    """Generate a context string from template."""
    template = template_data["template"]
    fields = template_data["fields"]

    # Fill in the template
    values = {}
    for field, options in fields.items():
        values[field] = rng.choice(options)

    return template.format(**values)


def generate_gold(context: str, template_data: Dict, rng: random.Random) -> Dict[str, Any]:
    """Generate ground truth summary from context."""
    # Extract key information from context
    sentences = [s.strip() for s in context.replace(". ", ".|").split("|") if s.strip()]

    # Generate key points (3-5 bullets)
    key_points = []
    for sent in sentences[:min(len(sentences), rng.randint(3, 5))]:
        # Simplify sentence for bullet
        if len(sent) > 60:
            sent = sent[:57] + "..."
        key_points.append(sent)

    # Generate summary (one sentence)
    short_summary = f"{template_data['summary_prefix']} detected, requires attention."
    if "latency" in context.lower():
        short_summary = "Latency issues affecting service performance and user experience."
    elif "cost" in context.lower():
        short_summary = "Cost escalation identified requiring optimization."
    elif "incident" in context.lower() or "outage" in context.lower():
        short_summary = "Service disruption impacted users, recovery actions taken."
    elif "security" in context.lower():
        short_summary = "Security concern requiring investigation and remediation."
    elif "data" in context.lower() or "pipeline" in context.lower():
        short_summary = "Data pipeline issue affecting downstream systems."
    elif "capacity" in context.lower():
        short_summary = "Capacity constraints approaching critical thresholds."
    elif "release" in context.lower():
        short_summary = "Release requires monitoring and validation."

    # Generate risk and action
    primary_risk = template_data["risk_focus"].capitalize()
    recommended_action = template_data["action_focus"].capitalize()

    return {
        "short_summary": short_summary,
        "key_points": key_points,
        "primary_risk": primary_risk,
        "recommended_action": recommended_action,
    }


def generate_task(task_id: str, context: str, gold: Dict) -> Dict:
    """Generate a complete T2 task record."""
    prompt = (
        "You are given a context note. Return JSON with short_summary (one sentence), "
        "key_points (3-8 concise bullets), primary_risk (string), and recommended_action (string). "
        "Stay faithful to the context; do not invent metrics.\n\n"
        f"Context:\n{context}"
    )

    return {
        "id": task_id,
        "task_type": "t2",
        "prompt": prompt,
        "schema_path": "tasks/schemas/t2_summary_schema.json",
        "gold": gold,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate additional T2 tasks')
    parser.add_argument('--output', '-o', type=str, default='tasks/t2_expanded.jsonl',
                        help='Output file path')
    parser.add_argument('--count', '-n', type=int, default=100,
                        help='Number of tasks to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--include-original', action='store_true',
                        help='Include original T2 tasks at the beginning')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tasks = []

    # Optionally include original tasks
    if args.include_original:
        original_path = Path('tasks/t2_grounded.jsonl')
        if original_path.exists():
            with open(original_path, 'r') as f:
                for line in f:
                    if line.strip():
                        tasks.append(json.loads(line))
            print(f"Loaded {len(tasks)} original T2 tasks")

    # Generate new tasks
    print(f"Generating {args.count} new T2 tasks...")

    for i in range(args.count):
        # Pick a random template
        template_data = rng.choice(SCENARIO_TEMPLATES)

        # Generate context and gold
        context = generate_context(template_data, rng)
        gold = generate_gold(context, template_data, rng)

        # Create task
        task_id = f"t2_gen_{i:04d}"
        task = generate_task(task_id, context, gold)
        tasks.append(task)

    # Shuffle (keep original at front if included)
    if args.include_original:
        original_count = 6  # Known original count
        generated = tasks[original_count:]
        rng.shuffle(generated)
        tasks = tasks[:original_count] + generated
    else:
        rng.shuffle(tasks)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')

    print(f"\nWrote {len(tasks)} T2 tasks to {output_path}")

    # Print sample
    print("\nSample generated task:")
    sample = tasks[-1]
    print(f"  ID: {sample['id']}")
    print(f"  Context: {sample['prompt'].split('Context:')[1][:100]}...")
    print(f"  Summary: {sample['gold']['short_summary']}")
    print(f"  Key points: {len(sample['gold']['key_points'])} bullets")

    return 0


if __name__ == '__main__':
    exit(main())
