#!/usr/bin/env python3
"""
Expand T1 (incident classification) and T2 (grounded summarization) datasets.

T1: Expands from 10 to 100 samples with diverse combinations of:
    - 5 categories × 4 severities × 4 time_windows = 80 combinations
    - 12 services, varied incident descriptions, edge cases

T2: Already expanded (106 samples) - validates and optionally regenerates.

Usage:
    python scripts/expand_t1_t2_samples.py --output-dir tasks/
    python scripts/expand_t1_t2_samples.py --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List

# T1 Constants
CATEGORIES = ["incident", "outage", "degradation", "maintenance", "security"]
SEVERITIES = ["low", "medium", "high", "critical"]
TIME_WINDOWS = ["last_15m", "last_hour", "last_day", "multi_day"]

SERVICES = [
    "checkout-api",
    "auth-service",
    "payment-webhook",
    "search-service",
    "notification-service",
    "data-pipeline",
    "cache-cluster",
    "billing-service",
    "user-service",
    "inventory-api",
    "order-service",
    "analytics-pipeline",
]

# Incident templates by category
T1_TEMPLATES = {
    "incident": [
        "{service} reporting {metric} above threshold for {duration}; {impact}.",
        "PagerDuty alert from {service}: {metric} for {percent}% of traffic in the {time_window} after {cause}.",
        "{service} experienced {issue} affecting {scope}; {status}.",
        "Anomaly detected in {service}: {metric} spiked {direction}; {response}.",
    ],
    "outage": [
        "{service} has been failing since {time_ref}; {downstream} and {user_impact}.",
        "Complete {service} unavailability for {duration}; {error_type} and {recovery_status}.",
        "{service} returning {error_codes} across {scope}; {retry_status}.",
        "Total service disruption in {service}: {root_cause}; {mitigation}.",
    ],
    "degradation": [
        "{service} reporting increased {metric} for the {time_window}; {user_experience}.",
        "{metric} in {service} has been {trend} for {duration}; {operational_status}.",
        "{service} experiencing {issue_type}; {flow_status} but {quality_impact}.",
        "Partial degradation in {service}: {symptom}; {workaround}.",
    ],
    "maintenance": [
        "{service} will be in {mode} during a {operation} window spanning {duration}; {feature_impact}.",
        "Scheduled {operation} for {service}: {timeline}; {user_notice}.",
        "{service} entering {mode} for {operation}; {service_impact}.",
        "Planned {operation} on {service}: {window}; {coordination_status}.",
    ],
    "security": [
        "{service} detected {threat_type} from {source} over the {time_window}; {response}.",
        "Security incident in {service}: {finding}; {mitigation_status}.",
        "{threat_indicator} detected in {service} {time_ref}; {action_taken}.",
        "Auth anomaly in {service}: {pattern}; {security_response}.",
    ],
}

# Fill-in values for templates
METRICS = [
    "p95 latency above 1800ms",
    "error rate at 5%",
    "CPU utilization at 95%",
    "memory pressure at 90%",
    "queue depth growing",
    "connection pool exhausted",
    "cache hit rate below 50%",
    "throughput dropped 30%",
]

CAUSES = [
    "Redis saturation",
    "config drift",
    "deployment rollout",
    "traffic spike",
    "database connection storm",
    "certificate expiry",
    "DNS propagation delay",
    "upstream provider issues",
]

IMPACTS = [
    "users experiencing slow responses",
    "checkout failures increasing",
    "search results stale",
    "notifications delayed",
    "payments timing out",
    "inventory sync broken",
    "analytics gaps forming",
    "login failures rising",
]

TAGS_BY_CATEGORY = {
    "incident": [
        ["latency", "performance", "timeout"],
        ["error-rate", "failures", "retries"],
        ["capacity", "scaling", "resources"],
        ["connectivity", "network", "dns"],
        ["cache", "hit-rate", "eviction"],
    ],
    "outage": [
        ["downtime", "unavailable", "critical"],
        ["data-loss-risk", "recovery", "backup"],
        ["errors", "5xx", "failures"],
        ["failover", "redundancy", "disaster-recovery"],
    ],
    "degradation": [
        ["slow", "latency", "queue"],
        ["partial", "degraded", "quality"],
        ["retries", "timeouts", "backpressure"],
        ["stale-data", "freshness", "sync"],
    ],
    "maintenance": [
        ["planned", "window", "scheduled"],
        ["migration", "upgrade", "patching"],
        ["read-only", "paused", "limited"],
        ["coordination", "change-management", "notice"],
    ],
    "security": [
        ["token", "credential", "auth"],
        ["anomaly", "suspicious", "investigation"],
        ["revocation", "rotation", "access"],
        ["ip-block", "rate-limit", "firewall"],
    ],
}

T1_PROMPT_TEMPLATE = """Given the incident text, return JSON with fields: category (incident/outage/degradation/maintenance/security), severity (low/medium/high/critical), source (string), time_window (last_15m/last_hour/last_day/multi_day), and tags (1-5 short strings). Do not invent fields or values.

Incident: {incident_text}"""


def generate_t1_incident(
    category: str,
    severity: str,
    time_window: str,
    service: str,
    seed: int,
) -> Dict[str, Any]:
    """Generate a single T1 incident sample."""
    rng = random.Random(seed)

    # Select template and fill it
    template = rng.choice(T1_TEMPLATES[category])

    # Create fill values based on category
    fill = {
        "service": service,
        "metric": rng.choice(METRICS),
        "cause": rng.choice(CAUSES),
        "impact": rng.choice(IMPACTS),
        "time_window": time_window.replace("_", " "),
        "duration": rng.choice(["15 minutes", "30 minutes", "1 hour", "several hours", "2 days"]),
        "percent": rng.choice([5, 10, 15, 20, 30]),
        "scope": rng.choice(["all requests", "EU region", "enterprise tier", "new users", "checkout flow"]),
        "status": rng.choice(["team investigating", "mitigation in progress", "monitoring elevated", "rollback initiated"]),
        "direction": rng.choice(["sharply", "gradually", "unexpectedly"]),
        "response": rng.choice(["alerts triggered", "on-call engaged", "auto-scaling activated"]),
        "time_ref": rng.choice(["midnight", "this morning", "2 hours ago", "since deployment"]),
        "downstream": rng.choice(["dashboards stale", "dependent services affected", "queues backing up"]),
        "user_impact": rng.choice(["users cannot access", "errors visible", "timeouts increasing"]),
        "error_type": rng.choice(["500s across the board", "connection refused", "timeout errors"]),
        "error_codes": rng.choice(["500s", "503s", "connection timeouts", "SSL errors"]),
        "recovery_status": rng.choice(["recovery in progress", "failover activated", "manual intervention needed"]),
        "root_cause": rng.choice(["database primary down", "network partition", "storage failure"]),
        "mitigation": rng.choice(["failover in progress", "restarting instances", "rolling back"]),
        "trend": rng.choice(["rising", "degrading", "fluctuating"]),
        "operational_status": rng.choice(["still flowing", "partially operational", "service degraded"]),
        "issue_type": rng.choice(["intermittent failures", "elevated latency", "partial unavailability"]),
        "flow_status": rng.choice(["traffic still flowing", "requests completing", "core function working"]),
        "quality_impact": rng.choice(["quality degraded", "some failures", "retries elevated"]),
        "symptom": rng.choice(["cache misses up", "stale results", "slow queries"]),
        "workaround": rng.choice(["caching helps", "retries succeed", "fallback enabled"]),
        "mode": rng.choice(["read-only mode", "maintenance mode", "limited functionality"]),
        "operation": rng.choice(["migration", "upgrade", "patching", "schema change", "cert rotation"]),
        "feature_impact": rng.choice(["releases paused", "writes disabled", "new signups blocked"]),
        "timeline": rng.choice(["next 2 hours", "overnight", "weekend window"]),
        "user_notice": rng.choice(["users notified", "banner displayed", "email sent"]),
        "service_impact": rng.choice(["expect brief interruptions", "reduced capacity", "delayed processing"]),
        "window": rng.choice(["Saturday 2-6am UTC", "next maintenance window", "rolling updates"]),
        "coordination_status": rng.choice(["stakeholders informed", "change approved", "teams coordinated"]),
        "threat_type": rng.choice(["leaked API token", "credential stuffing", "unusual access pattern"]),
        "source": rng.choice(["unknown IP", "new ASN", "suspicious region", "compromised account"]),
        "finding": rng.choice(["token exposed in logs", "elevated privileges detected", "unauthorized access"]),
        "mitigation_status": rng.choice(["token revoked", "access blocked", "investigation ongoing"]),
        "threat_indicator": rng.choice(["Suspicious login attempts", "API abuse detected", "Credential leak"]),
        "action_taken": rng.choice(["MFA enforced", "session invalidated", "IP blocked"]),
        "pattern": rng.choice(["repeated failures from new ASN", "unusual access times", "privilege escalation attempt"]),
        "security_response": rng.choice(["security team engaged", "monitoring enhanced", "audit triggered"]),
    }

    # Generate incident text
    try:
        incident_text = template.format(**fill)
    except KeyError:
        # Fallback to a simpler format
        incident_text = f"{service} experiencing {category}-level issues with {severity} impact over {time_window.replace('_', ' ')}."

    # Select tags
    tags = rng.choice(TAGS_BY_CATEGORY[category])[:3]
    # Add service-related tag
    service_tag = service.split("-")[0]
    if service_tag not in tags:
        tags = tags[:2] + [service_tag]

    # Create the sample
    sample = {
        "id": f"t1_gen_{category}_{severity}_{time_window}_{seed:04d}",
        "task_type": "t1",
        "prompt": T1_PROMPT_TEMPLATE.format(incident_text=incident_text),
        "schema_path": "tasks/schemas/t1_incident_schema.json",
        "gold": {
            "category": category,
            "severity": severity,
            "source": service,
            "time_window": time_window,
            "tags": tags,
        },
    }

    return sample


def generate_t1_expanded(target_count: int = 100, base_seed: int = 42) -> List[Dict[str, Any]]:
    """Generate expanded T1 dataset with diverse combinations."""
    samples = []
    rng = random.Random(base_seed)

    # Generate systematic combinations first (80 combinations)
    combo_idx = 0
    for category in CATEGORIES:
        for severity in SEVERITIES:
            for time_window in TIME_WINDOWS:
                service = SERVICES[combo_idx % len(SERVICES)]
                seed = base_seed + combo_idx
                sample = generate_t1_incident(category, severity, time_window, service, seed)
                samples.append(sample)
                combo_idx += 1

    # Add edge cases and variations to reach target_count
    while len(samples) < target_count:
        category = rng.choice(CATEGORIES)
        severity = rng.choice(SEVERITIES)
        time_window = rng.choice(TIME_WINDOWS)
        service = rng.choice(SERVICES)
        seed = base_seed + len(samples) + 1000
        sample = generate_t1_incident(category, severity, time_window, service, seed)
        # Make ID unique
        sample["id"] = f"t1_edge_{len(samples):04d}"
        samples.append(sample)

    return samples[:target_count]


# T2 Constants - Already have 106 samples, but can validate/regenerate
T2_SCENARIO_TYPES = [
    "reliability_review",
    "postmortem",
    "finops_report",
    "security_audit",
    "capacity_planning",
    "incident_retrospective",
    "performance_review",
    "deployment_analysis",
    "customer_impact",
    "sla_review",
]


def validate_jsonl(path: Path, schema_path: str) -> tuple[int, List[str]]:
    """Validate JSONL file against expected schema path."""
    errors = []
    count = 0

    if not path.exists():
        return 0, [f"File not found: {path}"]

    with open(path, "r") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                count += 1
                if record.get("schema_path") != schema_path:
                    errors.append(f"Line {i}: wrong schema_path")
                if "gold" not in record:
                    errors.append(f"Line {i}: missing gold")
                if "prompt" not in record:
                    errors.append(f"Line {i}: missing prompt")
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON: {e}")

    return count, errors


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Expand T1 and T2 task datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tasks"),
        help="Output directory for expanded files",
    )
    parser.add_argument(
        "--t1-count",
        type=int,
        default=100,
        help="Target T1 sample count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing files, don't generate",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t1_path = output_dir / "t1_expanded.jsonl"
    t2_path = output_dir / "t2_expanded.jsonl"

    # Validate existing files
    print("=== Validating existing task files ===")

    t1_count, t1_errors = validate_jsonl(t1_path, "tasks/schemas/t1_incident_schema.json")
    print(f"T1 ({t1_path}): {t1_count} samples, {len(t1_errors)} errors")
    for err in t1_errors[:5]:
        print(f"  - {err}")

    t2_count, t2_errors = validate_jsonl(t2_path, "tasks/schemas/t2_summary_schema.json")
    print(f"T2 ({t2_path}): {t2_count} samples, {len(t2_errors)} errors")
    for err in t2_errors[:5]:
        print(f"  - {err}")

    if args.validate_only:
        print("\n=== Validation complete ===")
        return

    # Generate T1 if needed
    if t1_count < args.t1_count or args.force:
        print(f"\n=== Generating T1 expanded dataset ({args.t1_count} samples) ===")
        t1_samples = generate_t1_expanded(args.t1_count, args.seed)

        with open(t1_path, "w") as f:
            for sample in t1_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"Wrote {len(t1_samples)} T1 samples to {t1_path}")
        print(f"Hash: {compute_file_hash(t1_path)}")
    else:
        print(f"\nT1 already has {t1_count} samples (target: {args.t1_count}), skipping")

    # T2 is already expanded (106 samples), just report
    if t2_count >= 100:
        print(f"\nT2 already has {t2_count} samples (≥100), no expansion needed")
    else:
        print(f"\nWARNING: T2 has only {t2_count} samples, may need manual expansion")

    print("\n=== Summary ===")
    print(f"T1: {t1_path} ({t1_count if t1_count >= args.t1_count else args.t1_count} samples)")
    print(f"T2: {t2_path} ({t2_count} samples)")


if __name__ == "__main__":
    main()
