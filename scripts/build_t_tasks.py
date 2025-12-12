#!/usr/bin/env python3
"""
Build deterministic T1/T2/T3 task suites used in the paper:
 - T1: structured extraction from incident-style text
 - T2: grounded summaries with risk/action fields
 - T3: tool-using episodes with a small catalog

Outputs are JSONL files under tasks/ by default and can be regenerated without
network access. Each record includes a task_type key for scoring.
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = {
    "t1": "tasks/schemas/t1_incident_schema.json",
    "t2": "tasks/schemas/t2_summary_schema.json",
    "t3": "tasks/schemas/t3_tool_call_schema.json",
}

T1_CASES = [
    {
        "id": "t1_incident_latency",
        "text": "PagerDuty alert from checkout-api: p95 latency above 1800ms for 30% of traffic in the last 15m after Redis saturation.",
        "category": "incident",
        "severity": "high",
        "source": "checkout-api",
        "time_window": "last_15m",
        "tags": ["latency", "checkout", "redis"],
    },
    {
        "id": "t1_outage_pipeline",
        "text": "ETL jobs in data-pipeline have failed since midnight; warehouse tables are stale and dashboards show zeros for the last day.",
        "category": "outage",
        "severity": "critical",
        "source": "data-pipeline",
        "time_window": "last_day",
        "tags": ["etl", "warehouse", "freshness"],
    },
    {
        "id": "t1_degradation_search",
        "text": "Search-service reporting increased cache misses and stale results for the last hour; relevance drops and retries climb.",
        "category": "degradation",
        "severity": "medium",
        "source": "search-service",
        "time_window": "last_hour",
        "tags": ["search", "cache", "stale-results"],
    },
    {
        "id": "t1_maintenance_flags",
        "text": "Feature-flags service will be in read-only mode during a migration window spanning the next 48 hours; releases paused.",
        "category": "maintenance",
        "severity": "low",
        "source": "feature-flags",
        "time_window": "multi_day",
        "tags": ["maintenance", "migration", "read-only"],
    },
    {
        "id": "t1_security_token",
        "text": "Auth-service detected a leaked API token used from an unknown IP over the last day; token has been revoked.",
        "category": "security",
        "severity": "high",
        "source": "auth-service",
        "time_window": "last_day",
        "tags": ["token", "revocation", "ip-alert"],
    },
    {
        "id": "t1_degradation_push",
        "text": "Mobile push queue depth has been rising for 40 minutes; delivery is delayed but still flowing.",
        "category": "degradation",
        "severity": "medium",
        "source": "push-service",
        "time_window": "last_hour",
        "tags": ["queueing", "mobile", "delay"],
    },
    {
        "id": "t1_outage_billing",
        "text": "Billing webhooks returning 500s across tenants during the last hour; retries exhausted and invoices stuck.",
        "category": "outage",
        "severity": "critical",
        "source": "billing-webhook",
        "time_window": "last_hour",
        "tags": ["billing", "webhooks", "errors"],
    },
    {
        "id": "t1_incident_capacity",
        "text": "GPU cluster utilization above 90% for multiple days; schedulers throttling new jobs and queue times growing.",
        "category": "incident",
        "severity": "high",
        "source": "gpu-cluster",
        "time_window": "multi_day",
        "tags": ["capacity", "scheduling", "queue"],
    },
    {
        "id": "t1_security_login",
        "text": "Repeated admin login attempts from a new ASN in the last 15m; MFA challenges failing.",
        "category": "security",
        "severity": "medium",
        "source": "admin-console",
        "time_window": "last_15m",
        "tags": ["login", "asn-change", "mfa"],
    },
    {
        "id": "t1_incident_cache",
        "text": "Cache cluster eviction storm for last day after config drift; hit rate dropped and application error rate spiked briefly.",
        "category": "incident",
        "severity": "medium",
        "source": "cache-cluster",
        "time_window": "last_day",
        "tags": ["cache", "eviction", "config-drift"],
    },
]

T2_CASES = [
    {
        "id": "t2_checkout_review",
        "context": "Weekly reliability review: checkout p95 improved to 1.1s after cache tuning. Search errors caused 2% retries yesterday. Cloud spend is up 12% week over week from burst capacity.",
        "short_summary": "Reliability better but search errors and spend are emerging risks.",
        "key_points": [
            "Checkout p95 now 1.1s after cache tuning",
            "Search errors caused 2% retries yesterday",
            "Cloud spend up 12% WoW from burst capacity"
        ],
        "primary_risk": "Search instability and higher cost",
        "recommended_action": "Stabilize search and trim burst capacity before the next peak",
    },
    {
        "id": "t2_postmortem_sync",
        "context": "Postmortem notes: release 482 introduced a bad feature flag default that broke SSO. Impact lasted 35 minutes with 8% auth failures. Rollback restored service. No data loss.",
        "short_summary": "Bad flag default in release 482 caused temporary SSO failures until rollback.",
        "key_points": [
            "Release 482 shipped a bad flag default",
            "SSO auth failures peaked at 8% for 35 minutes",
            "Rollback restored service without data loss"
        ],
        "primary_risk": "Flag safety gaps in rollout process",
        "recommended_action": "Add preflight checks for flag defaults before rollout",
    },
    {
        "id": "t2_finops",
        "context": "FinOps memo: GPU spend climbed 18% this week due to unbounded eval jobs. Batch windows slipped, and reserved instances are underutilized.",
        "short_summary": "Eval jobs pushed GPU spend up 18% and delayed batch windows.",
        "key_points": [
            "GPU spend up 18% from unbounded eval jobs",
            "Batch windows slipped",
            "Reserved instances underutilized"
        ],
        "primary_risk": "Rising cost and missed batch deadlines",
        "recommended_action": "Cap eval jobs and shift to reserved capacity",
    },
    {
        "id": "t2_support_load",
        "context": "Support report: ticket volume up 22% after pricing change; top themes are failed payments and duplicate invoices. CSAT dipped from 4.7 to 4.4.",
        "short_summary": "Pricing change increased support load and hurt CSAT.",
        "key_points": [
            "Ticket volume up 22% after pricing change",
            "Failed payments and duplicate invoices dominate",
            "CSAT dropped from 4.7 to 4.4"
        ],
        "primary_risk": "Customer frustration around billing",
        "recommended_action": "Fix billing defects and publish a help article before next billing cycle",
    },
    {
        "id": "t2_observability",
        "context": "Observability gap: staging lacks trace sampling and log retention beyond 24h. Incident responders could not replay traffic during last week's regression.",
        "short_summary": "Staging observability is too shallow to debug regressions.",
        "key_points": [
            "No trace sampling in staging",
            "Logs retained only 24h",
            "Recent regression lacked replay data"
        ],
        "primary_risk": "Slow incident resolution in staging",
        "recommended_action": "Enable trace sampling and extend log retention in staging",
    },
    {
        "id": "t2_data_quality",
        "context": "Data quality review: pipeline duplicates increased after schema change; BI dashboards showed double revenue for two hours. Backfill fixed totals but alerts were delayed.",
        "short_summary": "Schema change created duplicates and temporarily doubled revenue in dashboards.",
        "key_points": [
            "Schema change increased pipeline duplicates",
            "BI dashboards showed double revenue for two hours",
            "Backfill fixed totals but alerts lagged"
        ],
        "primary_risk": "Trust erosion in BI numbers",
        "recommended_action": "Tighten schema checks and add real-time duplicate alerts",
    },
]

T3_CASES = [
    {
        "id": "t3_lookup_customer_email",
        "ask": "Find the email and plan tier for customer 21888 before replying.",
        "tool": "lookup_customer",
        "arguments": {"customer_id": "21888", "fields": ["email", "plan_tier"]},
    },
    {
        "id": "t3_fetch_checkout_latency",
        "ask": "Retrieve the p95 checkout latency over the last hour to validate the SLO.",
        "tool": "fetch_metric",
        "arguments": {"metric": "checkout_latency_ms", "window": "last_hour", "aggregate": "p95"},
    },
    {
        "id": "t3_open_incident_webhooks",
        "ask": "Open an incident for failing payment webhooks with high severity and mention 500 errors on callbacks.",
        "tool": "open_incident",
        "arguments": {"severity": "high", "component": "payment-webhook", "summary": "500 errors on callbacks"},
    },
    {
        "id": "t3_summarize_report_root_cause",
        "ask": "Summarize report RPT-442 focusing on root cause for the exec brief.",
        "tool": "summarize_report",
        "arguments": {"report_id": "RPT-442", "focus": "root cause"},
    },
    {
        "id": "t3_fetch_auth_errors",
        "ask": "Check auth failure p99 for the last 15 minutes to see if the spike persists.",
        "tool": "fetch_metric",
        "arguments": {"metric": "auth_failures_perc", "window": "last_15m", "aggregate": "p99"},
    },
    {
        "id": "t3_lookup_customer_location",
        "ask": "Pull customer 9001's region and active_subscriptions before proposing a migration.",
        "tool": "lookup_customer",
        "arguments": {"customer_id": "9001", "fields": ["region", "active_subscriptions"]},
    },
    {
        "id": "t3_open_incident_cache",
        "ask": "File a critical incident for cache cluster resets affecting config-service.",
        "tool": "open_incident",
        "arguments": {"severity": "critical", "component": "config-service", "summary": "cache cluster resets"},
    },
    {
        "id": "t3_summarize_report_actions",
        "ask": "Summarize report REP-009 with a focus on follow-up actions.",
        "tool": "summarize_report",
        "arguments": {"report_id": "REP-009", "focus": "follow-up actions"},
    },
    {
        "id": "t3_fetch_signup_rate",
        "ask": "Retrieve p90 signup rate over the last day for the growth dashboard.",
        "tool": "fetch_metric",
        "arguments": {"metric": "signup_rate_pct", "window": "last_day", "aggregate": "p90"},
    },
    {
        "id": "t3_lookup_customer_plan",
        "ask": "Check customer 404's current plan and support tier before escalating.",
        "tool": "lookup_customer",
        "arguments": {"customer_id": "404", "fields": ["plan_tier", "support_tier"]},
    },
    {
        "id": "t3_open_incident_db",
        "ask": "File a high severity incident for db-write failures on orders-service.",
        "tool": "open_incident",
        "arguments": {"severity": "high", "component": "orders-db", "summary": "write failures on primary"},
    },
    {
        "id": "t3_fetch_queue_depth",
        "ask": "Fetch queue depth p95 over the last 15 minutes for email-delivery.",
        "tool": "fetch_metric",
        "arguments": {"metric": "email_queue_depth", "window": "last_15m", "aggregate": "p95"},
    },
    {
        "id": "t3_lookup_customer_org",
        "ask": "Get organization and active seats for customer 1337.",
        "tool": "lookup_customer",
        "arguments": {"customer_id": "1337", "fields": ["organization", "active_seats"]},
    },
    {
        "id": "t3_summarize_report_incident",
        "ask": "Summarize incident report IR-220 focusing on remediation steps.",
        "tool": "summarize_report",
        "arguments": {"report_id": "IR-220", "focus": "remediation steps"},
    },
    {
        "id": "t3_open_incident_latency",
        "ask": "Open a critical incident for API latency p99 breaching SLO for auth-service.",
        "tool": "open_incident",
        "arguments": {"severity": "critical", "component": "auth-service", "summary": "API latency p99 breach"},
    },
    {
        "id": "t3_fetch_billing_errors",
        "ask": "Pull p95 billing error rate over the last hour.",
        "tool": "fetch_metric",
        "arguments": {"metric": "billing_error_rate", "window": "last_hour", "aggregate": "p95"},
    },
]


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {path} ({len(rows)} records)")


def _build_t1(seed: int, limit: int | None):
    rng = random.Random(seed)
    cases = list(T1_CASES)
    rng.shuffle(cases)
    if limit:
        cases = cases[:limit]
    rows = []
    for c in cases:
        prompt = (
            "Given the incident text, return JSON with fields: "
            "category (incident/outage/degradation/maintenance/security), "
            "severity (low/medium/high/critical), source (string), "
            "time_window (last_15m/last_hour/last_day/multi_day), and tags (1-5 short strings). "
            "Do not invent fields or values.\n\n"
            f"Incident: {c['text']}"
        )
        rows.append(
            {
                "id": c["id"],
                "task_type": "t1",
                "prompt": prompt,
                "schema_path": SCHEMAS["t1"],
                "gold": {
                    "category": c["category"],
                    "severity": c["severity"],
                    "source": c["source"],
                    "time_window": c["time_window"],
                    "tags": c["tags"],
                },
            }
        )
    return rows


def _build_t2(seed: int, limit: int | None):
    rng = random.Random(seed)
    cases = list(T2_CASES)
    rng.shuffle(cases)
    if limit:
        cases = cases[:limit]
    rows = []
    for c in cases:
        prompt = (
            "You are given a context note. Return JSON with short_summary (one sentence), "
            "key_points (3-8 concise bullets), primary_risk (string), and recommended_action (string). "
            "Stay faithful to the context; do not invent metrics.\n\n"
            f"Context:\n{c['context']}"
        )
        rows.append(
            {
                "id": c["id"],
                "task_type": "t2",
                "prompt": prompt,
                "schema_path": SCHEMAS["t2"],
                "gold": {
                    "short_summary": c["short_summary"],
                    "key_points": c["key_points"],
                    "primary_risk": c["primary_risk"],
                    "recommended_action": c["recommended_action"],
                },
            }
        )
    return rows


def _build_t3(seed: int, limit: int | None):
    rng = random.Random(seed)
    cases = list(T3_CASES)
    rng.shuffle(cases)
    if limit:
        cases = cases[:limit]
    rows = []
    tool_hint = (
        "Choose the single best tool and fill arguments precisely. "
        "Tools: lookup_customer(customer_id, fields[]), "
        "fetch_metric(metric, window=last_15m|last_hour|last_day, aggregate=p50|p90|p95|p99), "
        "open_incident(severity, component, summary), "
        "summarize_report(report_id, focus)."
    )
    for c in cases:
        prompt = f"{tool_hint}\n\nRequest: {c['ask']}"
        rows.append(
            {
                "id": c["id"],
                "task_type": "t3",
                "prompt": prompt,
                "schema_path": SCHEMAS["t3"],
                "gold": {"tool": c["tool"], "arguments": c["arguments"]},
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="tasks", help="Directory for JSONL outputs.")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--t1-count", type=int, default=0, help="0 means all T1 cases")
    ap.add_argument("--t2-count", type=int, default=0, help="0 means all T2 cases")
    ap.add_argument("--t3-count", type=int, default=0, help="0 means all T3 cases")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    t1 = _build_t1(args.seed, args.t1_count or None)
    t2 = _build_t2(args.seed, args.t2_count or None)
    t3 = _build_t3(args.seed, args.t3_count or None)

    _write_jsonl(out_dir / "t1_structured.jsonl", t1)
    _write_jsonl(out_dir / "t2_grounded.jsonl", t2)
    _write_jsonl(out_dir / "t3_tools.jsonl", t3)


if __name__ == "__main__":
    main()
