#!/usr/bin/env python3
"""
Check scored evaluation files against thresholds in criteria.yaml.
Usage:
  python3 tools/check_criteria.py --scored out/aggregate/scored_qwen_gold.csv --criteria criteria.yaml --out out/aggregate/check_qwen_gold.json
"""
import argparse
import json
import os

import pandas as pd
import yaml


def _first_present(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _add_pass_pct(summary: dict, df: pd.DataFrame, col: str, threshold, key: str, op: str = ">="):
    if threshold is None or col not in df.columns:
        return
    series = df[col]
    if op == "<=":
        summary[key] = float((series <= threshold).mean())
    else:
        summary[key] = float((series >= threshold).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--criteria", default="criteria.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.criteria, "r", encoding="utf-8") as f:
        crit = yaml.safe_load(f)
    df = pd.read_csv(args.scored)
    summary = {}

    structure = crit.get("structure", {})
    accuracy = crit.get("accuracy", {})
    faith = crit.get("faithfulness", {})
    slo = crit.get("slo", {})

    # Structure
    json_valid_min = _first_present(structure, ["json_valid_min", "json_valid"], 1)
    schema_valid_min = _first_present(structure, ["schema_valid_min", "schema_valid"], None)
    _add_pass_pct(summary, df, "json_valid", json_valid_min, "json_valid_pass_pct")
    _add_pass_pct(summary, df, "schema_valid", schema_valid_min, "schema_valid_pass_pct")

    # Accuracy (new schema)
    _add_pass_pct(
        summary,
        df,
        "clinc_intent_macro_f1",
        _first_present(accuracy, ["clinc_intent_macro_f1_min"], None),
        "clinc_intent_macro_f1_pass_pct",
    )
    _add_pass_pct(
        summary,
        df,
        "clinc_intent_accuracy",
        _first_present(accuracy, ["clinc_intent_accuracy_min"], None),
        "clinc_intent_accuracy_pass_pct",
    )
    _add_pass_pct(
        summary,
        df,
        "hotpot_answer_exact_match",
        _first_present(accuracy, ["hotpot_answer_exact_match_min"], None),
        "hotpot_answer_exact_match_pass_pct",
    )
    _add_pass_pct(
        summary,
        df,
        "hotpot_answer_f1",
        _first_present(accuracy, ["hotpot_answer_f1_min"], None),
        "hotpot_answer_f1_pass_pct",
    )

    # Legacy columns
    _add_pass_pct(summary, df, "f1", _first_present(structure, ["f1"], None), "f1_pass_pct")
    _add_pass_pct(summary, df, "em", _first_present(structure, ["em"], None), "em_pass_pct")

    # Faithfulness
    _add_pass_pct(
        summary,
        df,
        "hotpot_faithfulness",
        _first_present(faith, ["hotpot_faithfulness_min", "faithfulness"], None),
        "hotpot_faithfulness_pass_pct",
    )
    _add_pass_pct(
        summary,
        df,
        "hotpot_contradiction_rate",
        _first_present(faith, ["hotpot_contradiction_rate_max"], None),
        "hotpot_contradiction_rate_within_max_pct",
        op="<=",
    )

    # Legacy faithfulness fields
    _add_pass_pct(
        summary,
        df,
        "faithfulness",
        _first_present(faith, ["faithfulness"], None),
        "faithfulness_pass_pct",
    )
    _add_pass_pct(
        summary,
        df,
        "cite_acc",
        _first_present(faith, ["cite_acc"], None),
        "cite_acc_pass_pct",
    )

    # SLO metrics
    if "latency_ms" in df.columns:
        summary["p95_ms"] = float(df["latency_ms"].quantile(0.95))
        summary["p99_ms"] = float(df["latency_ms"].quantile(0.99))
        summary["p95_within_slo"] = summary["p95_ms"] <= _first_present(
            slo, ["p95_ms_max", "p95_ms"], float("inf")
        )
        summary["p99_within_slo"] = summary["p99_ms"] <= _first_present(slo, ["p99_ms_max"], float("inf"))
        on_time_budget = _first_present(slo, ["on_time_budget_ms", "hard_timeout_ms"], None)
        if on_time_budget is not None:
            summary["on_time_rate"] = float((df["latency_ms"] <= on_time_budget).mean())

    if "success_at_slo" in df.columns:
        summary["success_at_slo_rate"] = float(df["success_at_slo"].mean())
        min_slo = _first_present(slo, ["success_at_slo_min", "success_at_slo"], None)
        if min_slo is not None:
            summary["success_at_slo_within_min"] = summary["success_at_slo_rate"] >= float(min_slo)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] criteria check -> {args.out}")


if __name__ == "__main__":
    main()
