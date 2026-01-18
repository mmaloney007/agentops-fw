#!/usr/bin/env python3
"""
Test all configs with LLM-enhanced Data Doctor.

This script:
1. Validates all config files load correctly
2. Runs full Data Doctor analysis on configs with local data
3. Tests both with and without LLM enhancement

Usage:
    pixi run -e dev python scripts/test_configs_with_llm.py
    pixi run -e dev python scripts/test_configs_with_llm.py --llm  # Enable LLM

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import dask.dataframe as dd


def load_config_without_env_check(config_path: Path) -> dict:
    """Load config YAML without environment validation."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_all_configs(configs_dir: Path) -> dict:
    """Validate all config files can be parsed."""
    print("\n" + "=" * 70)
    print("  PHASE 1: Config Validation (Dry Run)")
    print("=" * 70)

    results = {"passed": [], "failed": []}

    for config_file in sorted(configs_dir.glob("*.yaml")):
        try:
            config = load_config_without_env_check(config_file)

            # Check for required sections
            required = ["input", "output"]
            missing = [r for r in required if r not in config]

            if missing:
                results["failed"].append((config_file.name, f"Missing sections: {missing}"))
                print(f"  FAIL  {config_file.name}: Missing {missing}")
            else:
                # Check LLM settings
                ids_llm = config.get("ids", {}).get("detection", {}).get("llm_enabled", False)
                dd_llm = config.get("data_doctor", {}).get("llm_enabled", False)
                source = config.get("input", {}).get("source", "unknown")

                results["passed"].append(config_file.name)
                print(f"  OK    {config_file.name}")
                print(f"        source={source}, ids_llm={ids_llm}, doctor_llm={dd_llm}")

        except Exception as e:
            results["failed"].append((config_file.name, str(e)))
            print(f"  FAIL  {config_file.name}: {e}")

    print(f"\n  Summary: {len(results['passed'])} passed, {len(results['failed'])} failed")
    return results


def run_data_doctor_test(config_path: Path, enable_llm: bool = False) -> dict:
    """Run Data Doctor on a config with local data."""
    from neuralift_c360_prep.data_doctor import analyze_data, build_minimal_data_dict

    config = load_config_without_env_check(config_path)
    input_cfg = config.get("input", {})

    # Get parquet path
    parquet_path = input_cfg.get("parquet_path")
    if not parquet_path:
        return {"status": "skipped", "reason": "No local parquet_path"}

    # Resolve path relative to project root
    project_root = config_path.parent.parent
    full_path = project_root / parquet_path

    if not full_path.exists():
        return {"status": "skipped", "reason": f"File not found: {full_path}"}

    print(f"\n  Loading data from: {full_path}")

    # Read data
    ddf = dd.read_parquet(str(full_path))
    print(f"  Columns: {len(ddf.columns)}, Partitions: {ddf.npartitions}")

    # Build data_dict using the helper
    data_dict = build_minimal_data_dict(ddf)

    print(f"  LLM Enabled: {enable_llm}")

    # Run Data Doctor
    report = analyze_data(
        ddf,
        data_dict,
        use_llm=enable_llm,
        llm_model="gpt-4o-mini",
        show_progress=True,
    )

    # Count all suggestions
    total_suggestions = (
        len(report.feature_suggestions)
        + len(report.fill_suggestions)
        + len(report.kpi_candidates)
        + len(report.ratio_opportunities)
    )

    return {
        "status": "success",
        "suggestions_count": total_suggestions,
        "feature_suggestions": len(report.feature_suggestions),
        "fill_suggestions": len(report.fill_suggestions),
        "kpi_candidates": len(report.kpi_candidates),
        "ratio_opportunities": len(report.ratio_opportunities),
        "llm_suggestions_count": len(report._llm_suggestions) if report._llm_suggestions else 0,
        "executive_summary": report._llm_executive_summary is not None,
    }


def run_full_pipeline_tests(configs_dir: Path, enable_llm: bool = False) -> dict:
    """Run full Data Doctor tests on configs with local data."""
    print("\n" + "=" * 70)
    print(f"  PHASE 2: Full Data Doctor Pipeline (LLM={'ON' if enable_llm else 'OFF'})")
    print("=" * 70)

    # Configs known to have local data
    local_configs = [
        "data_prep.yaml",
        "wine_and_cheese_local_full.yaml",
    ]

    results = {"passed": [], "failed": [], "skipped": []}

    for config_name in local_configs:
        config_path = configs_dir / config_name
        if not config_path.exists():
            results["skipped"].append((config_name, "File not found"))
            continue

        print(f"\n  Testing: {config_name}")
        print("  " + "-" * 50)

        try:
            result = run_data_doctor_test(config_path, enable_llm=enable_llm)

            if result["status"] == "skipped":
                results["skipped"].append((config_name, result["reason"]))
                print(f"  SKIP  {result['reason']}")
            else:
                results["passed"].append(config_name)
                print(f"  OK    Total Suggestions: {result['suggestions_count']}")
                print(f"        - Feature: {result['feature_suggestions']}")
                print(f"        - Fill: {result['fill_suggestions']}")
                print(f"        - KPI Candidates: {result['kpi_candidates']}")
                print(f"        - Ratio Opportunities: {result['ratio_opportunities']}")
                print(f"        LLM Suggestions: {result['llm_suggestions_count']}")
                print(f"        Executive Summary: {result['executive_summary']}")

        except Exception as e:
            results["failed"].append((config_name, str(e)))
            print(f"  FAIL  {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Summary: {len(results['passed'])} passed, {len(results['failed'])} failed, {len(results['skipped'])} skipped")
    return results


def main():
    parser = argparse.ArgumentParser(description="Test configs with Data Doctor")
    parser.add_argument("--llm", action="store_true", help="Enable LLM analysis")
    parser.add_argument("--skip-validation", action="store_true", help="Skip config validation")
    args = parser.parse_args()

    configs_dir = Path(__file__).parent.parent / "configs"

    print("\n" + "=" * 70)
    print("  CONFIG & DATA DOCTOR TEST SUITE")
    print("=" * 70)
    print(f"  Configs directory: {configs_dir}")
    print(f"  LLM enabled: {args.llm}")

    # Phase 1: Validate all configs
    if not args.skip_validation:
        validation_results = validate_all_configs(configs_dir)

    # Phase 2: Run Data Doctor on local configs
    pipeline_results = run_full_pipeline_tests(configs_dir, enable_llm=args.llm)

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    if not args.skip_validation:
        print(f"  Config Validation: {len(validation_results['passed'])}/{len(validation_results['passed']) + len(validation_results['failed'])} passed")

    print(f"  Data Doctor Tests: {len(pipeline_results['passed'])}/{len(pipeline_results['passed']) + len(pipeline_results['failed'])} passed")

    if pipeline_results["failed"]:
        print("\n  Failed tests:")
        for name, error in pipeline_results["failed"]:
            print(f"    - {name}: {error}")
        sys.exit(1)

    print("\n  All tests passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
