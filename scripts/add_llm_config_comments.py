#!/usr/bin/env python3
"""
Add LLM configuration comments and sections to all config files.

This script adds documented LLM configuration sections to each config file,
showing how to enable/disable ID detection and Data Doctor LLM features.

Usage:
    pixi run -e dev python scripts/add_llm_config_comments.py

Author: Mike Maloney - Neuralift, Inc.
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import re
from pathlib import Path


# LLM config template to insert after 'ids:' section
IDS_LLM_TEMPLATE = """  # ---------------------------------------------------------------------------
  # LLM-Enhanced ID Detection (optional, disabled by default)
  # ---------------------------------------------------------------------------
  # detection:
  #   llm_enabled: false           # Set to true to enable LLM semantic analysis
  #   llm_provider:
  #     provider: auto             # Options: openai, anthropic, auto
  #     model: null                # Use provider default or specify: gpt-4o-mini
  #   max_llm_columns: 20          # Limit columns sent to LLM (cost control)
  #   llm_cache_enabled: true      # Cache LLM responses to save cost
  #   uniqueness_threshold: 0.95   # Ratio to consider column unique
  #   gray_zone_lower: 0.80        # Lower bound for ambiguous uniqueness
  #   detect_uuid_format: true     # Enable UUID/GUID pattern detection
"""

# Data Doctor config template
DATA_DOCTOR_TEMPLATE = """# -----------------------------------------------------------------------------
# Data Doctor - Automated data quality analysis and suggestions
# -----------------------------------------------------------------------------
# Runs after metadata generation to provide actionable improvement suggestions.
# Rule-based analysis is always on. LLM enhancement is optional.
# -----------------------------------------------------------------------------
data_doctor:
  enabled: true                    # Run Data Doctor analysis (rule-based)
  save_yaml: true                  # Save suggestions.yaml alongside output
  # ---------------------------------------------------------------------------
  # LLM Enhancement (optional, disabled by default)
  # ---------------------------------------------------------------------------
  # llm_enabled: false             # Set to true to enable LLM analysis
  # llm_provider:
  #   provider: auto               # Options: openai, anthropic, auto
  #   model: null                  # Use provider default or specify: gpt-4o-mini
  # max_llm_columns: null          # Limit columns sent to LLM (null = all columns)
  # llm_cache_enabled: true        # Cache LLM responses to save cost
  # generate_executive_summary: true  # Generate natural language summary

"""


def add_llm_config_to_file(file_path: Path) -> bool:
    """Add LLM configuration comments to a config file."""
    content = file_path.read_text()
    modified = False

    # Check if IDs section exists and doesn't have LLM config
    if "ids:" in content and "llm_enabled" not in content:
        # Find the ids section and add LLM config after auto_detect
        ids_pattern = r"(ids:\n(?:  [^\n]+\n)*?  auto_detect: \w+\n)"
        match = re.search(ids_pattern, content)
        if match:
            ids_section = match.group(1)
            new_ids_section = ids_section + IDS_LLM_TEMPLATE
            content = content.replace(ids_section, new_ids_section)
            modified = True
            print("  Added IDs LLM config")

    # Check if data_doctor section exists
    if "data_doctor:" not in content:
        # Add data_doctor section before output section
        if "output:" in content:
            content = content.replace("output:", DATA_DOCTOR_TEMPLATE + "output:")
            modified = True
            print("  Added Data Doctor section")
    elif "llm_enabled" not in content:
        # data_doctor exists but no LLM config - add LLM comments
        dd_pattern = r"(data_doctor:\n(?:  [^\n]+\n)*)"
        match = re.search(dd_pattern, content)
        if match:
            dd_section = match.group(1)
            # Add LLM comments at end of data_doctor section
            llm_comments = """  # ---------------------------------------------------------------------------
  # LLM Enhancement (optional, disabled by default)
  # ---------------------------------------------------------------------------
  # llm_enabled: false             # Set to true to enable LLM analysis
  # llm_provider:
  #   provider: auto               # Options: openai, anthropic, auto
  #   model: null                  # Use provider default or specify: gpt-4o-mini
  # max_llm_columns: null          # Limit columns sent to LLM (null = all columns)
  # generate_executive_summary: true  # Generate natural language summary

"""
            new_dd_section = dd_section + llm_comments
            content = content.replace(dd_section, new_dd_section)
            modified = True
            print("  Added Data Doctor LLM comments")

    if modified:
        file_path.write_text(content)
        return True

    print("  No changes needed (LLM config already present or structure different)")
    return False


def main():
    configs_dir = Path(__file__).parent.parent / "configs"

    print("\n" + "=" * 70)
    print("  Adding LLM Configuration Comments to Config Files")
    print("=" * 70)

    updated = 0
    for config_file in sorted(configs_dir.glob("*.yaml")):
        print(f"\n  Processing: {config_file.name}")
        if add_llm_config_to_file(config_file):
            updated += 1

    print(f"\n  Summary: {updated} files updated")


if __name__ == "__main__":
    main()
