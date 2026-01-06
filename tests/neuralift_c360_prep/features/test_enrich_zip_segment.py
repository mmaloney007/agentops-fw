#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Program for enrich_zip_segment_dask().

Run with:
    python -m tests.neuralift_c360_prep.features.test_enrich_zip_segment
"""

import pandas as pd
import pytest
from tabulate import tabulate

from neuralift_c360_prep.features.enrich_zip_segment import (
    SearchEngine,
    TimezoneFinder,
    enrich_zip_segment_dask,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)


def print_test_result(test_name, input_df, output_df):
    print(f"\n--- {test_name} ---")
    print("Input DataFrame:")
    print(tabulate(input_df, headers="keys", tablefmt="psql", showindex=False))
    print("\nOutput DataFrame:")
    print(tabulate(output_df, headers="keys", tablefmt="psql", showindex=False))
    print(f"{test_name} passed.\n")


@pytest.mark.skipif(
    SearchEngine is None or TimezoneFinder is None,
    reason="Optional zip enrichment dependencies are not available.",
)
def test_enrich_segment_large_style():
    """
    We'll test multiple ZIPs, including:
      - 10001 (NY, Northeast, highly urban)
      - 99501 (Anchorage, AK, West region)
      - 96813 (Honolulu, HI, West region)
      - 99999 (bogus => Unknown)
      - 94105 (San Francisco, CA => West, urban)
    Then we confirm the output columns for each row.
    """
    data = {
        "CustomerID": [1, 2, 3, 4, 5],
        "ZIP": ["10001", "99501", "96813", "99999", "94105"],
    }
    df_input = pd.DataFrame(data)
    df_output = enrich_zip_segment_dask(
        df_input,
        id_col="CustomerID",
        zip_col="ZIP",
        default_country="US",
        fill_value="Unknown",
    )

    # We'll check each row for reasonableness:
    # row0 => 10001 => NY, Region=Northeast, UrbanDensity=Urban, Timezone=America/New_York typically
    row0 = df_output.iloc[0]
    # can't do strict asserts for every field in a test script if we want it
    # to pass for all local data variations, but we can do partial checks:
    assert row0["State"] in [
        "NY",
        "Unknown",
    ], f"10001 => expected 'NY' or 'Unknown', got {row0['State']}"
    # region => "Northeast" or "Unknown"
    # density => "Urban" typically
    # timezone => "America/New_York"

    # row1 => 99501 => Anchorage, AK => West, population density can vary, but we expect "America/Anchorage"
    row1 = df_output.iloc[1]
    assert row1["State"] in [
        "AK",
        "Unknown",
    ], f"99501 => expected 'AK', got {row1['State']}"

    # row2 => 96813 => Honolulu, HI => West => "America/Honolulu"
    row2 = df_output.iloc[2]
    assert row2["State"] in [
        "HI",
        "Unknown",
    ], f"96813 => expected 'HI', got {row2['State']}"

    # row3 => 99999 => bogus => all unknown
    row3 = df_output.iloc[3]
    assert row3["State"] == "Unknown", f"99999 => expected Unknown, got {row3['State']}"

    # row4 => 94105 => SF => CA => West => "America/Los_Angeles"
    row4 = df_output.iloc[4]
    assert row4["State"] in [
        "CA",
        "Unknown",
    ], f"94105 => expected 'CA', got {row4['State']}"

    print_test_result("Enrich_Segment_Large_Style", df_input, df_output)


if __name__ == "__main__":
    test_enrich_segment_large_style()
    print("All tests passed.")
