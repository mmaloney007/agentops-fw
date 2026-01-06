"""
python -m tests.neuralift_c360_prep.features.test_map_state_and_region
"""

import pandas as pd

from neuralift_c360_prep.features.map_state_and_region import map_state_and_region


def test_map_state_and_region_mixed_inputs():
    """
    Test with a mixture of abbreviations, full names, different cases,
    and unrecognized states.
    """
    df_in = pd.DataFrame(
        {"STATE": ["tx", "TEXAS", "Ma", "Massachusetts", "ny ", "Cali", None, ""]}
    )

    df_out = map_state_and_region(df_in, state_col="STATE")

    # For row 0: "tx" -> "TX", "Texas", "South"
    assert df_out.loc[0, "usps_abbrev"] == "TX"
    assert df_out.loc[0, "full_state_name"] == "Texas"
    assert df_out.loc[0, "census_region"] == "South"

    # For row 1: "TEXAS" -> "TX"
    assert df_out.loc[1, "usps_abbrev"] == "TX"
    assert df_out.loc[1, "full_state_name"] == "Texas"
    assert df_out.loc[1, "census_region"] == "South"

    # For row 2: "Ma" -> "MA"
    assert df_out.loc[2, "usps_abbrev"] == "MA"
    assert df_out.loc[2, "full_state_name"] == "Massachusetts"
    assert df_out.loc[2, "census_region"] == "Northeast"

    # For row 3: "Massachusetts" -> "MA"
    assert df_out.loc[3, "usps_abbrev"] == "MA"
    assert df_out.loc[3, "full_state_name"] == "Massachusetts"
    assert df_out.loc[3, "census_region"] == "Northeast"

    # For row 4: "ny " -> "NY"
    assert df_out.loc[4, "usps_abbrev"] == "NY"
    assert df_out.loc[4, "full_state_name"] == "New York"
    assert df_out.loc[4, "census_region"] == "Northeast"

    # For row 5: "Cali" -> unrecognized
    assert pd.isna(df_out.loc[5, "usps_abbrev"]), "Expected 'Cali' to be NaN"
    assert pd.isna(df_out.loc[5, "full_state_name"])
    assert pd.isna(df_out.loc[5, "census_region"])

    # For row 6: None -> unrecognized
    assert pd.isna(df_out.loc[6, "usps_abbrev"])
    assert pd.isna(df_out.loc[6, "full_state_name"])
    assert pd.isna(df_out.loc[6, "census_region"])

    # For row 7: "" -> unrecognized
    assert pd.isna(df_out.loc[7, "usps_abbrev"])
    assert pd.isna(df_out.loc[7, "full_state_name"])
    assert pd.isna(df_out.loc[7, "census_region"])

    print("test_map_state_and_region_mixed_inputs passed.")


def test_map_state_and_region_single_input():
    """
    Test a single input row, verifying correct handling.
    """
    df_in = pd.DataFrame({"STATE": ["  new york  "]})
    df_out = map_state_and_region(df_in, state_col="STATE")

    # "new york" -> "NY", "New York", "Northeast"
    assert df_out.loc[0, "usps_abbrev"] == "NY"
    assert df_out.loc[0, "full_state_name"] == "New York"
    assert df_out.loc[0, "census_region"] == "Northeast"

    print("test_map_state_and_region_single_input passed.")


if __name__ == "__main__":
    test_map_state_and_region_mixed_inputs()
    test_map_state_and_region_single_input()
    print("All tests passed.")
