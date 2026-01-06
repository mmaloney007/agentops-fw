import pandas as pd

from neuralift_c360_prep.features.zip_utils import (
    add_distance_indicators_dask,
    clean_zip_codes_dask,
)


def test_clean_zip_codes_dask():
    df = pd.DataFrame({"ZIP_Code": ["12345", "123456", "ABCDE", "00000", None]})
    out = clean_zip_codes_dask(df, zip_column="ZIP_Code")

    expected = ["12345", "12345", pd.NA, pd.NA, pd.NA]
    for value, exp in zip(out["ZIP5"].tolist(), expected):
        if pd.isna(exp):
            assert pd.isna(value)
        else:
            assert value == exp


def test_add_distance_indicators_dask():
    df = pd.DataFrame({"ZIP5": ["12345", "67890"], "value": [1, 2]})
    distance_df = pd.DataFrame(
        {"ZIP5": ["12345"], "distance_miles": [10.0], "duration_min": [20.0]}
    )

    merged = add_distance_indicators_dask(df, distance_df, on="ZIP5")
    assert merged.loc[0, "distance_miles"] == 10.0
    assert pd.isna(merged.loc[1, "distance_miles"])

    subset = add_distance_indicators_dask(
        df,
        distance_df,
        on="ZIP5",
        keep_cols=("ZIP5", "distance_miles"),
    )
    assert list(subset.columns) == ["ZIP5", "distance_miles"]
