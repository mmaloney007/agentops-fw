import numpy as np
import pandas as pd
import dask.dataframe as dd
import pytest

from neuralift_c360_prep.config import BundleConfig
from neuralift_c360_prep.functions import apply_functions


def _base_cfg():
    return {
        "runtime": {"engine": "local"},
        "input": {"source": "parquet", "parquet_path": "x"},
        "ids": {"columns": ["id"]},
        "output": {
            "uc_catalog": "c",
            "uc_schema": "s",
            "uc_table": "t",
            "s3_base": "s3://b/",
        },
    }


def test_apply_functions_zsml_and_binning():
    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "revenue": [10.0, 20.0, 30.0],
            "age": [25, 35, 45],
            "customer_complained_recently": [0, 1, 0],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "kpis": [
            {
                "type": "zsml",
                "source_col": "revenue",
                "out_col": "RevenueTier",
                "quantiles": [0.33, 0.66],
            }
        ],
        "identity_kpis": [{"column": "customer_complained_recently"}],
        "features": [
            {
                "type": "binning",
                "source_col": "age",
                "out_col": "AgeBucket",
            }
        ],
    }

    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()
    assert "revenue_tier" in out.columns
    assert "age_bucket" in out.columns
    assert "customer_complained_recently" in out.columns


def test_apply_functions_return_mode_new_only():
    pdf = pd.DataFrame(
        {
            "id": [1, 2],
            "birth_date": ["2000-01-01", "1990-05-05"],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "features": [
            {
                "type": "callable",
                "callable": "neuralift_c360_prep.features.birthday:add_birth_month",
                "inputs": ["birth_date"],
                "return": "new_only",
            }
        ]
    }
    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()

    assert list(out.columns) == ["id", "birth_month", "birth_day"]


def test_apply_functions_winsorize():
    pdf = pd.DataFrame({"id": [1, 2, 3], "value": [1.0, 2.0, 100.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "features": [
            {
                "type": "winsorize",
                "source_col": "value",
                "lower_bound": 0.0,
                "upper_bound": 10.0,
            }
        ]
    }

    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()
    assert out["value_winsorized"].max() == 10.0


def test_apply_functions_log_transform():
    pdf = pd.DataFrame({"id": [1], "amount": [9.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "features": [
            {
                "type": "log_transform",
                "source_col": "amount",
                "log_method": "log1p",
                "log_clip_min": 0.0,
            }
        ]
    }

    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()
    assert out["amount_log"].iloc[0] == pytest.approx(np.log1p(9.0))


def test_apply_functions_log_transform_on_fallback():
    pdf = pd.DataFrame({"id": [1], "amount_spent_wine": [10.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "features": [
            {
                "type": "log_transform",
                "source_col": "AmountSpentOnWine",
                "log_method": "log1p",
                "log_clip_min": 0.0,
            }
        ]
    }

    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()
    assert out["amount_spent_wine_log"].iloc[0] == pytest.approx(np.log1p(10.0))


def test_apply_functions_string_normalize():
    pdf = pd.DataFrame({"id": [1, 2], "city": ["  New  York ", "SAN\tFRANCISCO"]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "features": [
            {
                "type": "string_normalize",
                "source_col": "city",
                "out_col": "city_clean",
            }
        ]
    }

    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()
    assert list(out["city_clean"]) == ["new york", "san francisco"]


def test_apply_functions_frequency_encode():
    pdf = pd.DataFrame({"id": [1, 2, 3, 4], "tier": ["a", "a", "b", None]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "features": [
            {
                "type": "frequency_encode",
                "source_col": "tier",
                "out_col": "tier_freq",
                "normalize": True,
            }
        ]
    }

    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()
    assert out["tier_freq"].iloc[0] == pytest.approx(0.5)
    assert out["tier_freq"].iloc[2] == pytest.approx(0.25)
    # Unmapped values (None → "None" string) get filled with 0 (no NULLs in output)
    assert out["tier_freq"].iloc[3] == 0


def test_apply_functions_days_since():
    pdf = pd.DataFrame({"id": [1], "event_date": ["2024-01-05"]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "features": [
            {
                "type": "days_since",
                "source_col": "event_date",
                "reference_date": "2024-01-10",
            }
        ]
    }

    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()
    assert out["event_date_days_since"].iloc[0] == pytest.approx(5.0)


def test_apply_functions_date_parts_drop_source():
    pdf = pd.DataFrame({"id": [1], "join_date": ["2024-01-15"]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "features": [
            {
                "type": "date_parts",
                "source_col": "join_date",
                "date_parts": ["year", "month", "day"],
                "drop_source": True,
            }
        ]
    }

    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()
    assert "join_date" not in out.columns
    assert {"join_date_year", "join_date_month", "join_date_day"}.issubset(out.columns)


def test_apply_functions_date_parts_auto_daypart():
    pdf = pd.DataFrame(
        {
            "id": [1, 2],
            "event_ts": ["2024-01-15 09:30:00", "2024-01-15 23:00:00"],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "features": [
            {
                "type": "date_parts",
                "source_col": "event_ts",
                "date_parts": ["year"],
            }
        ]
    }

    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()
    assert "event_ts_year" in out.columns
    assert "event_ts_hour" in out.columns
    assert "event_ts_daypart" in out.columns
    assert list(out["event_ts_daypart"]) == ["morning", "night"]


def test_apply_functions_categorical_bucket():
    pdf = pd.DataFrame({"id": [1, 2, 3, 4], "city": ["a", "a", "b", "c"]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "features": [
            {
                "type": "categorical_bucket",
                "source_col": "city",
                "top_k": 1,
                "other_label": "other",
            }
        ]
    }

    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()
    assert list(out["city_bucket"]) == ["a", "a", "other", "other"]


def test_apply_functions_ratio():
    pdf = pd.DataFrame({"id": [1, 2], "num": [10.0, 0.0], "den": [2.0, 0.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_data = _base_cfg()
    cfg_data["functions"] = {
        "features": [
            {
                "type": "ratio",
                "numerator_col": "num",
                "denominator_col": "den",
                "out_col": "num_per_den",
                "on_zero": "zero",
            }
        ]
    }

    cfg = BundleConfig.model_validate(cfg_data)
    out = apply_functions(ddf, cfg).compute()
    assert list(out["num_per_den"]) == [5.0, 0.0]
