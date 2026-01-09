import pandas as pd
import dask.dataframe as dd

from neuralift_c360_prep.config import BundleConfig
from neuralift_c360_prep.preprocess import preprocess


def test_feature_functions_and_rename(tmp_path):
    pdf = pd.DataFrame(
        {
            "ID": [1],
            "state": ["TX"],
            "birth_date": ["1980-01-01"],
            "ZIP_Code": ["78701"],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg = BundleConfig.model_validate(
        {
            "logging": {"level": "info"},
            "runtime": {"engine": "local"},
            "input": {"source": "parquet", "parquet_path": "dummy", "id_cols": ["ID"]},
            "feature_functions": [
                "neuralift_c360_prep.features.birthday:add_birth_month",
                "neuralift_c360_prep.features.map_state_and_region:map_state_and_region",
                "neuralift_c360_prep.features.zip_utils:clean_zip_codes_dask",
            ],
            "preprocessing": {"rename_to_snake": True, "missing_fill": "auto"},
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
            },
        }
    )

    out = preprocess(ddf, cfg).compute()
    assert "birth_month" in out.columns
    assert "usps_abbrev" in out.columns
    assert "id" in out.columns  # renamed to snake_case
    assert "zip5" in out.columns


def test_kpi_functions_zsml():
    pdf = pd.DataFrame({"id": [1, 2, 3], "x": [0, 10, 100]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = BundleConfig.model_validate(
        {
            "logging": {"level": "info"},
            "runtime": {"engine": "local"},
            "input": {"source": "parquet", "parquet_path": "dummy", "id_cols": ["id"]},
            "kpi_functions": [
                {
                    "type": "zsml",
                    "source_col": "x",
                    "out_col": "x_tier",
                    "zero_threshold": 0.0,
                    "quantiles": [0.33, 0.66],
                }
            ],
            "preprocessing": {"rename_to_snake": True, "missing_fill": "none"},
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
            },
        }
    )
    out = preprocess(ddf, cfg).compute()
    assert "x_tier" in out.columns
