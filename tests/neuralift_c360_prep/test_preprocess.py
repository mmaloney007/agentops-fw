import pandas as pd
import dask.dataframe as dd
import pytest

from neuralift_c360_prep.config import BundleConfig
from neuralift_c360_prep.preprocess import preprocess


def test_preprocess_rename_and_fill_legacy(tmp_path):
    pdf = pd.DataFrame(
        {
            "ID": [1, 2],
            "state": ["TX", "CA"],
            "birth_date": ["1980-01-01", "1970-05-05"],
            "ZIP_Code": ["78701", "94105"],
            "income": [None, 10],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1)

    with pytest.warns(DeprecationWarning):
        cfg = BundleConfig.model_validate(
            {
                "logging": {"level": "info"},
                "runtime": {"engine": "local"},
                "input": {"source": "parquet", "parquet_path": "dummy"},
                "ids": {"columns": ["ID"]},
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
    assert "id" in out.columns  # renamed to snake_case
    assert pd.isna(out.loc[0, "income"])
    assert out.loc[1, "income"] == 10
