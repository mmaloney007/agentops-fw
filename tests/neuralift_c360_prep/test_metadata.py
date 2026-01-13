import pandas as pd
import dask.dataframe as dd

from neuralift_c360_prep.config import BundleConfig
from neuralift_c360_prep.metadata import (
    build_metadata,
    build_minimal_config,
    build_table_comment,
)


def test_metadata_builds():
    pdf = pd.DataFrame({"id": [1, 2], "revenue": [1.0, 2.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {"source": "parquet", "parquet_path": "x", "id_cols": ["id"]},
            "metadata": {
                "context": "demo",
                "tags": {"id_cols": ["id"], "kpi_cols": ["revenue"]},
            },
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
            },
        }
    )
    meta, text = build_metadata(ddf, cfg)
    assert meta["table"] == "t"
    assert "revenue" in text
    assert "id" in meta["columns"]
    assert "kpi" in meta["columns"]["revenue"]["tags"]
    assert build_table_comment(cfg)


def test_build_minimal_config_defaults():
    config = build_minimal_config(row_count=10000, run_name="demo_run")
    assert config["use_wandb"] is True
    assert config["wandb"]["project"] == "demo_run"
    assert config["dae"]["data_module"]["batch_size"] == 8192
    assert config["dae"]["data_module"]["compute_stats_from"] == "full"
    assert config["dae"]["trainer"]["max_epochs"] == 50
    assert config["explainability"] == {"num_features": 50, "top_n": 10}
    assert config["tuner"]["segment_size"] == "M"
    assert config["tuner"]["min_cluster_threshold_min"] == 50


def test_build_minimal_config_scales_threshold():
    config = build_minimal_config(row_count=50000, run_name="bigger_run")
    assert config["tuner"]["min_cluster_threshold_min"] == 200
