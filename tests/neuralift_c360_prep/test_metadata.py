import math
import pandas as pd
import dask.dataframe as dd

from neuralift_c360_prep.config import BundleConfig
from neuralift_c360_prep import metadata as metadata_mod
from neuralift_c360_prep.metadata import (
    build_metadata,
    build_table_comment,
    build_pretty_config_from_data_dict,
    render_config_yaml_with_comments,
    suggest_explainability_hparams,
    suggest_autoencoder_dims,
    suggest_batch_size,
    suggest_segmenter_hparams,
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


def test_autoencoder_dims_exclude_latent():
    encoder, decoder, latent = suggest_autoencoder_dims(128)
    assert latent not in encoder
    assert latent not in decoder
    assert decoder == list(reversed(encoder))


def test_batch_size_targets_steps_per_epoch():
    n_rows = 10000
    batch_size = suggest_batch_size(
        use_gpu=True,
        n_features=10,
        n_rows=n_rows,
        device_mem_gb=64,
        target_steps_per_epoch=(10, 20),
    )
    steps = math.ceil(n_rows / batch_size)
    assert 10 <= steps <= 20


def test_render_config_yaml_includes_default_comment():
    config = {"dae": {"trainer": {"max_epochs": 100}}, "use_gpu": True}
    defaults = {"dae": {"trainer": {"max_epochs": 50}}, "use_gpu": True}
    text = render_config_yaml_with_comments(config, defaults=defaults)
    assert "max_epochs: 100 # default: 50" in text
    assert "use_gpu: true" in text
    assert "use_gpu: true # default" not in text


def test_explainability_top_n_minimum():
    explain = suggest_explainability_hparams(1)
    assert explain["top_n"] >= 10
    assert explain["top_n"] <= explain["num_features"]


def test_segmenter_pct_fields_null():
    seg = suggest_segmenter_hparams(10000)
    assert seg["min_cluster_pct"] is None
    assert seg["min_samples_pct"] is None


def test_wandb_nulls_artifact_filenames():
    pdf = pd.DataFrame({"id": [1, 2], "x": [1.0, 2.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    data_dict = {
        "columns": [
            {"name": "id", "type": "id"},
            {"name": "x", "type": "continuous"},
        ]
    }
    config = build_pretty_config_from_data_dict(
        data_dict=data_dict,
        ddf=ddf,
        use_wandb=True,
        wandb_project="test_project",
    )
    assert config["use_wandb"] is True
    assert "delete_existing_artifacts" not in config
    assert "labels_file_name" not in config
    assert "ranked_points_file_name" not in config

    debug_config = build_pretty_config_from_data_dict(
        data_dict=data_dict,
        ddf=ddf,
        use_wandb=True,
        wandb_project="test_project",
        config_debug=True,
    )
    assert debug_config["labels_file_name"] is None
    assert debug_config["ranked_points_file_name"] is None


def test_build_pretty_config_uses_row_count(monkeypatch):
    pdf = pd.DataFrame({"id": [1, 2], "x": [1.0, 2.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    data_dict = {
        "columns": [
            {"name": "id", "type": "id", "unique_count": 2},
            {"name": "x", "type": "continuous", "unique_count": 2},
        ]
    }

    def _boom(_ddf):
        raise AssertionError("_safe_row_count should not be called when row_count is provided")

    monkeypatch.setattr(metadata_mod, "_safe_row_count", _boom)

    config = build_pretty_config_from_data_dict(
        data_dict=data_dict,
        ddf=ddf,
        use_wandb=False,
        row_count=2,
    )
    assert config["dae"]["data_module"]["batch_size"] > 0
