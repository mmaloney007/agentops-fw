import contextlib
import os
from pathlib import Path

import pytest
import yaml
import dask.dataframe as dd

from neuralift_c360_prep import pipeline as pipeline_mod
from neuralift_c360_prep import write as write_mod
from neuralift_c360_prep.config import BundleConfig, load_config


def test_pipeline_local_end_to_end(tmp_path, monkeypatch):
    data_path = Path("tests/fixtures/wine/wine.csv").resolve()
    cfg_data = {
        "runtime": {"engine": "local"},
        "input": {"source": "csv", "csv_path": data_path.as_posix(), "id_cols": ["id"]},
        "output": {
            "uc_catalog": "c",
            "uc_schema": "s",
            "uc_table": "t",
            "s3_base": tmp_path.as_posix(),
            "partitions": ["id"],
            "target_mb_per_part": 10,
        },
        "metadata": {"context": "pipeline test"},
    }
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg_data, sort_keys=False))

    monkeypatch.setattr(write_mod, "_assert_volume_absent", lambda cfg: None)

    cfg = load_config(cfg_file)
    base = pipeline_mod.run_from_config(cfg)
    assert Path(base, "input_data").exists()
    assert Path(base, "config.yaml").exists()
    assert Path(base, "bundleconfig.yaml").exists()
    assert Path(base, "data_dictionary.json").exists()


def test_pipeline_delta_path_maps_to_delta(monkeypatch):
    class _Stop(Exception):
        pass

    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {
                "source": "delta_path",
                "delta_path": "s3://bucket/delta/",
                "id_cols": ["id"],
            },
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
            },
            "metadata": {"context": "delta path test"},
        }
    )

    class DummyClient:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(pipeline_mod, "get_client", lambda cfg: DummyClient())

    def fake_load_lazy_dask(*, fmt, uri, **_kwargs):
        assert fmt == "delta"
        assert uri == "s3://bucket/delta/"
        raise _Stop

    monkeypatch.setattr(pipeline_mod, "load_lazy_dask", fake_load_lazy_dask)

    with pytest.raises(_Stop):
        pipeline_mod.run_from_config(cfg)


def test_pipeline_data_prep_config_runs(monkeypatch):
    monkeypatch.setattr(
        pipeline_mod, "get_client", lambda _cfg: contextlib.nullcontext()
    )

    cfg = load_config("configs/data_prep.yaml")
    base = pipeline_mod.run_from_config(cfg)
    assert Path(base, "input_data").exists()
    assert Path(base, "config.yaml").exists()
    assert Path(base, "bundleconfig.yaml").exists()
    assert Path(base, "data_dictionary.json").exists()


def test_pipeline_wine_and_cheese_config_runs(monkeypatch):
    monkeypatch.setattr(
        pipeline_mod, "get_client", lambda _cfg: contextlib.nullcontext()
    )

    def _stub_load_lazy_dask(*_args, **_kwargs):
        return dd.read_parquet("example-data/wine_cheese.parquet")

    def _stub_create_volume(**_kwargs):
        return (
            "local_volume",
            "staging-c360.kaggle-marketing.local_volume",
            "./local-output",
            "2026-01-01",
            "2026-01-01T00:00:00Z",
        )

    monkeypatch.setattr(pipeline_mod, "load_lazy_dask", _stub_load_lazy_dask)
    monkeypatch.setattr(
        pipeline_mod, "create_managed_uc_volume_via_sql", _stub_create_volume
    )
    monkeypatch.setattr(pipeline_mod, "tag_uc_volume_via_sql", lambda **_kwargs: None)

    cfg = load_config("configs/wine_and_cheese.yaml")
    base = pipeline_mod.run_from_config(cfg)
    assert Path(base, "input_data").exists()
    assert Path(base, "config.yaml").exists()
    assert Path(base, "bundleconfig.yaml").exists()
    assert Path(base, "data_dictionary.json").exists()


@pytest.mark.integration
def test_pipeline_wine_and_cheese_real():
    if os.getenv("NL_INTEGRATION") != "1":
        pytest.skip("Set NL_INTEGRATION=1 to run the real wine_and_cheese pipeline.")

    cfg = load_config("configs/wine_and_cheese.yaml")
    base = pipeline_mod.run_from_config(cfg)
    assert Path(base, "input_data").exists()
    assert Path(base, "config.yaml").exists()
    assert Path(base, "bundleconfig.yaml").exists()
    assert Path(base, "data_dictionary.json").exists()
