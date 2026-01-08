import json
from pathlib import Path

import pandas as pd
import dask.dataframe as dd

from neuralift_c360_prep.config import BundleConfig
from neuralift_c360_prep import write as write_mod


def test_write_outputs_local(tmp_path, monkeypatch):
    pdf = pd.DataFrame({"id": [1, 2], "val": [1, 2]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {"source": "parquet", "parquet_path": "x", "id_cols": ["id"]},
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": tmp_path.as_posix(),
                "partitions": ["id"],
                "target_mb_per_part": 1,
            },
        }
    )

    monkeypatch.setattr(write_mod, "_assert_volume_absent", lambda cfg: None)

    base = write_mod.write_outputs(
        ddf, cfg, meta_json_text=json.dumps({"table": "t"}), config_text="config: demo"
    )
    assert Path(base, "input_data").exists()
    assert Path(base, "data_dictionary.json").exists()
    assert Path(base, "config.yaml").exists()


def test_write_min_two_partitions(tmp_path):
    pdf = pd.DataFrame({"id": [1, 2], "val": [1, 2]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    base = write_mod.write_ddf_and_yaml_to_s3(
        ddf=ddf,
        s3_base=tmp_path.as_posix(),
        config_yaml_text="config: demo",
        meta_json_text=json.dumps({"table": "t"}),
        partition_on=None,
        target_mb_per_part=512,
        force_npartitions=None,
        show_progress=False,
    )

    assert write_mod.count_parquet_files(s3_base=base) == 2
