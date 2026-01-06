import yaml
from pathlib import Path

from neuralift_c360_prep.config import load_config
from neuralift_c360_prep import pipeline as pipe
from neuralift_c360_prep import write as write_mod


def test_csv_dtype_overrides(tmp_path, monkeypatch):
    data_path = Path("tests/fixtures/wine/wine.csv").resolve()
    cfg_data = {
        "runtime": {"engine": "local"},
        "input": {
            "source": "csv",
            "csv_path": data_path.as_posix(),
            "id_cols": ["id"],
            "dtype_overrides": {"revenue": "float64"},
        },
        "output": {
            "uc_catalog": "c",
            "uc_schema": "s",
            "uc_table": "t",
            "s3_base": tmp_path.as_posix(),
            "partitions": ["id"],
            "target_mb_per_part": 10,
        },
        "metadata": {"context": "dtype override test"},
    }
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg_data, sort_keys=False))

    monkeypatch.setattr(write_mod, "_assert_volume_absent", lambda cfg: None)

    cfg = load_config(cfg_file)
    pipe.run_from_config(cfg)
    assert (tmp_path / "input_data").exists()
