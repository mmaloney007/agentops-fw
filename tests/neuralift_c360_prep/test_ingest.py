import json
import re

import pandas as pd
import pytest

from neuralift_c360_prep import ingest as ingest_mod
from neuralift_c360_prep.config import BundleConfig
from neuralift_c360_prep.ingest import load_ddf


def _base_output():
    return {
        "uc_catalog": "c",
        "uc_schema": "s",
        "uc_table": "t",
        "s3_base": "s3://b/",
    }


def test_ingest_parquet_roundtrip(tmp_path):
    src = tmp_path / "data.parquet"

    pdf = pd.DataFrame({"id": [1, 2], "state": ["TX", "CA"], "revenue": [1.0, 2.0]})
    pdf.to_parquet(src, index=False)

    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {"source": "parquet", "parquet_path": str(src), "id_cols": ["id"]},
            "output": _base_output(),
        }
    )
    ddf = load_ddf(cfg)
    out = ddf.compute()
    assert list(out["id"]) == [1, 2]


def test_ingest_azure_not_implemented(tmp_path):
    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {
                "source": "azure_parquet",
                "azure": {"account_name": "a", "container": "c", "path": "p"},
            },
            "output": _base_output(),
        }
    )
    with pytest.raises(NotImplementedError):
        load_ddf(cfg)


def test_delta_log_mapping_from_metadata(tmp_path):
    log_dir = tmp_path / "_delta_log"
    log_dir.mkdir()

    schema = {
        "type": "struct",
        "fields": [
            {
                "name": "id",
                "type": "long",
                "nullable": True,
                "metadata": {
                    "delta.columnMapping.physicalName": "col_11111111_1111_1111_1111_111111111111"
                },
            },
            {
                "name": "state",
                "type": "string",
                "nullable": True,
                "metadata": {
                    "delta.columnMapping.physicalName": "col_22222222_2222_2222_2222_222222222222"
                },
            },
        ],
    }
    action = {"metaData": {"schemaString": json.dumps(schema)}}
    (log_dir / "00000000000000000005.json").write_text(
        json.dumps(action) + "\n", encoding="utf-8"
    )

    mapping = ingest_mod._read_delta_log_mapping(
        tmp_path.as_posix(), storage_options=None
    )
    assert mapping == {
        "col_11111111_1111_1111_1111_111111111111": "id",
        "col_22222222_2222_2222_2222_222222222222": "state",
    }


def test_require_logical_names_blocks_physical(tmp_path, monkeypatch):
    physical_a = "col_aaaaaaaa_aaaa_aaaa_aaaa_aaaaaaaaaaaa"
    physical_b = "col_bbbbbbbb_bbbb_bbbb_bbbb_bbbbbbbbbbbb"
    pdf = pd.DataFrame({physical_a: [1], physical_b: [2]})
    parquet_path = tmp_path / "data.parquet"
    pdf.to_parquet(parquet_path, index=False)

    monkeypatch.setattr(
        ingest_mod, "_lookup_storage_location", lambda _: parquet_path.as_posix()
    )
    monkeypatch.setattr(ingest_mod, "_phys_to_log_mapping", lambda *args, **kwargs: {})
    monkeypatch.setattr(ingest_mod, "_logical_cols_uc", lambda *args, **kwargs: [])

    with pytest.raises(RuntimeError, match="Logical column mapping failed"):
        ingest_mod.load_lazy_dask(
            fmt="databricks_table",
            uri="dummy.table",
            id_cols=[],
            prefer_delta=False,
            require_logical_names=True,
        )


def test_partial_mapping_drops_extra_physical(tmp_path, monkeypatch):
    physical_a = "col_aaaaaaaa_aaaa_aaaa_aaaa_aaaaaaaaaaaa"
    physical_b = "col_bbbbbbbb_bbbb_bbbb_bbbb_bbbbbbbbbbbb"
    physical_extra = "col_cccccccc_cccc_cccc_cccc_cccccccccccc"
    pdf = pd.DataFrame({physical_a: [1], physical_b: [2], physical_extra: [3]})
    parquet_path = tmp_path / "data.parquet"
    pdf.to_parquet(parquet_path, index=False)

    monkeypatch.setattr(
        ingest_mod, "_lookup_storage_location", lambda _: parquet_path.as_posix()
    )
    monkeypatch.setattr(
        ingest_mod,
        "_phys_to_log_mapping",
        lambda *args, **kwargs: {physical_a: "id", physical_b: "state"},
    )
    monkeypatch.setattr(
        ingest_mod, "_logical_cols_uc", lambda *args, **kwargs: ["id", "state"]
    )

    ddf = ingest_mod.load_lazy_dask(
        fmt="databricks_table",
        uri="dummy.table",
        id_cols=[],
        prefer_delta=False,
        require_logical_names=True,
    )

    assert list(ddf.columns) == ["id", "state"]


def _normalize_table_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def test_format_debug_head_table_transposed():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]})
    table, row_count = ingest_mod._format_debug_head_table(pdf, rows=5)
    assert row_count == 5
    lines = [_normalize_table_line(line) for line in table.splitlines()]
    assert lines[0] == "| column | v1 | v2 | v3 | v4 | v5 |"
    assert lines[2] == "| a | 1 | 2 | 3 | 4 | 5 |"
    assert lines[3] == "| b | x | y | z | w | v |"
