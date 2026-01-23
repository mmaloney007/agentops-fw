import pytest
import yaml

from neuralift_c360_prep.config import BundleConfig, load_config


def test_load_config_sample():
    cfg = load_config("configs/staging/data_prep.yaml")
    assert cfg.input.source == "parquet"
    assert cfg.output.target_mb_per_part == 512
    assert cfg.output.include_c360_tag is True
    assert cfg.metadata.op_timeout == "45m"


def test_metadata_defaults():
    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {"source": "parquet", "parquet_path": "x"},
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
            },
        }
    )
    assert cfg.metadata.max_concurrency == 75
    assert cfg.metadata.lift_strict is False


def test_bigquery_requires_project(monkeypatch, tmp_path):
    data = {
        "runtime": {"engine": "local"},
        "input": {"source": "bigquery", "bigquery": {"dataset": "d", "table": "t"}},
        "output": {
            "uc_catalog": "c",
            "uc_schema": "s",
            "uc_table": "t",
            "s3_base": "s3://b/",
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_dump(data))
    with pytest.raises(SystemExit):
        load_config(p)


def test_azure_allowed_but_ingest_raises():
    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {
                "source": "azure_parquet",
                "azure": {"account_name": "a", "container": "c", "path": "p"},
            },
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
            },
        }
    )
    assert cfg.input.source == "azure_parquet"


def test_legacy_min_partition_maps_to_target():
    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {"source": "parquet", "parquet_path": "x"},
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
                "min_partition_mb": 123,
            },
        }
    )
    assert cfg.output.target_mb_per_part == 123


def test_include_c360_tag_false():
    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {"source": "parquet", "parquet_path": "x"},
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
                "include_c360_tag": False,
            },
        }
    )
    assert cfg.output.include_c360_tag is False


def test_coiled_vm_types_valid():
    cfg = BundleConfig.model_validate(
        {
            "runtime": {
                "engine": "coiled",
                "coiled": {
                    "worker_vm_types": ["f1.2xlarge"],
                    "scheduler_vm_types": ["f1.2xlarge"],
                },
            },
            "input": {"source": "parquet", "parquet_path": "x"},
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
            },
        }
    )
    assert cfg.runtime.coiled.worker_vm_types == ["f1.2xlarge"]
    assert cfg.runtime.coiled.scheduler_vm_types == ["f1.2xlarge"]


def yaml_dump(data) -> str:
    return yaml.safe_dump(data, sort_keys=False)


def test_data_doctor_defaults():
    """Test DataDoctorConfig has sensible defaults."""
    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {"source": "parquet", "parquet_path": "x"},
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
            },
        }
    )
    # Check defaults
    assert cfg.data_doctor.enabled is True
    assert cfg.data_doctor.save_yaml is True
    assert cfg.data_doctor.show_business_alternatives is True
    assert cfg.data_doctor.high_null_threshold == 100
    assert cfg.data_doctor.high_cardinality_threshold == 50
    assert cfg.data_doctor.top_k_bucket == 10


def test_data_doctor_can_be_disabled():
    """Test that data doctor can be disabled via config."""
    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {"source": "parquet", "parquet_path": "x"},
            "data_doctor": {
                "enabled": False,
            },
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
            },
        }
    )
    assert cfg.data_doctor.enabled is False


def test_data_doctor_custom_thresholds():
    """Test that data doctor thresholds can be customized."""
    cfg = BundleConfig.model_validate(
        {
            "runtime": {"engine": "local"},
            "input": {"source": "parquet", "parquet_path": "x"},
            "data_doctor": {
                "high_null_threshold": 50,
                "high_cardinality_threshold": 100,
                "top_k_bucket": 20,
            },
            "output": {
                "uc_catalog": "c",
                "uc_schema": "s",
                "uc_table": "t",
                "s3_base": "s3://b/",
            },
        }
    )
    assert cfg.data_doctor.high_null_threshold == 50
    assert cfg.data_doctor.high_cardinality_threshold == 100
    assert cfg.data_doctor.top_k_bucket == 20
