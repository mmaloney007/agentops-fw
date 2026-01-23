"""Tests for config_generator module."""

import pytest
import yaml
from unittest.mock import MagicMock, patch

from neuralift_c360_prep.config_generator import (
    generate_config_from_columns,
    generate_config_from_uc_table,
)


class TestGenerateConfigFromColumns:
    """Tests for generate_config_from_columns function."""

    def test_extracts_id_columns(self):
        """ID columns should be extracted from type=id tag."""
        columns = [
            {"name": "customer_id", "data_type": "string", "tags": {"type": "id"}},
            {"name": "name", "data_type": "string", "tags": {"type": "cat"}},
        ]

        config = generate_config_from_columns(
            uc_table="catalog.schema.table",
            columns=columns,
        )

        assert config["ids"]["columns"] == ["customer_id"]
        assert config["ids"]["auto_detect"] is False

    def test_extracts_kpi_columns(self):
        """KPI columns should be extracted with identity function."""
        columns = [
            {"name": "customer_id", "data_type": "string", "tags": {"type": "id"}},
            {"name": "is_churned", "data_type": "boolean", "tags": {"type": "kpi"}},
            {
                "name": "total_spent_tier",
                "data_type": "string",
                "tags": {"type": "kpi"},
            },
        ]

        config = generate_config_from_columns(
            uc_table="catalog.schema.table",
            columns=columns,
        )

        assert len(config["functions"]) == 2
        assert config["functions"][0] == {
            "type": "identity",
            "column": "is_churned",
            "kpi": True,
        }
        assert config["functions"][1] == {
            "type": "identity",
            "column": "total_spent_tier",
            "kpi": True,
        }

    def test_extracts_lift_metadata(self):
        """KPI columns with lift tags should include lift metadata."""
        columns = [
            {
                "name": "total_spent_tier",
                "data_type": "string",
                "tags": {
                    "type": "kpi",
                    "value_sum_column": "total_spent",
                    "value_sum_unit": "USD",
                    "event_sum_column": "months_subscribed",
                    "event_sum_unit": "months",
                },
            },
        ]

        config = generate_config_from_columns(
            uc_table="catalog.schema.table",
            columns=columns,
        )

        assert config["functions"][0]["lift"] == {
            "value_sum_column": "total_spent",
            "value_sum_unit": "USD",
            "event_sum_column": "months_subscribed",
            "event_sum_unit": "months",
        }

    def test_partial_lift_metadata(self):
        """KPI with partial lift tags should include only present fields."""
        columns = [
            {
                "name": "revenue_tier",
                "data_type": "string",
                "tags": {
                    "type": "kpi",
                    "value_sum_column": "revenue",
                    "value_sum_unit": "USD",
                },
            },
        ]

        config = generate_config_from_columns(
            uc_table="catalog.schema.table",
            columns=columns,
        )

        assert config["functions"][0]["lift"] == {
            "value_sum_column": "revenue",
            "value_sum_unit": "USD",
        }

    def test_no_lift_when_no_lift_tags(self):
        """KPI without lift tags should not have lift key."""
        columns = [
            {"name": "is_active", "data_type": "boolean", "tags": {"type": "kpi"}},
        ]

        config = generate_config_from_columns(
            uc_table="catalog.schema.table",
            columns=columns,
        )

        assert "lift" not in config["functions"][0]

    def test_handles_hyphenated_catalog_names(self):
        """Should correctly parse hyphenated catalog/schema names."""
        columns = [
            {"name": "customer_id", "data_type": "string", "tags": {"type": "id"}},
        ]

        config = generate_config_from_columns(
            uc_table="staging-c360.my-schema.my_table",
            columns=columns,
        )

        assert config["output"]["uc_catalog"] == "staging-c360"
        assert config["output"]["uc_schema"] == "my-schema"
        assert config["output"]["uc_table"] == "my_table_prepared"

    def test_sets_workspace_and_software_env(self):
        """Should use provided workspace and software_env."""
        columns = []

        config = generate_config_from_columns(
            uc_table="catalog.schema.table",
            columns=columns,
            workspace="neuralift-prod",
            software_env="custom_env",
        )

        assert config["runtime"]["coiled"]["workspace"] == "neuralift-prod"
        assert config["runtime"]["coiled"]["software_env"] == "custom_env"

    def test_auto_detects_dev_workspace_for_staging_c360(self):
        """Should auto-detect neuralift-dev for staging-c360 catalog."""
        columns = []

        config = generate_config_from_columns(
            uc_table="staging-c360.media.my_table",
            columns=columns,
        )

        assert config["runtime"]["coiled"]["workspace"] == "neuralift-dev"

    def test_auto_detects_prod_workspace_for_other_catalogs(self):
        """Should auto-detect neuralift-prod for non-staging-c360 catalogs."""
        columns = []

        config = generate_config_from_columns(
            uc_table="staging-media-source.default.media_lp",
            columns=columns,
        )

        assert config["runtime"]["coiled"]["workspace"] == "neuralift-prod"

    def test_metadata_has_required_fields(self):
        """Metadata should have model, sample_rows, max_concurrency, context."""
        columns = [
            {"name": "customer_id", "data_type": "string", "tags": {"type": "id"}},
        ]

        config = generate_config_from_columns(
            uc_table="catalog.schema.table",
            columns=columns,
        )

        assert config["metadata"]["model"] == "gpt-5-nano"
        assert config["metadata"]["sample_rows"] == 5000
        assert config["metadata"]["max_concurrency"] == 50
        assert "context" in config["metadata"]

    def test_writes_yaml_to_file(self, tmp_path):
        """Should write valid YAML to output path."""
        columns = [
            {"name": "customer_id", "data_type": "string", "tags": {"type": "id"}},
            {"name": "is_churned", "data_type": "boolean", "tags": {"type": "kpi"}},
        ]
        output_file = tmp_path / "generated.yaml"

        generate_config_from_columns(
            uc_table="staging-c360.media.vivastream_27m",
            columns=columns,
            output_path=output_file,
        )

        assert output_file.exists()
        content = output_file.read_text()
        assert "# Generated from: staging-c360.media.vivastream_27m" in content

        # Verify it's valid YAML (skip comment lines)
        yaml_content = "\n".join(
            line for line in content.split("\n") if not line.startswith("#")
        )
        parsed = yaml.safe_load(yaml_content)
        assert parsed["ids"]["columns"] == ["customer_id"]

    def test_empty_columns_produces_valid_config(self):
        """Should produce valid config even with no tagged columns."""
        columns = [
            {"name": "some_col", "data_type": "string", "tags": {}},
        ]

        config = generate_config_from_columns(
            uc_table="catalog.schema.table",
            columns=columns,
        )

        assert config["ids"]["columns"] == []
        assert config["functions"] == []

    def test_uses_table_comment_as_context(self):
        """Should use table comment for metadata context if provided."""
        columns = [
            {"name": "customer_id", "data_type": "string", "tags": {"type": "id"}},
        ]

        config = generate_config_from_columns(
            uc_table="catalog.schema.table",
            columns=columns,
            table_comment="This is a customer analytics table with churn data.",
        )

        assert (
            config["metadata"]["context"]
            == "This is a customer analytics table with churn data."
        )

    def test_uses_default_context_without_table_comment(self):
        """Should use default context when no table comment provided."""
        columns = []

        config = generate_config_from_columns(
            uc_table="catalog.schema.my_table",
            columns=columns,
        )

        assert (
            "Generated config from catalog.schema.my_table"
            in config["metadata"]["context"]
        )

    def test_ignores_unknown_type_tags(self):
        """Columns with unknown type tags should be ignored."""
        columns = [
            {"name": "customer_id", "data_type": "string", "tags": {"type": "id"}},
            {"name": "unknown_col", "data_type": "string", "tags": {"type": "unknown"}},
        ]

        config = generate_config_from_columns(
            uc_table="catalog.schema.table",
            columns=columns,
        )

        assert config["ids"]["columns"] == ["customer_id"]
        assert len(config["functions"]) == 0


class TestGenerateConfigFromUcTable:
    """Tests for generate_config_from_uc_table with mocked Databricks client."""

    def test_calls_get_column_tags(self):
        """Should fetch columns from UC and pass to generate_config_from_columns."""
        mock_columns = [
            {"name": "customer_id", "data_type": "string", "tags": {"type": "id"}},
        ]
        mock_comment = "Test table comment"

        with patch(
            "neuralift_c360_prep.config_generator._get_column_tags",
            return_value=(mock_columns, mock_comment),
        ) as mock_get:
            config = generate_config_from_uc_table(
                uc_table="staging-c360.media.vivastream_27m",
            )

            mock_get.assert_called_once_with("staging-c360.media.vivastream_27m")
            assert config["ids"]["columns"] == ["customer_id"]
            assert config["metadata"]["context"] == "Test table comment"


class TestGetColumnTags:
    """Tests for _get_column_tags with mocked Databricks SDK."""

    def test_parses_column_tags(self):
        """Should parse column tags from Databricks SDK and SQL."""
        from neuralift_c360_prep.config_generator import _get_column_tags

        mock_column = MagicMock()
        mock_column.name = "customer_id"
        mock_column.type_name = "STRING"

        mock_table_info = MagicMock()
        mock_table_info.columns = [mock_column]
        mock_table_info.comment = "Test table comment"

        mock_client = MagicMock()
        mock_client.tables.get.return_value = mock_table_info

        # Mock SQL tags fetch
        mock_tags = {"customer_id": {"type": "id"}}

        with patch("databricks.sdk.WorkspaceClient", return_value=mock_client):
            with patch(
                "neuralift_c360_prep.config_generator._get_column_tags_via_sql",
                return_value=mock_tags,
            ):
                columns, table_comment = _get_column_tags(
                    "staging-c360.media.vivastream_27m"
                )

                assert len(columns) == 1
                assert columns[0]["name"] == "customer_id"
                assert columns[0]["data_type"] == "STRING"
                assert columns[0]["tags"] == {"type": "id"}
                assert table_comment == "Test table comment"

    def test_handles_columns_without_tags(self):
        """Should handle columns that have no tags."""
        from neuralift_c360_prep.config_generator import _get_column_tags

        mock_column = MagicMock()
        mock_column.name = "some_column"
        mock_column.type_name = "INT"

        mock_table_info = MagicMock()
        mock_table_info.columns = [mock_column]
        mock_table_info.comment = None

        mock_client = MagicMock()
        mock_client.tables.get.return_value = mock_table_info

        with patch("databricks.sdk.WorkspaceClient", return_value=mock_client):
            with patch(
                "neuralift_c360_prep.config_generator._get_column_tags_via_sql",
                return_value={},
            ):
                columns, table_comment = _get_column_tags("catalog.schema.table")

                assert columns[0]["tags"] == {}
                assert table_comment is None

    def test_invalid_table_format_raises(self):
        """Should raise ValueError for invalid table format."""
        from neuralift_c360_prep.config_generator import _parse_uc_table

        with pytest.raises(ValueError, match="Invalid UC table format"):
            _parse_uc_table("invalid_format")
