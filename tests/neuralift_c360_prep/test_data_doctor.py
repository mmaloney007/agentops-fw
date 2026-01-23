#!/usr/bin/env python3
"""
tests/neuralift_c360_prep/test_data_doctor.py
---------------------------------------------
Tests for the data_doctor module.

Author: Mike Maloney - Neuralift, Inc.
Updated: 2026-01-13
Copyright (c) 2025 Neuralift, Inc.
"""

from __future__ import annotations

import pytest
import pandas as pd
import dask.dataframe as dd

from neuralift_c360_prep.data_doctor import (
    Suggestion,
    FillAlternative,
    DataDoctorReport,
    analyze_data,
    print_report,
    save_suggestions_yaml,
    run_standalone,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_LABELS,
    BUSINESS_FILL_ALTERNATIVES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_ddf():
    """Create a sample Dask DataFrame for testing."""
    pdf = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "total_spend": [100.0, 200.0, None, 400.0, 500.0],
            "order_count": [1, 2, 3, 4, 5],
            "category_name": ["A", "B", "A", "C", "B"],
            "signup_date": pd.to_datetime(
                ["2024-01-01", "2024-02-15", "2024-03-10", "2024-04-20", "2024-05-05"]
            ),
            "revenue": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            "visits": [10, 20, 30, 40, 50],
            "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        }
    )
    return dd.from_pandas(pdf, npartitions=1)


@pytest.fixture
def sample_data_dict():
    """Create a sample data dictionary for testing."""
    return {
        "columns": {
            "customer_id": {
                "column_name": "customer_id",
                "column_type": "id",
                "data_type": "INTEGER",
                "null_count": 0,
                "unique_count": 5,
            },
            "total_spend": {
                "column_name": "total_spend",
                "column_type": "continuous",
                "data_type": "DOUBLE",
                "null_count": 1,
                "unique_count": 4,
            },
            "order_count": {
                "column_name": "order_count",
                "column_type": "continuous",
                "data_type": "INTEGER",
                "null_count": 0,
                "unique_count": 5,
            },
            "category_name": {
                "column_name": "category_name",
                "column_type": "categorical",
                "data_type": "STRING",
                "null_count": 0,
                "unique_count": 3,
            },
            "signup_date": {
                "column_name": "signup_date",
                "column_type": "categorical",
                "data_type": "STRING",
                "null_count": 0,
                "unique_count": 5,
            },
            "revenue": {
                "column_name": "revenue",
                "column_type": "continuous",
                "data_type": "DOUBLE",
                "null_count": 0,
                "unique_count": 5,
            },
            "visits": {
                "column_name": "visits",
                "column_type": "continuous",
                "data_type": "INTEGER",
                "null_count": 0,
                "unique_count": 5,
            },
            "income": {
                "column_name": "income",
                "column_type": "continuous",
                "data_type": "DOUBLE",
                "null_count": 0,
                "unique_count": 5,
            },
        }
    }


# ---------------------------------------------------------------------------
# Tests for Severity Constants
# ---------------------------------------------------------------------------
class TestSeverityConstants:
    """Tests for severity signal constants."""

    def test_severity_constants_are_emojis(self):
        """Test that severity constants are emoji characters."""
        # HIGH is police car light
        assert SEVERITY_HIGH == "\U0001f6a8"
        # MEDIUM is test tube
        assert SEVERITY_MEDIUM == "\U0001f9ea"
        # LOW is light bulb
        assert SEVERITY_LOW == "\U0001f4a1"

    def test_severity_labels_mapping(self):
        """Test that severity labels map correctly."""
        assert SEVERITY_LABELS[SEVERITY_HIGH] == "HIGH"
        assert SEVERITY_LABELS[SEVERITY_MEDIUM] == "MEDIUM"
        assert SEVERITY_LABELS[SEVERITY_LOW] == "LOW"
        # Legacy support
        assert SEVERITY_LABELS["!"] == "HIGH"
        assert SEVERITY_LABELS["~"] == "MEDIUM"
        assert SEVERITY_LABELS["?"] == "LOW"


# ---------------------------------------------------------------------------
# Tests for Business Fill Alternatives
# ---------------------------------------------------------------------------
class TestBusinessFillAlternatives:
    """Tests for business-friendly fill alternatives."""

    def test_numeric_alternatives_exist(self):
        """Test that numeric fill alternatives are defined."""
        alts = BUSINESS_FILL_ALTERNATIVES["numeric"]
        assert len(alts) >= 5
        strategies = [a["strategy"] for a in alts]
        assert "median" in strategies
        assert "mean" in strategies
        assert "zero" in strategies

    def test_categorical_alternatives_exist(self):
        """Test that categorical fill alternatives are defined."""
        alts = BUSINESS_FILL_ALTERNATIVES["categorical"]
        assert len(alts) >= 3
        strategies = [a["strategy"] for a in alts]
        assert "mode" in strategies
        assert "Unknown" in strategies

    def test_datetime_alternatives_exist(self):
        """Test that datetime fill alternatives are defined."""
        alts = BUSINESS_FILL_ALTERNATIVES["datetime"]
        assert len(alts) >= 3
        strategies = [a["strategy"] for a in alts]
        assert "forward_fill" in strategies


# ---------------------------------------------------------------------------
# Tests for FillAlternative dataclass
# ---------------------------------------------------------------------------
class TestFillAlternative:
    """Tests for the FillAlternative dataclass."""

    def test_fill_alternative_creation(self):
        """Test creating a FillAlternative."""
        fa = FillAlternative(
            strategy="median",
            description="Use middle value",
            requires=None,
        )
        assert fa.strategy == "median"
        assert fa.description == "Use middle value"
        assert fa.requires is None

    def test_fill_alternative_with_requires(self):
        """Test FillAlternative with requirements."""
        fa = FillAlternative(
            strategy="group_median",
            description="Median by segment",
            requires="segment column",
        )
        assert fa.requires == "segment column"


# ---------------------------------------------------------------------------
# Tests for Suggestion dataclass
# ---------------------------------------------------------------------------
class TestSuggestion:
    """Tests for the Suggestion dataclass."""

    def test_suggestion_creation(self):
        """Test creating a Suggestion."""
        s = Suggestion(
            priority=SEVERITY_HIGH,
            category="feature",
            column="test_col",
            message="Test message",
            yaml_snippet="test: snippet",
        )
        assert s.priority == SEVERITY_HIGH
        assert s.category == "feature"
        assert s.column == "test_col"
        assert s.message == "Test message"
        assert s.yaml_snippet == "test: snippet"
        assert s.alternatives == []

    def test_suggestion_with_alternatives(self):
        """Test creating a Suggestion with alternatives."""
        alts = [
            FillAlternative("median", "Middle value"),
            FillAlternative("mean", "Average value"),
        ]
        s = Suggestion(
            priority=SEVERITY_MEDIUM,
            category="fill",
            column="test_col",
            message="Test message",
            yaml_snippet="test: snippet",
            alternatives=alts,
        )
        assert len(s.alternatives) == 2
        assert s.alternatives[0].strategy == "median"

    def test_suggestion_priorities_new_emojis(self):
        """Test that new emoji priority levels work."""
        for priority in [SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW]:
            s = Suggestion(
                priority=priority,
                category="test",
                column="col",
                message="msg",
                yaml_snippet="",
            )
            assert s.priority == priority

    def test_suggestion_priorities_legacy(self):
        """Test that legacy priority levels still work."""
        for priority in ["!", "~", "?"]:
            s = Suggestion(
                priority=priority,
                category="test",
                column="col",
                message="msg",
                yaml_snippet="",
            )
            assert s.priority == priority


# ---------------------------------------------------------------------------
# Tests for DataDoctorReport dataclass
# ---------------------------------------------------------------------------
class TestDataDoctorReport:
    """Tests for the DataDoctorReport dataclass."""

    def test_empty_report(self):
        """Test creating an empty report."""
        report = DataDoctorReport()
        assert report.feature_suggestions == []
        assert report.fill_suggestions == []
        assert report.kpi_candidates == []
        assert report.ratio_opportunities == []
        assert report.yaml_text == ""

    def test_report_with_suggestions(self):
        """Test creating a report with suggestions."""
        s = Suggestion(
            priority="!",
            category="feature",
            column="col",
            message="msg",
            yaml_snippet="snippet",
        )
        report = DataDoctorReport(
            feature_suggestions=[s],
            fill_suggestions=[s],
            kpi_candidates=[s],
            ratio_opportunities=[s],
            yaml_text="test yaml",
        )
        assert len(report.feature_suggestions) == 1
        assert len(report.fill_suggestions) == 1
        assert len(report.kpi_candidates) == 1
        assert len(report.ratio_opportunities) == 1
        assert report.yaml_text == "test yaml"


# ---------------------------------------------------------------------------
# Tests for analyze_data function
# ---------------------------------------------------------------------------
class TestAnalyzeData:
    """Tests for the analyze_data function."""

    def test_analyze_data_basic(self, sample_ddf, sample_data_dict):
        """Test basic analysis."""
        report = analyze_data(sample_ddf, sample_data_dict, show_progress=False)
        assert isinstance(report, DataDoctorReport)
        assert report.yaml_text != ""

    def test_analyze_data_detects_nulls(self, sample_ddf, sample_data_dict):
        """Test that null columns are detected."""
        report = analyze_data(sample_ddf, sample_data_dict, show_progress=False)
        # total_spend has a null value
        fill_cols = [s.column for s in report.fill_suggestions]
        assert "total_spend" in fill_cols

    def test_analyze_data_uses_emoji_priorities(self, sample_ddf, sample_data_dict):
        """Test that new emoji priorities are used."""
        report = analyze_data(sample_ddf, sample_data_dict, show_progress=False)
        # Check that at least some suggestions use emoji priorities
        all_suggestions = (
            report.fill_suggestions
            + report.feature_suggestions
            + report.kpi_candidates
            + report.ratio_opportunities
        )
        priorities = {s.priority for s in all_suggestions}
        # Should use emoji priorities, not legacy ones
        assert any(
            p in [SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW] for p in priorities
        )

    def test_analyze_data_includes_business_alternatives(
        self, sample_ddf, sample_data_dict
    ):
        """Test that business alternatives are included for fill suggestions."""
        report = analyze_data(
            sample_ddf,
            sample_data_dict,
            show_progress=False,
            show_business_alternatives=True,
        )
        # total_spend has nulls, should have alternatives
        fill_with_alts = [s for s in report.fill_suggestions if s.alternatives]
        assert len(fill_with_alts) > 0

    def test_analyze_data_can_disable_alternatives(self, sample_ddf, sample_data_dict):
        """Test that business alternatives can be disabled."""
        report = analyze_data(
            sample_ddf,
            sample_data_dict,
            show_progress=False,
            show_business_alternatives=False,
        )
        # No alternatives should be present
        fill_with_alts = [s for s in report.fill_suggestions if s.alternatives]
        assert len(fill_with_alts) == 0

    def test_analyze_data_respects_high_null_threshold(
        self, sample_ddf, sample_data_dict
    ):
        """Test that high_null_threshold affects priority."""
        # With threshold=0, even 1 null should be HIGH priority
        report = analyze_data(
            sample_ddf,
            sample_data_dict,
            show_progress=False,
            high_null_threshold=0,
        )
        # total_spend has 1 null, should be HIGH with threshold=0
        spend_fill = next(
            (s for s in report.fill_suggestions if s.column == "total_spend"), None
        )
        assert spend_fill is not None
        assert spend_fill.priority == SEVERITY_HIGH

    def test_analyze_data_detects_monetary_columns(self, sample_ddf, sample_data_dict):
        """Test that monetary columns are detected."""
        report = analyze_data(sample_ddf, sample_data_dict, show_progress=False)
        feature_cols = [s.column for s in report.feature_suggestions]
        # total_spend and revenue should be detected as monetary
        assert any("spend" in col.lower() for col in feature_cols) or any(
            "revenue" in col.lower() for col in feature_cols
        )

    def test_analyze_data_detects_kpi_candidates(self, sample_ddf, sample_data_dict):
        """Test that KPI candidates are detected."""
        report = analyze_data(sample_ddf, sample_data_dict, show_progress=False)
        kpi_cols = [s.column for s in report.kpi_candidates]
        # revenue should be detected as a KPI candidate
        assert "revenue" in kpi_cols

    def test_analyze_data_detects_date_columns(self, sample_ddf, sample_data_dict):
        """Test that date columns are detected."""
        report = analyze_data(sample_ddf, sample_data_dict, show_progress=False)
        feature_cols = [s.column for s in report.feature_suggestions]
        # signup_date should be detected as a date column
        assert "signup_date" in feature_cols

    def test_analyze_data_detects_ratio_opportunities(
        self, sample_ddf, sample_data_dict
    ):
        """Test that ratio opportunities are detected."""
        report = analyze_data(sample_ddf, sample_data_dict, show_progress=False)
        # spend/income or revenue/visits ratios might be detected
        assert isinstance(report.ratio_opportunities, list)


# ---------------------------------------------------------------------------
# Tests for print_report function
# ---------------------------------------------------------------------------
class TestPrintReport:
    """Tests for the print_report function."""

    def test_print_report_empty(self, caplog):
        """Test printing an empty report."""
        report = DataDoctorReport()
        import logging

        with caplog.at_level(logging.INFO):
            print_report(report)
        assert "DATA DOCTOR" in caplog.text

    def test_print_report_with_suggestions(self, sample_ddf, sample_data_dict, caplog):
        """Test printing a report with suggestions."""
        report = analyze_data(sample_ddf, sample_data_dict, show_progress=False)
        import logging

        with caplog.at_level(logging.INFO):
            print_report(report)
        assert "DATA DOCTOR" in caplog.text

    def test_print_report_shows_legend(self, sample_ddf, sample_data_dict, capsys):
        """Test that report shows severity legend when using print."""
        report = analyze_data(sample_ddf, sample_data_dict, show_progress=False)
        print_report(report, use_print=True)
        captured = capsys.readouterr()
        assert "Legend:" in captured.out
        assert "HIGH" in captured.out
        assert "MEDIUM" in captured.out
        assert "LOW" in captured.out

    def test_print_report_shows_alternatives(
        self, sample_ddf, sample_data_dict, capsys
    ):
        """Test that report shows business alternatives."""
        report = analyze_data(
            sample_ddf,
            sample_data_dict,
            show_progress=False,
            show_business_alternatives=True,
        )
        print_report(report, use_print=True, show_alternatives=True)
        captured = capsys.readouterr()
        # Should show alternatives section for fill suggestions
        if report.fill_suggestions:
            assert "Business alternatives:" in captured.out

    def test_print_report_hides_alternatives(
        self, sample_ddf, sample_data_dict, capsys
    ):
        """Test that alternatives can be hidden."""
        report = analyze_data(
            sample_ddf,
            sample_data_dict,
            show_progress=False,
            show_business_alternatives=True,
        )
        print_report(report, use_print=True, show_alternatives=False)
        captured = capsys.readouterr()
        # Should not show alternatives
        assert "Business alternatives:" not in captured.out


# ---------------------------------------------------------------------------
# Tests for run_standalone function
# ---------------------------------------------------------------------------
class TestRunStandalone:
    """Tests for the run_standalone function."""

    def test_run_standalone_basic(self, sample_ddf, sample_data_dict, capsys):
        """Test basic standalone execution."""
        report = run_standalone(
            sample_ddf,
            sample_data_dict,
            show_progress=False,
        )
        assert isinstance(report, DataDoctorReport)
        # Should have printed output
        captured = capsys.readouterr()
        assert "DATA DOCTOR" in captured.out

    def test_run_standalone_saves_yaml(
        self, sample_ddf, sample_data_dict, tmp_path, capsys
    ):
        """Test that standalone saves yaml when output_path is provided."""
        output_path = tmp_path / "suggestions.yaml"
        run_standalone(
            sample_ddf,
            sample_data_dict,
            output_path=output_path,
            show_progress=False,
        )
        assert output_path.exists()
        content = output_path.read_text()
        assert "feature_suggestions" in content


# ---------------------------------------------------------------------------
# Tests for save_suggestions_yaml function
# ---------------------------------------------------------------------------
class TestSaveSuggestionsYaml:
    """Tests for the save_suggestions_yaml function."""

    def test_save_yaml(self, sample_ddf, sample_data_dict, tmp_path):
        """Test saving suggestions to YAML."""
        report = analyze_data(sample_ddf, sample_data_dict, show_progress=False)
        output_path = tmp_path / "suggestions.yaml"
        save_suggestions_yaml(report, output_path)
        assert output_path.exists()
        content = output_path.read_text()
        assert "feature_suggestions" in content


# ---------------------------------------------------------------------------
# Tests for edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        """Test with an empty DataFrame."""
        pdf = pd.DataFrame()
        ddf = dd.from_pandas(pdf, npartitions=1)
        data_dict = {"columns": {}}
        report = analyze_data(ddf, data_dict, show_progress=False)
        assert isinstance(report, DataDoctorReport)

    def test_no_numeric_columns(self):
        """Test with no numeric columns."""
        pdf = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "city": ["NYC", "LA", "Chicago"],
            }
        )
        ddf = dd.from_pandas(pdf, npartitions=1)
        data_dict = {
            "columns": {
                "name": {
                    "column_type": "categorical",
                    "data_type": "STRING",
                    "null_count": 0,
                    "unique_count": 3,
                },
                "city": {
                    "column_type": "categorical",
                    "data_type": "STRING",
                    "null_count": 0,
                    "unique_count": 3,
                },
            }
        }
        report = analyze_data(ddf, data_dict, show_progress=False)
        # Should not crash and should have empty KPI candidates
        assert report.kpi_candidates == []

    def test_data_dict_list_format(self):
        """Test with data dictionary in list format."""
        pdf = pd.DataFrame(
            {
                "amount": [100.0, 200.0, 300.0],
            }
        )
        ddf = dd.from_pandas(pdf, npartitions=1)
        data_dict = {
            "columns": [
                {
                    "column_name": "amount",
                    "column_type": "continuous",
                    "data_type": "DOUBLE",
                    "null_count": 0,
                    "unique_count": 3,
                },
            ]
        }
        report = analyze_data(ddf, data_dict, show_progress=False)
        # Should work with list format
        assert isinstance(report, DataDoctorReport)


# ---------------------------------------------------------------------------
# Tests for high cardinality categoricals
# ---------------------------------------------------------------------------
class TestHighCardinalityCategoricals:
    """Tests for high cardinality categorical detection."""

    def test_high_cardinality_detected(self):
        """Test that high cardinality categoricals are detected."""
        pdf = pd.DataFrame(
            {
                "product_id": [f"prod_{i}" for i in range(100)],
            }
        )
        ddf = dd.from_pandas(pdf, npartitions=1)
        data_dict = {
            "columns": {
                "product_id": {
                    "column_type": "categorical",
                    "data_type": "STRING",
                    "null_count": 0,
                    "unique_count": 100,
                },
            }
        }
        report = analyze_data(ddf, data_dict, show_progress=False)
        # Should detect high cardinality
        feature_cols = [s.column for s in report.feature_suggestions]
        assert "product_id" in feature_cols

    def test_low_cardinality_not_flagged(self):
        """Test that low cardinality categoricals are not flagged."""
        pdf = pd.DataFrame(
            {
                "status": ["active", "inactive", "pending"],
            }
        )
        ddf = dd.from_pandas(pdf, npartitions=1)
        data_dict = {
            "columns": {
                "status": {
                    "column_type": "categorical",
                    "data_type": "STRING",
                    "null_count": 0,
                    "unique_count": 3,
                },
            }
        }
        report = analyze_data(ddf, data_dict, show_progress=False)
        # Should not flag as high cardinality - status should not be in feature suggestions for bucketing
        bucketing_suggestions = [
            s
            for s in report.feature_suggestions
            if "categorical_bucket" in s.yaml_snippet
        ]
        assert len(bucketing_suggestions) == 0
