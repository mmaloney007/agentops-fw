"""Tests for agent_stable_slo.bench.power_monitor.

Covers: PowerSample, PowerSummary (to_dict), PowerMonitor start/stop,
_parse_plist_sample extraction, and graceful fallback when powermetrics
is unavailable.
"""
import time
import plistlib
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from agent_stable_slo.bench.power_monitor import (
    PowerSample,
    PowerSummary,
    PowerMonitor,
)


# ---- PowerSample ----

class TestPowerSample:
    def test_fields(self):
        s = PowerSample(timestamp=1000.0, cpu_w=5.0, gpu_w=3.0, ane_w=1.0, total_w=9.0)
        assert s.timestamp == 1000.0
        assert s.cpu_w == 5.0
        assert s.gpu_w == 3.0
        assert s.ane_w == 1.0
        assert s.total_w == 9.0

    def test_total_independent_of_components(self):
        """total_w is stored as-is, not computed from cpu+gpu+ane."""
        s = PowerSample(timestamp=0.0, cpu_w=1.0, gpu_w=2.0, ane_w=3.0, total_w=99.0)
        assert s.total_w == 99.0


# ---- PowerSummary ----

class TestPowerSummary:
    def _make_summary(self):
        samples = [
            PowerSample(timestamp=0.0, cpu_w=10.0, gpu_w=4.0, ane_w=2.0, total_w=16.0),
            PowerSample(timestamp=0.1, cpu_w=12.0, gpu_w=6.0, ane_w=3.0, total_w=21.0),
        ]
        return PowerSummary(
            samples=samples,
            mean_cpu_w=11.0,
            mean_gpu_w=5.0,
            mean_ane_w=2.5,
            mean_total_w=18.5,
            duration_s=0.1,
            energy_j=1.85,
        )

    def test_to_dict_keys(self):
        d = self._make_summary().to_dict()
        expected_keys = {
            "mean_cpu_w", "mean_gpu_w", "mean_ane_w", "mean_total_w",
            "duration_s", "energy_j", "num_samples",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values_rounded(self):
        d = self._make_summary().to_dict()
        assert d["mean_cpu_w"] == 11.0
        assert d["mean_gpu_w"] == 5.0
        assert d["mean_ane_w"] == 2.5
        assert d["mean_total_w"] == 18.5
        assert d["duration_s"] == 0.1
        assert d["energy_j"] == 1.85
        assert d["num_samples"] == 2

    def test_to_dict_rounds_long_floats(self):
        summary = PowerSummary(
            samples=[],
            mean_cpu_w=10.123456789,
            mean_gpu_w=5.987654321,
            mean_ane_w=2.111111111,
            mean_total_w=18.222222222,
            duration_s=1.999999999,
            energy_j=3.14159265,
        )
        d = summary.to_dict()
        # All float values should be rounded to 3 decimal places
        assert d["mean_cpu_w"] == 10.123
        assert d["mean_gpu_w"] == 5.988
        assert d["mean_ane_w"] == 2.111
        assert d["mean_total_w"] == 18.222
        assert d["duration_s"] == 2.0
        assert d["energy_j"] == 3.142


# ---- PowerMonitor._parse_plist_sample ----

class TestParsePlistSample:
    def test_extracts_cpu_gpu_ane(self):
        data = {
            "processor": {
                "clusters": [
                    {"hw_power": 3.5},
                    {"hw_power": 2.0},
                ]
            },
            "gpu": {"hw_power": 8.0},
            "ane": {"hw_power": 1.5},
        }
        mon = PowerMonitor(interval_ms=100)
        sample = mon._parse_plist_sample(data)

        assert sample.cpu_w == pytest.approx(5.5)  # 3.5 + 2.0
        assert sample.gpu_w == pytest.approx(8.0)
        assert sample.ane_w == pytest.approx(1.5)
        assert sample.total_w == pytest.approx(15.0)  # 5.5 + 8.0 + 1.5

    def test_missing_ane_defaults_zero(self):
        data = {
            "processor": {
                "clusters": [{"hw_power": 4.0}]
            },
            "gpu": {"hw_power": 6.0},
        }
        mon = PowerMonitor(interval_ms=100)
        sample = mon._parse_plist_sample(data)

        assert sample.ane_w == 0.0
        assert sample.total_w == pytest.approx(10.0)

    def test_missing_gpu_defaults_zero(self):
        data = {
            "processor": {
                "clusters": [{"hw_power": 4.0}]
            },
        }
        mon = PowerMonitor(interval_ms=100)
        sample = mon._parse_plist_sample(data)

        assert sample.gpu_w == 0.0
        assert sample.ane_w == 0.0
        assert sample.cpu_w == pytest.approx(4.0)
        assert sample.total_w == pytest.approx(4.0)

    def test_empty_clusters(self):
        data = {
            "processor": {"clusters": []},
            "gpu": {"hw_power": 2.0},
        }
        mon = PowerMonitor(interval_ms=100)
        sample = mon._parse_plist_sample(data)

        assert sample.cpu_w == 0.0
        assert sample.gpu_w == 2.0

    def test_no_processor_key(self):
        data = {"gpu": {"hw_power": 5.0}}
        mon = PowerMonitor(interval_ms=100)
        sample = mon._parse_plist_sample(data)

        assert sample.cpu_w == 0.0
        assert sample.gpu_w == 5.0


# ---- PowerMonitor start/stop with mocked subprocess ----

def _build_plist_bytes(cpu_cluster_powers, gpu_power, ane_power=None):
    """Build a plist XML bytes block that powermetrics would emit."""
    data = {
        "processor": {
            "clusters": [{"hw_power": p} for p in cpu_cluster_powers]
        },
        "gpu": {"hw_power": gpu_power},
    }
    if ane_power is not None:
        data["ane"] = {"hw_power": ane_power}
    return plistlib.dumps(data)


class TestPowerMonitorStartStop:
    @patch("agent_stable_slo.bench.power_monitor.subprocess.Popen")
    @patch("agent_stable_slo.bench.power_monitor.shutil.which", return_value="/usr/bin/powermetrics")
    def test_start_stop_returns_valid_summary(self, mock_which, mock_popen):
        """Mocked powermetrics emits two plist blocks; stop() returns summary."""
        plist1 = _build_plist_bytes([3.0, 2.0], 5.0, 1.0)
        plist2 = _build_plist_bytes([4.0, 1.0], 6.0, 2.0)
        combined = plist1 + b"\n" + plist2

        proc = MagicMock()
        proc.stdout = MagicMock()
        # Simulate reading: first return all bytes, then empty to signal EOF
        proc.stdout.read1 = MagicMock(side_effect=[combined, b""])
        proc.stdout.readable = MagicMock(return_value=True)
        proc.poll = MagicMock(return_value=None)
        mock_popen.return_value = proc

        mon = PowerMonitor(interval_ms=100)
        mon.start()
        # Give reader thread a moment
        time.sleep(0.1)
        summary = mon.stop()

        assert isinstance(summary, PowerSummary)
        assert len(summary.samples) == 2
        # First sample: cpu=5.0, gpu=5.0, ane=1.0
        # Second sample: cpu=5.0, gpu=6.0, ane=2.0
        assert summary.mean_cpu_w == pytest.approx(5.0)
        assert summary.mean_gpu_w == pytest.approx(5.5)
        assert summary.mean_ane_w == pytest.approx(1.5)

    @patch("agent_stable_slo.bench.power_monitor.shutil.which", return_value=None)
    def test_powermetrics_unavailable_returns_zeros(self, mock_which):
        """When powermetrics binary is not found, start() warns and stop() returns zeros."""
        mon = PowerMonitor(interval_ms=100)
        mon.start()
        summary = mon.stop()

        assert isinstance(summary, PowerSummary)
        assert summary.mean_cpu_w == 0.0
        assert summary.mean_gpu_w == 0.0
        assert summary.mean_ane_w == 0.0
        assert summary.mean_total_w == 0.0
        assert summary.energy_j == 0.0
        assert summary.duration_s == 0.0
        assert len(summary.samples) == 0

    @patch("agent_stable_slo.bench.power_monitor.subprocess.Popen")
    @patch("agent_stable_slo.bench.power_monitor.shutil.which", return_value="/usr/bin/powermetrics")
    def test_popen_failure_returns_zeros(self, mock_which, mock_popen):
        """If Popen raises (e.g., permission denied), stop() returns zeros."""
        mock_popen.side_effect = PermissionError("sudo required")

        mon = PowerMonitor(interval_ms=100)
        mon.start()
        summary = mon.stop()

        assert isinstance(summary, PowerSummary)
        assert summary.mean_total_w == 0.0
        assert len(summary.samples) == 0

    def test_stop_without_start_returns_zeros(self):
        """Calling stop() without start() should not crash."""
        mon = PowerMonitor(interval_ms=100)
        summary = mon.stop()

        assert isinstance(summary, PowerSummary)
        assert summary.mean_total_w == 0.0


# ---- PowerMonitor init ----

class TestPowerMonitorInit:
    def test_default_interval(self):
        mon = PowerMonitor()
        assert mon._interval_ms == 100

    def test_custom_interval(self):
        mon = PowerMonitor(interval_ms=500)
        assert mon._interval_ms == 500

    def test_initial_samples_empty(self):
        mon = PowerMonitor()
        assert mon._samples == []
