"""Power measurement harness wrapping macOS powermetrics.

Captures CPU, GPU, and ANE power draw during inference or training windows.
Gracefully degrades when powermetrics is unavailable (CI, non-root) -- warns
and returns zero-valued summaries, never crashes.

Usage::

    mon = PowerMonitor(interval_ms=100)
    mon.start()
    # ... do inference or training ...
    summary = mon.stop()
    print(summary.mean_cpu_w, summary.mean_gpu_w, summary.energy_j)
"""

from __future__ import annotations

import plistlib
import shutil
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PowerSample:
    """A single power measurement snapshot."""

    timestamp: float
    cpu_w: float
    gpu_w: float
    ane_w: float
    total_w: float


@dataclass
class PowerSummary:
    """Aggregated power statistics over a measurement window."""

    samples: List[PowerSample]
    mean_cpu_w: float
    mean_gpu_w: float
    mean_ane_w: float
    mean_total_w: float
    duration_s: float
    energy_j: float

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict with rounded values."""
        return {
            "mean_cpu_w": round(self.mean_cpu_w, 3),
            "mean_gpu_w": round(self.mean_gpu_w, 3),
            "mean_ane_w": round(self.mean_ane_w, 3),
            "mean_total_w": round(self.mean_total_w, 3),
            "duration_s": round(self.duration_s, 3),
            "energy_j": round(self.energy_j, 3),
            "num_samples": len(self.samples),
        }


def _zero_summary() -> PowerSummary:
    """Return a zero-valued summary for cases where no data was collected."""
    return PowerSummary(
        samples=[],
        mean_cpu_w=0.0,
        mean_gpu_w=0.0,
        mean_ane_w=0.0,
        mean_total_w=0.0,
        duration_s=0.0,
        energy_j=0.0,
    )


class PowerMonitor:
    """Captures CPU/GPU/ANE power draw via macOS ``powermetrics``.

    Parameters
    ----------
    interval_ms : int
        Sampling interval in milliseconds (default 100).
    sudo_password : str or None
        Not used directly -- powermetrics requires ``sudo -n`` (passwordless)
        or the process must already be running as root.
    """

    def __init__(self, interval_ms: int = 100, sudo_password: Optional[str] = None):
        self._interval_ms = interval_ms
        self._sudo_password = sudo_password
        self._samples: List[PowerSample] = []
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch powermetrics in background and begin collecting samples.

        Prints a warning and returns gracefully if powermetrics is not
        available or cannot be started (e.g., no sudo access).
        """
        if shutil.which("powermetrics") is None:
            warnings.warn(
                "powermetrics not found -- power monitoring disabled. "
                "Install Xcode Command Line Tools or run on macOS.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        cmd = [
            "sudo", "-n", "powermetrics",
            "--samplers", "cpu_power,gpu_power",
            "-i", str(self._interval_ms),
            "--format", "plist",
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, PermissionError) as exc:
            warnings.warn(
                f"Failed to start powermetrics: {exc} -- power monitoring disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._reader_thread = threading.Thread(
            target=self._reader, daemon=True, name="power-reader"
        )
        self._reader_thread.start()

    def stop(self) -> PowerSummary:
        """Stop collection, terminate powermetrics, and return aggregated results.

        Returns a zero-valued summary if no samples were collected (e.g.,
        powermetrics was unavailable).
        """
        self._stop_event.set()

        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5)

        with self._lock:
            samples = list(self._samples)

        if not samples:
            return _zero_summary()

        mean_cpu = sum(s.cpu_w for s in samples) / len(samples)
        mean_gpu = sum(s.gpu_w for s in samples) / len(samples)
        mean_ane = sum(s.ane_w for s in samples) / len(samples)
        mean_total = sum(s.total_w for s in samples) / len(samples)

        duration = samples[-1].timestamp - samples[0].timestamp
        if duration <= 0 and self._start_time is not None:
            duration = time.monotonic() - self._start_time

        energy = mean_total * duration  # joules = watts * seconds

        return PowerSummary(
            samples=samples,
            mean_cpu_w=mean_cpu,
            mean_gpu_w=mean_gpu,
            mean_ane_w=mean_ane,
            mean_total_w=mean_total,
            duration_s=duration,
            energy_j=energy,
        )

    # ------------------------------------------------------------------
    # Plist parsing
    # ------------------------------------------------------------------

    def _parse_plist_sample(self, data: dict) -> PowerSample:
        """Extract power values from a single parsed plist dict.

        CPU power is the sum of all ``processor.clusters[*].hw_power``.
        GPU power comes from ``gpu.hw_power``.
        ANE power comes from ``ane.hw_power`` (0.0 if absent).
        """
        # CPU: sum across all clusters
        cpu_w = 0.0
        proc = data.get("processor", {})
        for cluster in proc.get("clusters", []):
            cpu_w += cluster.get("hw_power", 0.0)

        # GPU
        gpu_w = data.get("gpu", {}).get("hw_power", 0.0)

        # ANE (may not be present on all hardware)
        ane_w = data.get("ane", {}).get("hw_power", 0.0)

        total_w = cpu_w + gpu_w + ane_w

        return PowerSample(
            timestamp=time.monotonic(),
            cpu_w=cpu_w,
            gpu_w=gpu_w,
            ane_w=ane_w,
            total_w=total_w,
        )

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------

    def _reader(self) -> None:
        """Background thread: reads plist blocks from powermetrics stdout."""
        buf = b""
        end_tag = b"</plist>"

        while not self._stop_event.is_set():
            if self._process is None or self._process.stdout is None:
                break

            try:
                chunk = self._process.stdout.read1(4096)  # type: ignore[attr-defined]
            except Exception:
                break

            if not chunk:
                break

            buf += chunk

            # Process all complete plist documents in the buffer
            while end_tag in buf:
                idx = buf.index(end_tag) + len(end_tag)
                plist_block = buf[:idx]
                buf = buf[idx:].lstrip()

                try:
                    data = plistlib.loads(plist_block)
                    sample = self._parse_plist_sample(data)
                    with self._lock:
                        self._samples.append(sample)
                except Exception:
                    # Malformed plist block -- skip
                    continue
