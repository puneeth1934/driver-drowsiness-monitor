"""
=============================================================================
Feature Logger Module
=============================================================================
Handles automatic, rate-limited CSV logging of all driver monitoring metrics.
Outputs clean time-series data compatible with pandas, matplotlib, and future
sensor synchronization with ESP32 telemetry (MAX30102, MPU6050, AS5600).
=============================================================================
"""

import os
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import config


class FeatureLogger:
    """
    Asynchronously or rate-limited writes computed facial features to CSV.
    """

    HEADERS = [
        "timestamp",
        "elapsed_time",
        "face_detected",
        "left_ear",
        "right_ear",
        "average_ear",
        "eye_status",
        "eye_closed_duration",
        "blink_event",
        "blink_count",
        "blink_rate",
        "perclos",
        "mouth_ratio",
        "mouth_status",
        "yawn_event",
        "yawn_count",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "fps"
    ]

    def __init__(
        self,
        csv_path: Path = config.CSV_LOG_PATH,
        frequency_hz: float = config.LOGGING_FREQUENCY_HZ,
        enabled: bool = config.LOG_TO_CSV
    ):
        self.csv_path = Path(csv_path)
        self.frequency_hz = frequency_hz
        self.min_interval = 1.0 / frequency_hz if frequency_hz > 0 else 0.1
        self.enabled = enabled

        self._last_log_time = 0.0
        self._start_time = time.time()
        self._file = None
        self._writer = None

        if self.enabled:
            self._init_csv()

    def _init_csv(self) -> None:
        """Initializes the CSV file and writes header if not present."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0

        self._file = open(self.csv_path, mode="a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADERS)

        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def log(self, metrics: Dict[str, Any], force: bool = False) -> None:
        """
        Logs a row of driver metrics to CSV if the sampling interval has elapsed
        or if an instantaneous event (e.g. blink or yawn) occurred.

        Args:
            metrics: Dictionary containing all feature values.
            force: Force writing regardless of rate limiter (used for discrete events).
        """
        if not self.enabled or self._writer is None:
            return

        now = time.time()
        # Always log discrete events immediately (blink_event=1 or yawn_event=1)
        has_event = bool(metrics.get("blink_event", 0)) or bool(metrics.get("yawn_event", 0))

        if not force and not has_event and (now - self._last_log_time) < self.min_interval:
            return

        self._last_log_time = now
        elapsed = now - self._start_time
        iso_ts = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        row = {
            "timestamp": iso_ts,
            "elapsed_time": f"{elapsed:.3f}",
            "face_detected": int(bool(metrics.get("face_detected", False))),
            "left_ear": f"{float(metrics.get('left_ear', 0.0)):.4f}",
            "right_ear": f"{float(metrics.get('right_ear', 0.0)):.4f}",
            "average_ear": f"{float(metrics.get('average_ear', 0.0)):.4f}",
            "eye_status": str(metrics.get("eye_status", "UNKNOWN")),
            "eye_closed_duration": f"{float(metrics.get('eye_closed_duration', 0.0)):.3f}",
            "blink_event": int(bool(metrics.get("blink_event", 0))),
            "blink_count": int(metrics.get("blink_count", 0)),
            "blink_rate": f"{float(metrics.get('blink_rate', 0.0)):.2f}",
            "perclos": f"{float(metrics.get('perclos', 0.0)):.2f}",
            "mouth_ratio": f"{float(metrics.get('mouth_ratio', 0.0)):.4f}",
            "mouth_status": str(metrics.get("mouth_status", "UNKNOWN")),
            "yawn_event": int(bool(metrics.get("yawn_event", 0))),
            "yawn_count": int(metrics.get("yawn_count", 0)),
            "head_pitch": f"{float(metrics.get('head_pitch', 0.0)):.2f}",
            "head_yaw": f"{float(metrics.get('head_yaw', 0.0)):.2f}",
            "head_roll": f"{float(metrics.get('head_roll', 0.0)):.2f}",
            "fps": f"{float(metrics.get('fps', 0.0)):.1f}"
        }

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Flushes and closes the CSV file."""
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None
