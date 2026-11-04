"""
=============================================================================
PERCLOS (Percentage of Eye Closure) Calculation Module
=============================================================================
Implements real-time timestamp-based rolling window calculation of PERCLOS.
PERCLOS is recognized by NHTSA and ISO standards as one of the most reliable
ocular indicators of slow eyelid closure.

Formula:
    PERCLOS = ( Total Valid Time Eyes Were Closed ) / ( Total Valid Observation Time ) * 100 %

Handles frame rate variations and periods of temporary face occlusion without
biasing the calculation.
=============================================================================
"""

import collections
from dataclasses import dataclass
from typing import Deque, Optional
import config


@dataclass
class EyeObservation:
    timestamp: float          # Unix/monotonic timestamp in seconds
    is_closed: bool           # Whether eyes were closed during this sample
    face_detected: bool       # Whether a valid face was present


class PERCLOSTracker:
    """
    Tracks eye closure over a configurable rolling time window (e.g. 60 seconds)
    using precise timestamps to calculate time-weighted PERCLOS.
    """

    def __init__(self, window_seconds: float = config.PERCLOS_WINDOW_SECONDS):
        self.window_seconds = float(window_seconds)
        self.history: Deque[EyeObservation] = collections.deque()
        self._current_perclos = 0.0

    def update(self, timestamp: float, is_closed: bool, face_detected: bool) -> float:
        """
        Appends a new observation sample and updates the rolling PERCLOS value.

        Args:
            timestamp: Precise current timestamp (seconds).
            is_closed: True if eyes are classified as closed in current frame.
            face_detected: True if face landmarks are present and valid.

        Returns:
            float: Current PERCLOS percentage [0.0 - 100.0%].
        """
        # Append current sample
        self.history.append(EyeObservation(
            timestamp=timestamp,
            is_closed=is_closed,
            face_detected=face_detected
        ))

        # Evict samples older than rolling window
        cutoff_time = timestamp - self.window_seconds
        while len(self.history) > 1 and self.history[0].timestamp < cutoff_time:
            self.history.popleft()

        # If less than 2 samples, cannot calculate interval dt
        if len(self.history) < 2:
            self._current_perclos = 0.0
            return 0.0

        total_valid_duration = 0.0
        total_closed_duration = 0.0

        # Integrate time intervals between consecutive samples
        # Sample state applies to the duration leading up to it
        for i in range(1, len(self.history)):
            prev_sample = self.history[i - 1]
            curr_sample = self.history[i]

            dt = curr_sample.timestamp - prev_sample.timestamp
            # Guard against clock anomalies or huge gaps (e.g. paused camera)
            if dt <= 0.0 or dt > 2.0:
                continue

            # Only accumulate if face was valid
            if curr_sample.face_detected and prev_sample.face_detected:
                total_valid_duration += dt
                if curr_sample.is_closed:
                    total_closed_duration += dt

        if total_valid_duration > 0.5:
            self._current_perclos = (total_closed_duration / total_valid_duration) * 100.0
        else:
            self._current_perclos = 0.0

        return self._current_perclos

    def get_perclos(self) -> float:
        """Returns the current rolling PERCLOS percentage."""
        return self._current_perclos

    def reset(self) -> None:
        """Clears the observation history buffer."""
        self.history.clear()
        self._current_perclos = 0.0
