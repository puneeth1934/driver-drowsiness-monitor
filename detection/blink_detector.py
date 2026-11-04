"""
=============================================================================
Blink & Prolonged Eye Closure Detector Module
=============================================================================
Implements a finite-state machine (FSM) for physiological blink detection
(OPEN -> CLOSING -> CLOSED -> OPEN) and continuous eye-closure duration tracking.
Distinguishes between involuntary rapid blinks and prolonged eyelid closure.
=============================================================================
"""

import time
import collections
from enum import Enum
from dataclasses import dataclass
from typing import Deque, Optional
import config


class EyeState(str, Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


@dataclass
class BlinkState:
    """Encapsulates the current blink and closure state of the driver."""
    eye_status: str               # "OPEN" or "CLOSED"
    blink_event: bool             # True for 1 frame when a valid blink cycle completes
    blink_count: int              # Total number of physiological blinks detected
    blink_rate: float             # Blinks per minute
    last_blink_duration_ms: float # Duration of the most recently finished blink (ms)
    eye_closed_duration: float    # Current continuous closure duration (seconds)
    max_closed_duration: float    # Session maximum continuous closure duration (seconds)
    prolonged_closure: bool       # True if eye_closed_duration >= threshold


class BlinkDetector:
    """
    Temporal State Machine for detecting blinks and monitoring prolonged eye closures.
    """

    def __init__(
        self,
        ear_closed_threshold: float = config.EAR_CLOSED_THRESHOLD,
        ear_open_threshold: float = config.EAR_OPEN_THRESHOLD,
        min_blink_duration_sec: float = config.MIN_BLINK_DURATION_SEC,
        max_blink_duration_sec: float = config.MAX_BLINK_DURATION_SEC,
        prolonged_closure_sec: float = config.PROLONGED_CLOSURE_THRESHOLD_SEC,
        rate_window_sec: float = 60.0
    ):
        self.ear_closed_threshold = ear_closed_threshold
        self.ear_open_threshold = ear_open_threshold
        self.min_blink_duration_sec = min_blink_duration_sec
        self.max_blink_duration_sec = max_blink_duration_sec
        self.prolonged_closure_sec = prolonged_closure_sec
        self.rate_window_sec = rate_window_sec

        # State tracking
        self.state: EyeState = EyeState.OPEN
        self.closure_start_time: Optional[float] = None
        self.current_closed_duration: float = 0.0
        self.max_closed_duration: float = 0.0

        # Blink statistics
        self.blink_count: int = 0
        self.last_blink_duration_ms: float = 0.0
        self.blink_timestamps: Deque[float] = collections.deque()

        # Session observation tracking for rate
        self.session_start_time: Optional[float] = None

    def update(self, current_ear: float, timestamp: float, face_detected: bool) -> BlinkState:
        """
        Updates the blink state machine given the current EAR and timestamp.

        Args:
            current_ear: Current average EAR.
            timestamp: Current timestamp in seconds.
            face_detected: True if face is present and tracking is active.

        Returns:
            BlinkState: Current blink and eye closure metrics.
        """
        if self.session_start_time is None:
            self.session_start_time = timestamp

        blink_event = False

        if not face_detected:
            # If face is lost, reset active closure timer to avoid false prolonged closure
            self.state = EyeState.OPEN
            self.closure_start_time = None
            self.current_closed_duration = 0.0
            blink_rate = self._compute_blink_rate(timestamp)
            return BlinkState(
                eye_status="UNKNOWN",
                blink_event=False,
                blink_count=self.blink_count,
                blink_rate=blink_rate,
                last_blink_duration_ms=self.last_blink_duration_ms,
                eye_closed_duration=0.0,
                max_closed_duration=self.max_closed_duration,
                prolonged_closure=False
            )

        # ---------------------------------------------------------
        # State Machine Transitions
        # ---------------------------------------------------------
        if self.state == EyeState.OPEN:
            if current_ear < self.ear_closed_threshold:
                # Transition: OPEN -> CLOSED
                self.state = EyeState.CLOSED
                self.closure_start_time = timestamp
                self.current_closed_duration = 0.0
            else:
                self.current_closed_duration = 0.0

        elif self.state == EyeState.CLOSED:
            if self.closure_start_time is not None:
                self.current_closed_duration = timestamp - self.closure_start_time
                if self.current_closed_duration > self.max_closed_duration:
                    self.max_closed_duration = self.current_closed_duration

            if current_ear >= self.ear_open_threshold:
                # Transition: CLOSED -> OPEN
                if self.closure_start_time is not None:
                    closure_duration = timestamp - self.closure_start_time
                    # Check if duration matches physiological blink duration
                    if self.min_blink_duration_sec <= closure_duration <= self.max_blink_duration_sec:
                        self.blink_count += 1
                        blink_event = True
                        self.last_blink_duration_ms = closure_duration * 1000.0
                        self.blink_timestamps.append(timestamp)

                self.state = EyeState.OPEN
                self.closure_start_time = None
                self.current_closed_duration = 0.0

        # Check prolonged eye closure flag
        is_closed = (self.state == EyeState.CLOSED)
        prolonged = (is_closed and self.current_closed_duration >= self.prolonged_closure_sec)
        eye_status_str = "CLOSED" if is_closed else "OPEN"

        # Compute blink rate
        blink_rate = self._compute_blink_rate(timestamp)

        return BlinkState(
            eye_status=eye_status_str,
            blink_event=blink_event,
            blink_count=self.blink_count,
            blink_rate=blink_rate,
            last_blink_duration_ms=self.last_blink_duration_ms,
            eye_closed_duration=self.current_closed_duration,
            max_closed_duration=self.max_closed_duration,
            prolonged_closure=prolonged
        )

    def _compute_blink_rate(self, current_timestamp: float) -> float:
        """
        Calculates rolling blink rate (blinks / minute) over the configured window.
        """
        # Evict blinks older than rolling window
        cutoff = current_timestamp - self.rate_window_sec
        while len(self.blink_timestamps) > 0 and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()

        if self.session_start_time is None:
            return 0.0

        # Effective duration of observation in minutes
        observed_sec = min(current_timestamp - self.session_start_time, self.rate_window_sec)
        if observed_sec < 5.0:  # First 5 seconds grace period
            return 0.0

        observed_minutes = observed_sec / 60.0
        return float(len(self.blink_timestamps) / observed_minutes)

    def reset(self) -> None:
        """Resets all counters and state."""
        self.state = EyeState.OPEN
        self.closure_start_time = None
        self.current_closed_duration = 0.0
        self.max_closed_duration = 0.0
        self.blink_count = 0
        self.last_blink_duration_ms = 0.0
        self.blink_timestamps.clear()
        self.session_start_time = None
