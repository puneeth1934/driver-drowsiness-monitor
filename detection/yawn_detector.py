"""
=============================================================================
Yawn Detector Module
=============================================================================
Detects yawning episodes using temporal Mouth Aspect Ratio (MAR) tracking.
Filters out momentary mouth opening (such as talking, singing, or laughing)
by requiring sustained vertical expansion over a configurable duration threshold.
=============================================================================
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import config
from features.ear import calculate_mar


@dataclass
class YawnState:
    """Encapsulates yawning detection metrics and status."""
    mar: float
    mouth_status: str               # "NORMAL" or "OPEN"
    yawn_event: bool                # True for 1 frame when a sustained yawn is confirmed
    yawn_count: int                 # Cumulative count of confirmed yawns
    current_open_duration: float    # Duration mouth has been continuously open (sec)
    is_yawning: bool                # True while a sustained yawn is actively in progress
    mouth_outer_pts: np.ndarray     # Contour points for drawing
    mouth_inner_pts: np.ndarray     # Contour points for drawing


class YawnDetector:
    """
    Temporal detector for yawning events based on sustained Mouth Aspect Ratio.
    """

    def __init__(
        self,
        mar_threshold: float = config.MAR_YAWN_THRESHOLD,
        min_yawn_duration_sec: float = config.MIN_YAWN_DURATION_SEC,
        cooldown_sec: float = config.YAWN_COOLDOWN_SEC,
        mar_indices: List[int] = config.MOUTH_MAR_INDICES,
        outer_indices: List[int] = config.MOUTH_OUTER_INDICES,
        inner_indices: List[int] = config.MOUTH_INNER_INDICES
    ):
        self.mar_threshold = mar_threshold
        self.min_yawn_duration_sec = min_yawn_duration_sec
        self.cooldown_sec = cooldown_sec
        self.mar_indices = mar_indices
        self.outer_indices = outer_indices
        self.inner_indices = inner_indices

        # State tracking
        self.yawn_count: int = 0
        self.open_start_time: Optional[float] = None
        self.current_open_duration: float = 0.0
        self.last_yawn_timestamp: float = -100.0
        self.yawn_already_counted: bool = False
        self.yawn_timestamps = []

    def update(self, landmarks_pixel: np.ndarray, timestamp: float, face_detected: bool) -> YawnState:
        """
        Processes facial landmarks and evaluates mouth opening & yawn duration.

        Args:
            landmarks_pixel: np.ndarray of shape (N, 2) facial landmark points.
            timestamp: Current timestamp in seconds.
            face_detected: True if face is present.

        Returns:
            YawnState with MAR, status, event pulses, and contours.
        """
        if not face_detected or landmarks_pixel is None or len(landmarks_pixel) < 468:
            self.open_start_time = None
            self.current_open_duration = 0.0
            self.yawn_already_counted = False
            return YawnState(
                mar=0.0,
                mouth_status="UNKNOWN",
                yawn_event=False,
                yawn_count=self.yawn_count,
                current_open_duration=0.0,
                is_yawning=False,
                mouth_outer_pts=np.empty((0, 2), dtype=np.int32),
                mouth_inner_pts=np.empty((0, 2), dtype=np.int32)
            )

        # Extract 6-point MAR landmarks
        mouth_pts = landmarks_pixel[self.mar_indices]
        mar = calculate_mar(mouth_pts)
        print(f"MAR VALUE: {mar:.3f}")

        outer_pts = landmarks_pixel[self.outer_indices]
        inner_pts = landmarks_pixel[self.inner_indices]

        mouth_is_open = (mar >= self.mar_threshold)
        mouth_status = "OPEN" if mouth_is_open else "NORMAL"
        yawn_event = False

        if mouth_is_open:
            if self.open_start_time is None:
                self.open_start_time = timestamp
                self.current_open_duration = 0.0
            else:
                self.current_open_duration = timestamp - self.open_start_time

            # Check if duration qualifies as a yawn and cooldown period has passed
            can_trigger = (timestamp - self.last_yawn_timestamp) > self.cooldown_sec
            if (
                self.current_open_duration >= self.min_yawn_duration_sec
                and can_trigger
                and not self.yawn_already_counted
            ):
                self.yawn_count += 1
                yawn_event = True
                self.yawn_already_counted = True
                self.last_yawn_timestamp = timestamp
        else:
            # Mouth closed / normal
            self.open_start_time = None
            self.current_open_duration = 0.0
            self.yawn_already_counted = False

        is_yawning = self.current_open_duration >= self.min_yawn_duration_sec and mouth_is_open

        return YawnState(
            mar=mar,
            mouth_status=mouth_status,
            yawn_event=yawn_event,
            yawn_count=self.yawn_count,
            current_open_duration=self.current_open_duration,
            is_yawning=is_yawning,
            mouth_outer_pts=outer_pts,
            mouth_inner_pts=inner_pts
        )

    def reset(self) -> None:
        """Resets yawn counters and state."""
        self.yawn_count = 0
        self.open_start_time = None
        self.current_open_duration = 0.0
        self.last_yawn_timestamp = -100.0
        self.yawn_already_counted = False
