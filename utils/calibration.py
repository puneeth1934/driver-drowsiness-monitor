"""
=============================================================================
Calibration Module
=============================================================================
Provides an interactive multi-step calibration routine to record the driver's
individual baseline ocular (EAR) and oral (MAR) metrics under current camera
and lighting conditions. Stores personalized reference baselines in JSON format.
=============================================================================
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import numpy as np
import config


@dataclass
class CalibrationProfile:
    timestamp: str
    baseline_open_ear: float
    baseline_closed_ear: float
    suggested_ear_threshold: float
    baseline_normal_mar: float
    suggested_mar_threshold: float
    baseline_pitch: float
    baseline_yaw: float
    baseline_roll: float


class CalibrationManager:
    """
    Guides the driver through a step-by-step calibration procedure to establish
    individualized baseline parameters.
    """

    STEPS = [
        ("Step 1/4: Look straight ahead with eyes normally OPEN (3s)", 3.0, "OPEN_EYES"),
        ("Step 2/4: Keep eyes firmly CLOSED (3s)", 3.0, "CLOSED_EYES"),
        ("Step 3/4: Keep mouth in NORMAL resting position (2s)", 2.0, "NORMAL_MOUTH"),
        ("Step 4/4: Look straight ahead to calibrate HEAD POSE (2s)", 2.0, "HEAD_POSE"),
    ]

    def __init__(self, output_path: Path = config.CALIBRATION_FILE_PATH):
        self.output_path = Path(output_path)
        self.is_active = False
        self.current_step_idx = 0
        self.step_start_time = 0.0

        # Accumulated samples per phase
        self._samples_open_ear: List[float] = []
        self._samples_closed_ear: List[float] = []
        self._samples_normal_mar: List[float] = []
        self._samples_pitch: List[float] = []
        self._samples_yaw: List[float] = []
        self._samples_roll: List[float] = []

        self.last_profile: Optional[CalibrationProfile] = None

    def start(self) -> None:
        """Starts the interactive calibration sequence."""
        self.is_active = True
        self.current_step_idx = 0
        self.step_start_time = time.time()
        self._samples_open_ear.clear()
        self._samples_closed_ear.clear()
        self._samples_normal_mar.clear()
        self._samples_pitch.clear()
        self._samples_yaw.clear()
        self._samples_roll.clear()
        print("\n" + "="*60)
        print("STARTING DRIVER BASELINE CALIBRATION")
        print("="*60)

    def process_frame(
        self,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        roll: float,
        face_detected: bool
    ) -> Optional[str]:
        """
        Feeds live metrics during active calibration and advances steps.

        Returns:
            Optional[str]: User instruction text to render on screen, or None if inactive.
        """
        if not self.is_active:
            return None

        if not face_detected:
            return "Face lost! Please align face with camera..."

        now = time.time()
        step_text, step_duration, step_key = self.STEPS[self.current_step_idx]
        elapsed = now - self.step_start_time
        remaining = max(0.0, step_duration - elapsed)

        # Collect data for current step
        if step_key == "OPEN_EYES":
            if ear > 0.1:
                self._samples_open_ear.append(ear)
        elif step_key == "CLOSED_EYES":
            if ear > 0.0:
                self._samples_closed_ear.append(ear)
        elif step_key == "NORMAL_MOUTH":
            if mar > 0.0:
                self._samples_normal_mar.append(mar)
        elif step_key == "HEAD_POSE":
            self._samples_pitch.append(pitch)
            self._samples_yaw.append(yaw)
            self._samples_roll.append(roll)

        instruction = f"{step_text} [{remaining:.1f}s]"

        # Check step completion
        if elapsed >= step_duration:
            self.current_step_idx += 1
            self.step_start_time = time.time()

            if self.current_step_idx >= len(self.STEPS):
                # Calibration finished
                self._finalize_calibration()
                return "Calibration complete! Resuming monitoring..."

        return instruction

    def _finalize_calibration(self) -> None:
        """Calculates averages, suggests tuned thresholds, and saves to JSON."""
        self.is_active = False

        avg_open_ear = float(np.median(self._samples_open_ear)) if self._samples_open_ear else 0.30
        avg_closed_ear = float(np.median(self._samples_closed_ear)) if self._samples_closed_ear else 0.15
        avg_normal_mar = float(np.median(self._samples_normal_mar)) if self._samples_normal_mar else 0.25

        avg_pitch = float(np.mean(self._samples_pitch)) if self._samples_pitch else 0.0
        avg_yaw = float(np.mean(self._samples_yaw)) if self._samples_yaw else 0.0
        avg_roll = float(np.mean(self._samples_roll)) if self._samples_roll else 0.0

        # Suggested EAR threshold: mid-point weighted towards closed
        suggested_ear = float(avg_closed_ear + 0.40 * (avg_open_ear - avg_closed_ear))
        # Suggested MAR threshold: ~2.2x normal resting MAR
        suggested_mar = float(max(0.55, avg_normal_mar * 2.2))

        profile = CalibrationProfile(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            baseline_open_ear=round(avg_open_ear, 4),
            baseline_closed_ear=round(avg_closed_ear, 4),
            suggested_ear_threshold=round(suggested_ear, 4),
            baseline_normal_mar=round(avg_normal_mar, 4),
            suggested_mar_threshold=round(suggested_mar, 4),
            baseline_pitch=round(avg_pitch, 2),
            baseline_yaw=round(avg_yaw, 2),
            baseline_roll=round(avg_roll, 2)
        )

        self.last_profile = profile

        # Save profile to JSON
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(asdict(profile), f, indent=4)
            print(f"[Calibration] Profile saved successfully to {self.output_path}")
        except Exception as e:
            print(f"[Calibration] Failed to write calibration JSON: {e}")

        print("\n" + "="*60)
        print("CALIBRATION RESULTS SUMMARY")
        print("="*60)
        print(f" Baseline Open EAR:       {profile.baseline_open_ear:.4f}")
        print(f" Baseline Closed EAR:     {profile.baseline_closed_ear:.4f}")
        print(f" Suggested EAR Threshold: {profile.suggested_ear_threshold:.4f}")
        print(f" Baseline Normal MAR:     {profile.baseline_normal_mar:.4f}")
        print(f" Suggested MAR Threshold: {profile.suggested_mar_threshold:.4f}")
        print(f" Baseline Head Pose (P/Y/R): {profile.baseline_pitch:+.1f} / {profile.baseline_yaw:+.1f} / {profile.baseline_roll:+.1f} deg")
        print("="*60 + "\n")
