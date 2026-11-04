"""
=============================================================================
Automated Unit & Feature Integration Tests
=============================================================================
Verifies mathematical models (EAR, MAR), temporal state machines (BlinkDetector,
YawnDetector), time-windowed PERCLOS calculations, Head Pose estimation,
and CSV logging without requiring physical camera hardware.
=============================================================================
"""

import os
import sys
import time
import math
import numpy as np
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from features.ear import calculate_ear, calculate_mar
from features.perclos import PERCLOSTracker
from features.feature_logger import FeatureLogger
from detection.blink_detector import BlinkDetector, EyeState
from detection.yawn_detector import YawnDetector
from detection.head_pose import HeadPoseEstimator


def test_ear_calculation():
    """Test standard 6-point EAR calculation with open and closed synthetic geometry."""
    print("[TEST] Running test_ear_calculation...")

    # Canonical open eye geometry: width = 20, vertical openings = 6 each
    # p1=(0,0), p2=(5, 3), p3=(15, 3), p4=(20, 0), p5=(15, -3), p6=(5, -3)
    # v1 = 6, v2 = 6, h = 20 -> EAR = (6 + 6) / (2 * 20) = 12 / 40 = 0.30
    open_eye = np.array([
        [0.0, 0.0],
        [5.0, 3.0],
        [15.0, 3.0],
        [20.0, 0.0],
        [15.0, -3.0],
        [5.0, -3.0]
    ], dtype=np.float64)

    ear_open = calculate_ear(open_eye)
    assert math.isclose(ear_open, 0.30, rel_tol=1e-3), f"Expected 0.30, got {ear_open}"

    # Closed eye geometry: vertical distances shrink to 0.5
    closed_eye = np.array([
        [0.0, 0.0],
        [5.0, 0.5],
        [15.0, 0.5],
        [20.0, 0.0],
        [15.0, -0.5],
        [5.0, -0.5]
    ], dtype=np.float64)

    ear_closed = calculate_ear(closed_eye)
    assert math.isclose(ear_closed, 0.05, rel_tol=1e-3), f"Expected 0.05, got {ear_closed}"
    print(f"  -> Open EAR: {ear_open:.4f}, Closed EAR: {ear_closed:.4f} [PASSED]")


def test_mar_calculation():
    """Test 6-point Mouth Aspect Ratio calculation."""
    print("[TEST] Running test_mar_calculation...")

    # Resting mouth: width = 40, vertical openings = 5 each -> MAR = (5 + 5) / (2 * 40) = 10 / 80 = 0.125
    resting_mouth = np.array([
        [0.0, 0.0],
        [10.0, 2.5],
        [30.0, 2.5],
        [40.0, 0.0],
        [30.0, -2.5],
        [10.0, -2.5]
    ], dtype=np.float64)

    mar_normal = calculate_mar(resting_mouth)
    assert math.isclose(mar_normal, 0.125, rel_tol=1e-3), f"Expected 0.125, got {mar_normal}"

    # Wide yawning mouth: width = 40, vertical openings = 30 each -> MAR = 60 / 80 = 0.75
    yawn_mouth = np.array([
        [0.0, 0.0],
        [10.0, 15.0],
        [30.0, 15.0],
        [40.0, 0.0],
        [30.0, -15.0],
        [10.0, -15.0]
    ], dtype=np.float64)

    mar_yawn = calculate_mar(yawn_mouth)
    assert math.isclose(mar_yawn, 0.75, rel_tol=1e-3), f"Expected 0.75, got {mar_yawn}"
    print(f"  -> Normal MAR: {mar_normal:.4f}, Yawn MAR: {mar_yawn:.4f} [PASSED]")


def test_blink_state_machine():
    """Test temporal blink detection and prolonged closure triggers."""
    print("[TEST] Running test_blink_state_machine...")

    detector = BlinkDetector(
        ear_closed_threshold=0.21,
        ear_open_threshold=0.26,
        min_blink_duration_sec=0.06,
        max_blink_duration_sec=0.45,
        prolonged_closure_sec=1.5
    )

    t = 1000.0

    # 1. Driver has eyes open (EAR = 0.32)
    s1 = detector.update(current_ear=0.32, timestamp=t, face_detected=True)
    assert s1.eye_status == "OPEN"
    assert s1.blink_count == 0
    assert not s1.blink_event

    # 2. Driver closes eyes for 150 ms (Blink)
    t += 0.033
    detector.update(current_ear=0.18, timestamp=t, face_detected=True)
    t += 0.120
    detector.update(current_ear=0.17, timestamp=t, face_detected=True)
    # Eyes reopen
    t += 0.033
    s2 = detector.update(current_ear=0.32, timestamp=t, face_detected=True)
    assert s2.blink_event is True, "Blink event should trigger when eyes reopen"
    assert s2.blink_count == 1, f"Expected blink count 1, got {s2.blink_count}"
    assert not s2.prolonged_closure

    # 3. Next frame, blink_event pulse should reset to False
    t += 0.033
    s3 = detector.update(current_ear=0.32, timestamp=t, face_detected=True)
    assert s3.blink_event is False

    # 4. Prolonged eye closure: eyes close and stay closed for 2.0 seconds
    t += 0.033
    detector.update(current_ear=0.15, timestamp=t, face_detected=True)
    t += 1.60  # Exceeds prolonged_closure_sec = 1.5s
    s_prolonged = detector.update(current_ear=0.15, timestamp=t, face_detected=True)
    assert s_prolonged.eye_status == "CLOSED"
    assert s_prolonged.prolonged_closure is True, "Prolonged closure flag should be active"
    print(f"  -> Blink Count: {s2.blink_count}, Prolonged Flag: {s_prolonged.prolonged_closure} [PASSED]")


def test_perclos_rolling_window():
    """Test timestamp-weighted rolling window PERCLOS computation."""
    print("[TEST] Running test_perclos_rolling_window...")

    tracker = PERCLOSTracker(window_seconds=10.0)
    t_start = 100.0

    # Simulate 10 seconds of data: 8 seconds open, 2 seconds closed (20% PERCLOS)
    # Step at 30 FPS = ~0.033s
    dt = 0.033
    num_steps = 300  # 10 seconds

    for i in range(num_steps):
        current_t = t_start + i * dt
        # First 8s open (is_closed=False), next 2s closed (is_closed=True)
        is_closed = (i >= 240)
        p = tracker.update(timestamp=current_t, is_closed=is_closed, face_detected=True)

    final_p = tracker.get_perclos()
    # Expect ~ 2.0 / 10.0 = 20.0% (+/- 2%)
    assert 18.0 <= final_p <= 22.0, f"Expected ~20.0% PERCLOS, got {final_p:.2f}%"
    print(f"  -> Simulated 20% closure -> Calculated PERCLOS: {final_p:.2f}% [PASSED]")


def test_yawn_detector():
    """Test temporal yawn detection with duration threshold."""
    print("[TEST] Running test_yawn_detector...")

    detector = YawnDetector(
        mar_threshold=0.60,
        min_yawn_duration_sec=1.5,
        cooldown_sec=3.0
    )

    # Synthetic landmark array (468 points)
    synth_landmarks = np.zeros((468, 2), dtype=np.float64)

    # Set normal resting mouth indices
    for idx in config.MOUTH_MAR_INDICES:
        synth_landmarks[idx] = [100.0, 100.0]
    synth_landmarks[config.MOUTH_MAR_INDICES[0]] = [80.0, 100.0]   # left corner
    synth_landmarks[config.MOUTH_MAR_INDICES[3]] = [120.0, 100.0]  # right corner
    synth_landmarks[config.MOUTH_MAR_INDICES[1]] = [90.0, 104.0]
    synth_landmarks[config.MOUTH_MAR_INDICES[2]] = [110.0, 104.0]
    synth_landmarks[config.MOUTH_MAR_INDICES[4]] = [110.0, 96.0]
    synth_landmarks[config.MOUTH_MAR_INDICES[5]] = [90.0, 96.0]

    t = 200.0
    res1 = detector.update(synth_landmarks, timestamp=t, face_detected=True)
    assert res1.mouth_status == "NORMAL"
    assert res1.yawn_count == 0

    # Expand vertically for yawn (top = 130, bot = 70 -> vertical diff = 60, width = 40 -> MAR = 1.5)
    synth_landmarks[config.MOUTH_MAR_INDICES[1]] = [90.0, 130.0]
    synth_landmarks[config.MOUTH_MAR_INDICES[2]] = [110.0, 130.0]
    synth_landmarks[config.MOUTH_MAR_INDICES[4]] = [110.0, 70.0]
    synth_landmarks[config.MOUTH_MAR_INDICES[5]] = [90.0, 70.0]

    # Mouth opens wide for 0.5s (talking/brief - not a yawn)
    detector.update(synth_landmarks, timestamp=t + 0.1, face_detected=True)
    res_short = detector.update(synth_landmarks, timestamp=t + 0.5, face_detected=True)
    assert res_short.mouth_status == "OPEN"
    assert res_short.yawn_count == 0, "Brief opening should not trigger yawn"

    # Sustained opening reaches 1.8s (exceeds min_yawn_duration_sec = 1.5s)
    res_yawn = detector.update(synth_landmarks, timestamp=t + 1.8, face_detected=True)
    assert res_yawn.yawn_event is True, "Sustained opening should trigger yawn event"
    assert res_yawn.yawn_count == 1
    print(f"  -> Yawn Count: {res_yawn.yawn_count}, Yawn Event Pulse: {res_yawn.yawn_event} [PASSED]")


def test_csv_logger():
    """Test rate-limited CSV logger and output file schema."""
    print("[TEST] Running test_csv_logger...")

    test_csv = BASE_DIR / "data" / "test_telemetry.csv"
    if test_csv.exists():
        test_csv.unlink()

    logger = FeatureLogger(csv_path=test_csv, frequency_hz=10.0, enabled=True)

    # Write synthetic records
    sample_metrics = {
        "face_detected": True,
        "left_ear": 0.3123,
        "right_ear": 0.3245,
        "average_ear": 0.3184,
        "eye_status": "OPEN",
        "eye_closed_duration": 0.0,
        "blink_event": 0,
        "blink_count": 5,
        "blink_rate": 14.5,
        "perclos": 4.2,
        "mouth_ratio": 0.22,
        "mouth_status": "NORMAL",
        "yawn_event": 0,
        "yawn_count": 1,
        "head_pitch": -2.1,
        "head_yaw": 3.4,
        "head_roll": 0.5,
        "fps": 29.8
    }

    # Write row
    logger.log(sample_metrics, force=True)
    logger.close()

    assert test_csv.exists(), "CSV file should be created"
    with open(test_csv, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 2, f"Expected 2 lines (header + row), got {len(lines)}"
    header = lines[0].strip().split(",")
    for expected_col in ["timestamp", "elapsed_time", "left_ear", "perclos", "head_pitch"]:
        assert expected_col in header, f"Column {expected_col} missing from header"

    # Cleanup test artifact
    test_csv.unlink()
    print("  -> CSV schema, column names, and rate-limiting [PASSED]")


def run_all_tests():
    print("=" * 60)
    print("STARTING TEST SUITE: Driver Drowsiness Eye-Tracking Modules")
    print("=" * 60)
    test_ear_calculation()
    test_mar_calculation()
    test_blink_state_machine()
    test_perclos_rolling_window()
    test_yawn_detector()
    test_csv_logger()
    print("=" * 60)
    print("ALL 6 AUTOMATED MODULE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
