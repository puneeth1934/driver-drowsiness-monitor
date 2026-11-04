"""
=============================================================================
Driver Drowsiness Detection System — Phase 1: Webcam Eye Tracking
=============================================================================
Main executable entry point for real-time driver monitoring.
Integrates face detection, eye tracking, EAR, PERCLOS, blink detection,
yawn detection, head pose estimation, OpenCV HUD dashboard, and CSV logging.
=============================================================================
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Add project root directory to path for clean modular imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from camera import CameraManager
from detection import (
    FaceDetector,
    EyeTracker,
    BlinkDetector,
    YawnDetector,
    HeadPoseEstimator
)
from features import PERCLOSTracker, FeatureLogger
from utils import HUDVisualizer, CalibrationManager


def parse_arguments() -> argparse.Namespace:
    """Parses optional command-line flags."""
    parser = argparse.ArgumentParser(description="AI-Based Multimodal Driver Drowsiness Detection System — Phase 1")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help=f"Camera index (default: {config.CAMERA_INDEX})")
    parser.add_argument("--calibrate", action="store_true", help="Start immediately in calibration mode")
    parser.add_argument("--no-log", action="store_true", help="Disable CSV feature logging")
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH, help="Camera width")
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT, help="Camera height")
    return parser.parse_args()


def print_startup_banner() -> None:
    """Prints the required system startup confirmation banner."""
    print("=" * 55)
    print("Driver Drowsiness Detection System")
    print("Phase 1: Eye Tracking")
    print("Camera: Connected")
    print("Face Detection: Ready")
    print("Eye Tracking: Ready")
    print("PERCLOS: Ready")
    print("Blink Detection: Ready")
    print("Yawn Detection: Ready")
    print("Data Logging: Ready")
    print("=" * 55)
    print("Controls:")
    print("  [Q] or [ESC] - Quit Application")
    print("  [C]         - Trigger User Baseline Calibration")
    print("  [R]         - Reset Statistics / Counters")
    print("=" * 55 + "\n")


def main() -> None:
    args = parse_arguments()

    # 1. Initialize Camera
    camera = CameraManager(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        target_fps=config.TARGET_FPS
    )
    if not camera.start():
        print(f"[Error] Failed to initialize camera index {args.camera}. Please check USB connection.")
        sys.exit(1)

    # 2. Initialize Detection & Feature Modules
    face_detector = FaceDetector()
    eye_tracker = EyeTracker()
    blink_detector = BlinkDetector()
    yawn_detector = YawnDetector()
    head_pose_estimator = HeadPoseEstimator()
    perclos_tracker = PERCLOSTracker(window_seconds=config.PERCLOS_WINDOW_SECONDS)
    feature_logger = FeatureLogger(enabled=not args.no_log)
    hud_visualizer = HUDVisualizer()
    calibration_mgr = CalibrationManager()

    # Print requested status banner
    print_startup_banner()

    if args.calibrate:
        calibration_mgr.start()

    window_name = "Driver Drowsiness Detection System - Phase 1: Eye & Pose Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    try:
        frame_count = 0
        # Distraction tracking
        distraction_start_time = None
        distraction_direction = None

        # Yawn tracking - 30 second window
        yawn_window_start = time.time()
        yawn_events_30s = 0

        # Eye closure tracking
        eye_closure_start_time = None
        while True:
            timestamp_now = time.time()

            # Acquire Frame
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                # Retry reading frame
                time.sleep(0.01)
                continue
            frame_count +=1

            h, w, _ = frame.shape
            fps = camera.get_fps()

            # Process Face Detection
            face_res = face_detector.process_frame(frame)

            # Feature placeholders
            left_ear = 0.0
            right_ear = 0.0
            avg_ear = 0.0
            mar_val = 0.0
            pitch, yaw, roll = 0.0, 0.0, 0.0
            nose_landmark = None
            nose_proj = None
            left_contour = None
            right_contour = None
            mouth_outer = None
            mouth_inner = None

            if face_res.detected and face_res.landmarks_pixel is not None:
                # 1. Eye Tracking & EAR
                eye_res = eye_tracker.track(face_res.landmarks_pixel)
                if eye_res is not None:
                    left_ear = eye_res.left_ear
                    right_ear = eye_res.right_ear
                    avg_ear = eye_res.average_ear
                    left_contour = eye_res.left_contour
                    right_contour = eye_res.right_contour

                # 2. Blink Detection & Prolonged Closure
                blink_res = blink_detector.update(
                    current_ear=avg_ear,
                    timestamp=timestamp_now,
                    face_detected=True
                )

                # 3. PERCLOS
                is_closed_now = (blink_res.eye_status == "CLOSED")
                current_perclos = perclos_tracker.update(
                    timestamp=timestamp_now,
                    is_closed=is_closed_now,
                    face_detected=True
                )

                # 4. Yawning Detection
                yawn_res = yawn_detector.update(
                    landmarks_pixel=face_res.landmarks_pixel,
                    timestamp=timestamp_now,
                    face_detected=True
                )
                mar_val = yawn_res.mar
                mouth_outer = yawn_res.mouth_outer_pts
                mouth_inner = yawn_res.mouth_inner_pts

                # 5. Head Pose Estimation
                pose_res = head_pose_estimator.estimate(
                    landmarks_pixel=face_res.landmarks_pixel,
                    frame_width=w,
                    frame_height=h
                )
                pitch, yaw, roll = pose_res.pitch, pose_res.yaw, pose_res.roll
                nose_proj = pose_res.nose_end_point_2d
                # Landmark index 1 is nose tip
                nose_landmark = face_res.landmarks_pixel[1]

            else:
                # No face detected in this frame
                blink_res = blink_detector.update(
                    current_ear=0.0,
                    timestamp=timestamp_now,
                    face_detected=False
                )
                current_perclos = perclos_tracker.update(
                    timestamp=timestamp_now,
                    is_closed=False,
                    face_detected=False
                )
                yawn_res = yawn_detector.update(
                    landmarks_pixel=None,
                    timestamp=timestamp_now,
                    face_detected=False
                )
                pose_res = head_pose_estimator.estimate(
                    landmarks_pixel=None,
                    frame_width=w,
                    frame_height=h
                )

            # Handle Active Calibration Mode
            calib_text = None
            if calibration_mgr.is_active:
                calib_text = calibration_mgr.process_frame(
                    ear=avg_ear,
                    mar=mar_val,
                    pitch=pitch,
                    yaw=yaw,
                    roll=roll,
                    face_detected=face_res.detected
                )

            # Compile Full Telemetry Dictionary
            metrics = {
                "timestamp": timestamp_now,
                "face_detected": face_res.detected,
                "multiple_faces": face_res.multiple_faces,
                "num_faces": face_res.num_faces,
                "left_ear": left_ear,
                "right_ear": right_ear,
                "average_ear": avg_ear,
                "eye_status": blink_res.eye_status,
                "eye_closed_duration": blink_res.eye_closed_duration,
                "prolonged_closure": blink_res.prolonged_closure,
                "blink_event": blink_res.blink_event,
                "blink_count": blink_res.blink_count,
                "blink_rate": blink_res.blink_rate,
                "perclos": current_perclos,
                "mouth_ratio": mar_val,
                "mouth_status": yawn_res.mouth_status,
                "yawn_event": yawn_res.yawn_event,
                "yawn_count": yawn_res.yawn_count,
                "is_yawning": yawn_res.is_yawning,
                "head_pitch": pitch,
                "head_yaw": yaw,
                "head_roll": roll,
                "head_orientation": pose_res.orientation_label,
                "fps": fps,
                # Drawing coordinates
                "left_contour_pts": left_contour,
                "right_contour_pts": right_contour,
                "mouth_outer_pts": mouth_outer,
                "mouth_inner_pts": mouth_inner,
                "nose_landmark_pt": nose_landmark,
                "nose_projected_pt": nose_proj
            }

            # CSV Logging (rate-limited, with discrete events prioritized)
            feature_logger.log(metrics)

            # Determine System State Label
            if calibration_mgr.is_active:
                sys_status = "CALIBRATION"
            elif blink_res.prolonged_closure:
                sys_status = "WARNING: CLOSURE"
            elif yawn_res.is_yawning:
                sys_status = "WARNING: YAWNING"
            elif not face_res.detected:
                sys_status = "SEARCHING FACE"
            else:
                sys_status = "MONITORING"

            # Render HUD Display
            annotated_frame = hud_visualizer.draw_hud(
                frame=frame,
                metrics=metrics,
                system_status=sys_status,
                calibration_text=calib_text
            )

            cv2.imshow(window_name, annotated_frame)

            # Keyboard Handling
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:  # 'Q' or ESC
                print("[System] User requested exit. Shutting down...")
                break
            elif key in [ord('c'), ord('C')]:
                if not calibration_mgr.is_active:
                    calibration_mgr.start()
            elif key in [ord('r'), ord('R')]:
                print("[System] Resetting all statistics and counters...")
                blink_detector.reset()
                yawn_detector.reset()
                perclos_tracker.reset()

    except KeyboardInterrupt:
        print("[System] KeyboardInterrupt received. Exiting cleanly...")
    finally:
        # Clean Resource Teardown
        print("[System] Cleaning up resources...")
        camera.release()
        face_detector.close()
        feature_logger.close()
        cv2.destroyAllWindows()
        print("[System] Shutdown complete.")


if __name__ == "__main__":
    main()
