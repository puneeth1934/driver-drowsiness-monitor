"""
=============================================================================
Drawing & HUD Visualization Module
=============================================================================
Renders a professional, non-intrusive automotive HUD overlay on OpenCV video
frames. Draws eye and mouth contours, 3D head pose projection axes, telemetry
cards, and prioritized alert banners without obscuring the driver's eyes.
=============================================================================
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import config


class HUDVisualizer:
    """
    Renders telemetry data, landmark contours, and status banners onto frames.
    """

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX
        self.card_width = 270

    def draw_hud(
        self,
        frame: np.ndarray,
        metrics: Dict[str, Any],
        system_status: str = "MONITORING",
        calibration_text: Optional[str] = None
    ) -> np.ndarray:
        """
        Draws the complete HUD dashboard on the given image frame.

        Args:
            frame: Input BGR image.
            metrics: Dictionary of current frame features and states.
            system_status: Current system mode ("MONITORING", "CALIBRATING", "PAUSED").
            calibration_text: Optional calibration instruction overlay.

        Returns:
            np.ndarray: Frame with HUD overlay.
        """
        output = frame.copy()
        h, w, _ = output.shape

        face_detected = bool(metrics.get("face_detected", False))
        multiple_faces = bool(metrics.get("multiple_faces", False))
        prolonged_closure = bool(metrics.get("prolonged_closure", False))
        is_yawning = bool(metrics.get("is_yawning", False))

        # 1. Draw facial feature contours if face is detected
        if face_detected:
            self._draw_facial_contours(output, metrics)
            self._draw_head_pose_vector(output, metrics)

        # 2. Draw side telemetry dashboard panel
        #self._draw_telemetry_panel(output, metrics, system_status)

        # 3. Draw alert/warning banners
        if not face_detected:
            self._draw_banner(output, "FACE NOT DETECTED", bg_color=(20, 20, 200), text_color=(255, 255, 255))
        elif multiple_faces:
            self._draw_banner(output, "MULTIPLE FACES DETECTED - TRACKING PRIMARY", bg_color=(20, 140, 240), text_color=(0, 0, 0))

        if prolonged_closure:
            self._draw_alert_bar(output, "PROLONGED EYE CLOSURE", bg_color=(0, 0, 220))
        elif is_yawning:
            self._draw_alert_bar(output, "YAWNING DETECTED", bg_color=(0, 120, 240))

        # 4. Draw calibration overlay if active
        if calibration_text:
            self._draw_calibration_modal(output, calibration_text)

        return output

    def _draw_telemetry_panel(self, frame: np.ndarray, metrics: Dict[str, Any], system_status: str) -> None:
        """Renders semi-transparent side card with numerical metrics."""
        h, w, _ = frame.shape
        panel_w = self.card_width
        panel_h = h - 20

        # Semi-transparent dark background card on left side
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), config.COLOR_BG_CARD, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Subtle card border
        cv2.rectangle(frame, (10, 10), (panel_w, panel_h), (60, 60, 70), 1)

        # Header Title
        x = 22
        y = 35
        cv2.putText(frame, "DRIVER MONITORING", (x, y), self.font, 0.55, config.COLOR_TEXT_WHITE, 2, cv2.LINE_AA)
        y += 18
        cv2.putText(frame, "Phase 1: Eye & Pose Tracking", (x, y), self.font_small, 0.35, config.COLOR_CYAN, 1, cv2.LINE_AA)

        y += 12
        cv2.line(frame, (18, y), (panel_w - 10, y), (80, 80, 90), 1)
        y += 20

        # Face Status
        face_detected = bool(metrics.get("face_detected", False))
        face_str = "DETECTED" if face_detected else "NO FACE"
        face_col = config.COLOR_NOMINAL if face_detected else config.COLOR_ALERT
        self._draw_row(frame, x, y, "Face:", face_str, val_color=face_col)
        y += 22

        # Eye Aspect Ratios
        left_ear = float(metrics.get("left_ear", 0.0))
        right_ear = float(metrics.get("right_ear", 0.0))
        avg_ear = float(metrics.get("average_ear", 0.0))

        self._draw_row(frame, x, y, "Left EAR:", f"{left_ear:.2f}")
        y += 18
        self._draw_row(frame, x, y, "Right EAR:", f"{right_ear:.2f}")
        y += 18
        self._draw_row(frame, x, y, "Average EAR:", f"{avg_ear:.3f}", val_color=config.COLOR_CYAN)
        y += 22

        # Eye Status
        eye_status = str(metrics.get("eye_status", "UNKNOWN"))
        eye_col = config.COLOR_NOMINAL if eye_status == "OPEN" else config.COLOR_ALERT
        self._draw_row(frame, x, y, "Eye Status:", eye_status, val_color=eye_col)
        y += 22

        # Blink Metrics
        blink_count = int(metrics.get("blink_count", 0))
        blink_rate = float(metrics.get("blink_rate", 0.0))
        self._draw_row(frame, x, y, "Blink Count:", f"{blink_count}")
        y += 18
        self._draw_row(frame, x, y, "Blink Rate:", f"{blink_rate:.1f} /min")
        y += 22

        # PERCLOS
        perclos = float(metrics.get("perclos", 0.0))
        perclos_col = config.COLOR_ALERT if perclos >= 15.0 else (config.COLOR_WARNING if perclos >= 8.0 else config.COLOR_NOMINAL)
        self._draw_row(frame, x, y, "PERCLOS (60s):", f"{perclos:.1f} %", val_color=perclos_col)
        y += 22

        # Mouth & Yawn
        mouth_status = str(metrics.get("mouth_status", "NORMAL"))
        yawn_count = int(metrics.get("yawn_count", 0))
        mar_val = float(metrics.get("mouth_ratio", 0.0))
        mouth_col = config.COLOR_ALERT if mouth_status == "OPEN" else config.COLOR_NOMINAL
        self._draw_row(frame, x, y, "Mouth Status:", mouth_status, val_color=mouth_col)
        y += 18
        self._draw_row(frame, x, y, "Mouth MAR:", f"{mar_val:.2f}")
        y += 18
        self._draw_row(frame, x, y, "Yawn Count:", f"{yawn_count}")
        y += 22

        # Head Pose
        pitch = float(metrics.get("head_pitch", 0.0))
        yaw = float(metrics.get("head_yaw", 0.0))
        roll = float(metrics.get("head_roll", 0.0))
        self._draw_row(frame, x, y, "Head Pitch:", f"{pitch:+.1f} deg")
        y += 18
        self._draw_row(frame, x, y, "Head Yaw:", f"{yaw:+.1f} deg")
        y += 18
        self._draw_row(frame, x, y, "Head Roll:", f"{roll:+.1f} deg")
        y += 22

        # FPS & System Status Footer
        cv2.line(frame, (18, y), (panel_w - 10, y), (80, 80, 90), 1)
        y += 18
        fps = float(metrics.get("fps", 0.0))
        self._draw_row(frame, x, y, "FPS:", f"{fps:.1f}", val_color=config.COLOR_NOMINAL)
        y += 18
        self._draw_row(frame, x, y, "System Status:", system_status, val_color=config.COLOR_CYAN)

    def _draw_row(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        label: str,
        value: str,
        val_color: Tuple[int, int, int] = config.COLOR_TEXT_WHITE
    ) -> None:
        """Helper to render a key-value label row with aligned columns."""
        cv2.putText(frame, label, (x, y), self.font_small, 0.40, config.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x + 130, y), self.font_small, 0.42, val_color, 1, cv2.LINE_AA)

    def _draw_facial_contours(self, frame: np.ndarray, metrics: Dict[str, Any]) -> None:
        """Draws fine eye contours and mouth contours on the detected face."""
        left_eye_pts = metrics.get("left_contour_pts")
        right_eye_pts = metrics.get("right_contour_pts")
        mouth_outer_pts = metrics.get("mouth_outer_pts")
        mouth_inner_pts = metrics.get("mouth_inner_pts")
        eye_status = metrics.get("eye_status", "OPEN")
        mouth_status = metrics.get("mouth_status", "NORMAL")

        eye_color = config.COLOR_NOMINAL if eye_status == "OPEN" else config.COLOR_ALERT
        mouth_color = config.COLOR_ALERT if mouth_status == "OPEN" else (180, 180, 180)

        # Draw left & right eye contours
        if left_eye_pts is not None and len(left_eye_pts) > 0:
            cv2.polylines(frame, [np.array(left_eye_pts, dtype=np.int32)], isClosed=True, color=eye_color, thickness=1, lineType=cv2.LINE_AA)

        if right_eye_pts is not None and len(right_eye_pts) > 0:
            cv2.polylines(frame, [np.array(right_eye_pts, dtype=np.int32)], isClosed=True, color=eye_color, thickness=1, lineType=cv2.LINE_AA)

        # Draw mouth contours
        if mouth_outer_pts is not None and len(mouth_outer_pts) > 0:
            cv2.polylines(frame, [np.array(mouth_outer_pts, dtype=np.int32)], isClosed=True, color=mouth_color, thickness=1, lineType=cv2.LINE_AA)
        if mouth_inner_pts is not None and len(mouth_inner_pts) > 0:
            cv2.polylines(frame, [np.array(mouth_inner_pts, dtype=np.int32)], isClosed=True, color=mouth_color, thickness=1, lineType=cv2.LINE_AA)

    def _draw_head_pose_vector(self, frame: np.ndarray, metrics: Dict[str, Any]) -> None:
        """Draws the 3D projected nose gaze axis on the face."""
        nose_pt = metrics.get("nose_landmark_pt")
        nose_proj = metrics.get("nose_projected_pt")

        if nose_pt is not None and nose_proj is not None:
            p1 = (int(nose_pt[0]), int(nose_pt[1]))
            p2 = (int(nose_proj[0]), int(nose_proj[1]))
            # Draw line representing head direction
            cv2.arrowedLine(frame, p1, p2, (0, 220, 255), 2, tipLength=0.25)
            cv2.circle(frame, p1, 3, (0, 255, 0), -1)

    def _draw_banner(
        self,
        frame: np.ndarray,
        text: str,
        bg_color: Tuple[int, int, int] = (20, 20, 200),
        text_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> None:
        """Draws a centered top notification banner."""
        h, w, _ = frame.shape
        banner_h = 36
        banner_y = 12

        overlay = frame.copy()
        cv2.rectangle(overlay, (self.card_width + 20, banner_y), (w - 20, banner_y + banner_h), bg_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Center text
        (tw, th), _ = cv2.getTextSize(text, self.font, 0.52, 2)
        tx = int((self.card_width + 20 + w - 20 - tw) / 2)
        ty = int(banner_y + (banner_h + th) / 2)
        cv2.putText(frame, text, (tx, ty), self.font, 0.52, text_color, 2, cv2.LINE_AA)

    def _draw_alert_bar(
        self,
        frame: np.ndarray,
        text: str,
        bg_color: Tuple[int, int, int] = (0, 0, 220)
    ) -> None:
        """Draws a bottom prominent flashing alert bar."""
        h, w, _ = frame.shape
        bar_h = 42
        bar_y = h - bar_h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (self.card_width + 20, bar_y), (w - 20, bar_y + bar_h), bg_color, -1)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

        (tw, th), _ = cv2.getTextSize(text, self.font, 0.65, 2)
        tx = int((self.card_width + 20 + w - 20 - tw) / 2)
        ty = int(bar_y + (bar_h + th) / 2)
        cv2.putText(frame, text, (tx, ty), self.font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_calibration_modal(self, frame: np.ndarray, text: str) -> None:
        """Renders calibration instructions in a floating modal box."""
        h, w, _ = frame.shape
        mw = 420
        mh = 90
        mx = int((w - mw) / 2)
        my = int(h / 2 - mh / 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), (35, 30, 20), -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 215, 255), 2)

        cv2.putText(frame, "CALIBRATION IN PROGRESS", (mx + 20, my + 28), self.font, 0.55, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (mx + 20, my + 60), self.font_small, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
