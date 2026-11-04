"""
=============================================================================
Face Detector Module (MediaPipe Face Mesh)
=============================================================================
Detects human faces and extracts 468/478 precise 3D facial landmarks using
MediaPipe FaceMesh. Supports multi-face detection handling, primary face selection,
and robust coordinate transformation from normalized to pixel space.
=============================================================================
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import cv2
import numpy as np
import mediapipe as mp
import config


@dataclass
class FaceDetectionResult:
    """Encapsulates the output of the face landmark detector for a single frame."""
    detected: bool
    num_faces: int
    multiple_faces: bool
    landmarks_pixel: Optional[np.ndarray] = None       # Shape: (N, 2) [x, y] in pixel coordinates
    landmarks_normalized: Optional[np.ndarray] = None  # Shape: (N, 3) [x, y, z] normalized [0, 1]
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x_min, y_min, x_max, y_max)


class FaceDetector:
    """
    MediaPipe FaceMesh wrapper designed for high-throughput, low-latency driver monitoring.
    """

    def __init__(
        self,
        max_num_faces: int = config.MAX_NUM_FACES,
        refine_landmarks: bool = config.REFINE_LANDMARKS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE
    ):
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame_bgr: np.ndarray) -> FaceDetectionResult:
        """
        Processes a BGR image frame and extracts primary face landmarks.

        Args:
            frame_bgr: Input OpenCV BGR image frame.

        Returns:
            FaceDetectionResult with primary driver landmark points and status flags.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return FaceDetectionResult(detected=False, num_faces=0, multiple_faces=False)

        h, w, _ = frame_bgr.shape

        # MediaPipe requires RGB image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Mark as not writeable to improve performance
        frame_rgb.flags.writeable = False
        results = self._face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return FaceDetectionResult(detected=False, num_faces=0, multiple_faces=False)

        num_faces = len(results.multi_face_landmarks)
        multiple_faces = num_faces > 1

        # If multiple faces detected, select the primary driver face based on largest bounding box
        selected_face_landmarks = None
        best_area = -1.0
        best_bbox = (0, 0, 0, 0)
        best_pixel_coords = None
        best_norm_coords = None

        for face_landmarks in results.multi_face_landmarks:
            # Extract normalized coordinates
            norm_pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark], dtype=np.float32)

            # Convert to pixel coordinates
            pixel_pts = np.zeros((norm_pts.shape[0], 2), dtype=np.int32)
            pixel_pts[:, 0] = np.clip(norm_pts[:, 0] * w, 0, w - 1).astype(np.int32)
            pixel_pts[:, 1] = np.clip(norm_pts[:, 1] * h, 0, h - 1).astype(np.int32)

            x_min = int(np.min(pixel_pts[:, 0]))
            y_min = int(np.min(pixel_pts[:, 1]))
            x_max = int(np.max(pixel_pts[:, 0]))
            y_max = int(np.max(pixel_pts[:, 1]))
            bbox_area = float((x_max - x_min) * (y_max - y_min))

            if bbox_area > best_area:
                best_area = bbox_area
                best_bbox = (x_min, y_min, x_max, y_max)
                best_pixel_coords = pixel_pts
                best_norm_coords = norm_pts

        return FaceDetectionResult(
            detected=True,
            num_faces=num_faces,
            multiple_faces=multiple_faces,
            landmarks_pixel=best_pixel_coords,
            landmarks_normalized=best_norm_coords,
            bounding_box=best_bbox
        )

    def close(self) -> None:
        """Releases MediaPipe FaceMesh resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
