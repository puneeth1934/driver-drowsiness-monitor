"""
=============================================================================
Head Pose Estimation Module (solvePnP)
=============================================================================
Estimates 3D head orientation (Pitch, Yaw, Roll) in degrees using OpenCV's
Perspective-n-Point (solvePnP) algorithm and MediaPipe 2D facial landmarks
mapped to a standard 3D anthropometric facial model.
=============================================================================
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2
import numpy as np
import config


@dataclass
class HeadPoseResult:
    """Encapsulates 3D head pose angles, orientation classification, and nose projection."""
    pitch: float                # Nodding: positive = up, negative = down (degrees)
    yaw: float                  # Looking: positive = right, negative = left (degrees)
    roll: float                 # Tilting: positive = clockwise, negative = counter-clockwise (degrees)
    orientation_label: str      # "FORWARD", "LOOKING LEFT", "LOOKING RIGHT", "LOOKING DOWN", etc.
    significant_movement: bool  # True if angles deviate substantially from baseline forward
    nose_end_point_2d: Optional[Tuple[int, int]] = None  # Projected 2D endpoint for drawing 3D vector


class HeadPoseEstimator:
    """
    Computes 3D head orientation using solvePnP.
    """

    def __init__(
        self,
        model_points_3d: List[Tuple[float, float, float]] = config.HEAD_POSE_3D_MODEL_POINTS,
        landmark_indices_2d: List[int] = config.HEAD_POSE_2D_LANDMARK_INDICES,
        pitch_down_thresh: float = config.PITCH_LOOK_DOWN_THRESHOLD,
        pitch_up_thresh: float = config.PITCH_LOOK_UP_THRESHOLD,
        yaw_left_thresh: float = config.YAW_LOOK_LEFT_THRESHOLD,
        yaw_right_thresh: float = config.YAW_LOOK_RIGHT_THRESHOLD,
        roll_thresh: float = config.ROLL_TILT_THRESHOLD
    ):
        self.model_points = np.array(model_points_3d, dtype=np.float64)
        self.landmark_indices = landmark_indices_2d
        self.pitch_down_thresh = pitch_down_thresh
        self.pitch_up_thresh = pitch_up_thresh
        self.yaw_left_thresh = yaw_left_thresh
        self.yaw_right_thresh = yaw_right_thresh
        self.roll_thresh = roll_thresh

        # Camera matrix cache
        self._cached_cam_matrix = None
        self._cached_dims = (0, 0)
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def _get_camera_matrix(self, width: int, height: int) -> np.ndarray:
        """Constructs an approximated camera intrinsic matrix based on image dimensions."""
        if self._cached_cam_matrix is None or self._cached_dims != (width, height):
            focal_length = float(width)
            center_x = width / 2.0
            center_y = height / 2.0
            self._cached_cam_matrix = np.array(
                [[focal_length, 0.0, center_x],
                 [0.0, focal_length, center_y],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64
            )
            self._cached_dims = (width, height)
        return self._cached_cam_matrix

    def estimate(self, landmarks_pixel: np.ndarray, frame_width: int, frame_height: int) -> HeadPoseResult:
        """
        Estimates Pitch, Yaw, and Roll from 2D pixel landmarks.

        Args:
            landmarks_pixel: np.ndarray of shape (N, 2)
            frame_width: Image width in pixels
            frame_height: Image height in pixels

        Returns:
            HeadPoseResult with Euler angles in degrees and orientation classification.
        """
        if landmarks_pixel is None or len(landmarks_pixel) < 468:
            return HeadPoseResult(
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                orientation_label="UNKNOWN",
                significant_movement=False,
                nose_end_point_2d=None
            )

        # 2D image points from landmarks
        image_points = np.array(
            [landmarks_pixel[idx] for idx in self.landmark_indices],
            dtype=np.float64
        )

        camera_matrix = self._get_camera_matrix(frame_width, frame_height)

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return HeadPoseResult(
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                orientation_label="UNKNOWN",
                significant_movement=False,
                nose_end_point_2d=None
            )

        # Convert rotation vector to 3x3 rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles
        # angles returned by RQDecomp3x3 are in degrees
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Project a 3D point (500mm in front of the nose) to 2D image plane for vector drawing
        nose_end_3d = np.array([(0.0, 0.0, 500.0)], dtype=np.float64)
        nose_end_2d, _ = cv2.projectPoints(
            nose_end_3d,
            rvec,
            tvec,
            camera_matrix,
            self._dist_coeffs
        )
        nose_proj_pt = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))

        # Classify orientation
        labels = []
        significant = False

        if yaw < self.yaw_left_thresh:
            labels.append("LOOKING LEFT")
            significant = True
        elif yaw > self.yaw_right_thresh:
            labels.append("LOOKING RIGHT")
            significant = True

        if pitch < self.pitch_down_thresh:
            labels.append("LOOKING DOWN")
            significant = True
        elif pitch > self.pitch_up_thresh:
            labels.append("LOOKING UP")
            significant = True

        if abs(roll) > self.roll_thresh:
            labels.append("HEAD TILTED")
            significant = True

        orientation_str = " / ".join(labels) if labels else "FORWARD"

        return HeadPoseResult(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            orientation_label=orientation_str,
            significant_movement=significant,
            nose_end_point_2d=nose_proj_pt
        )
