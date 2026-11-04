"""
=============================================================================
Eye Tracker Module
=============================================================================
Extracts ocular landmarks for left and right eyes, computes individual and
average Eye Aspect Ratios (EAR), and provides contour geometries for visualization.
=============================================================================
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import config
from features.ear import calculate_ear


@dataclass
class EyeTrackingResult:
    """Encapsulates Eye Aspect Ratio computations and landmark coordinates."""
    left_ear: float
    right_ear: float
    average_ear: float
    left_eye_pts: np.ndarray        # Shape: (6, 2)
    right_eye_pts: np.ndarray       # Shape: (6, 2)
    left_contour: np.ndarray        # Shape: (N, 2)
    right_contour: np.ndarray       # Shape: (N, 2)


class EyeTracker:
    """
    Tracks both eyes and computes left, right, and average EAR per frame.
    """

    def __init__(
        self,
        left_indices: List[int] = config.LEFT_EYE_EAR_INDICES,
        right_indices: List[int] = config.RIGHT_EYE_EAR_INDICES,
        left_contour_indices: List[int] = config.LEFT_EYE_CONTOUR_INDICES,
        right_contour_indices: List[int] = config.RIGHT_EYE_CONTOUR_INDICES
    ):
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.left_contour_indices = left_contour_indices
        self.right_contour_indices = right_contour_indices

    def track(self, landmarks_pixel: np.ndarray) -> Optional[EyeTrackingResult]:
        """
        Computes EAR and extracts ocular contours given 2D pixel landmarks.

        Args:
            landmarks_pixel: np.ndarray of shape (N, 2) containing face landmark coordinates.

        Returns:
            EyeTrackingResult if landmarks are valid, else None.
        """
        if landmarks_pixel is None or len(landmarks_pixel) < 468:
            return None

        # Extract 6-point EAR landmarks
        left_eye_pts = landmarks_pixel[self.left_indices]
        right_eye_pts = landmarks_pixel[self.right_indices]

        # Calculate EAR
        left_ear = calculate_ear(left_eye_pts)
        right_ear = calculate_ear(right_eye_pts)
        average_ear = (left_ear + right_ear) / 2.0

        # Extract full eye contours for drawing
        left_contour = landmarks_pixel[self.left_contour_indices]
        right_contour = landmarks_pixel[self.right_contour_indices]

        return EyeTrackingResult(
            left_ear=left_ear,
            right_ear=right_ear,
            average_ear=average_ear,
            left_eye_pts=left_eye_pts,
            right_eye_pts=right_eye_pts,
            left_contour=left_contour,
            right_contour=right_contour
        )
