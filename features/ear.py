"""
=============================================================================
Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR) Calculation Module
=============================================================================
Implements pure mathematical formulations for Eye Aspect Ratio (Soukupová & Čech, 2016)
and Mouth Aspect Ratio (MAR) using Euclidean distance between facial landmark points.
=============================================================================
"""

import numpy as np
from typing import Union, List


def euclidean_distance(pt1: Union[np.ndarray, List[float]], pt2: Union[np.ndarray, List[float]]) -> float:
    """
    Computes 2D Euclidean distance between two points.

    Args:
        pt1: Coordinate (x, y)
        pt2: Coordinate (x, y)

    Returns:
        float: Euclidean distance ||pt1 - pt2||_2
    """
    p1 = np.asarray(pt1, dtype=np.float64)[:2]
    p2 = np.asarray(pt2, dtype=np.float64)[:2]
    return float(np.linalg.norm(p1 - p2))


def calculate_ear(eye_landmarks: Union[np.ndarray, List]) -> float:
    """
    Calculates Eye Aspect Ratio (EAR) from 6 2D eye landmark points.

    Formula (Soukupová and Čech, 2016):
        EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2.0 * ||p1 - p4|| )

    Landmark Layout:
        p1: Outer/Inner eye corner
        p2: Upper eyelid landmark 1
        p3: Upper eyelid landmark 2
        p4: Opposite eye corner
        p5: Lower eyelid landmark 2
        p6: Lower eyelid landmark 1

    Args:
        eye_landmarks: Array-like of shape (6, 2) representing the 6 landmark coordinates.

    Returns:
        float: Calculated EAR value (typically 0.25-0.38 for open eye, <0.20 for closed eye).
               Returns 0.0 if points are invalid or degenerate.
    """
    pts = np.asarray(eye_landmarks, dtype=np.float64)
    if pts.shape[0] < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = pts[0], pts[1], pts[2], pts[3], pts[4], pts[5]

    # Vertical Euclidean distances
    dist_v1 = euclidean_distance(p2, p6)
    dist_v2 = euclidean_distance(p3, p5)

    # Horizontal Euclidean distance
    dist_h = euclidean_distance(p1, p4)

    if dist_h < 1e-6:
        return 0.0

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return float(ear)


def calculate_mar(mouth_landmarks: Union[np.ndarray, List]) -> float:
    """
    Calculates Mouth Aspect Ratio (MAR) from 6 2D mouth landmark points.

    Formula:
        MAR = ( ||m2 - m6|| + ||m3 - m5|| ) / ( 2.0 * ||m1 - m4|| )

    Landmark Layout:
        m1: Left mouth corner
        m2: Upper lip point 1
        m3: Upper lip point 2
        m4: Right mouth corner
        m5: Lower lip point 2
        m6: Lower lip point 1

    Args:
        mouth_landmarks: Array-like of shape (6, 2) representing 6 mouth landmark coordinates.

    Returns:
        float: Calculated MAR value (typically 0.15-0.35 for normal mouth, >0.60 for wide yawn).
    """
    pts = np.asarray(mouth_landmarks, dtype=np.float64)
    if pts.shape[0] < 6:
        return 0.0

    m1, m2, m3, m4, m5, m6 = pts[0], pts[1], pts[2], pts[3], pts[4], pts[5]

    dist_v1 = euclidean_distance(m2, m6)
    dist_v2 = euclidean_distance(m3, m5)
    dist_h = euclidean_distance(m1, m4)

    if dist_h < 1e-6:
        return 0.0

    mar = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return float(mar)
