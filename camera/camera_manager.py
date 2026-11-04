"""
=============================================================================
Camera Manager Module
=============================================================================
Manages video capture from USB webcams, handles frame acquisition, real-time
FPS computation with moving-average filtering, and graceful hardware release.
=============================================================================
"""

from mediapipe.python.solutions import selfie_segmentation
import time
import cv2
import numpy as np
from typing import Optional, Tuple
import config


class CameraManager:
    """
    Manages OpenCV VideoCapture stream with error recovery and FPS monitoring.
    """

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        target_fps: int = config.TARGET_FPS,
        backend: str = config.CAMERA_BACKEND
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.backend_str = backend

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

        # FPS calculation state
        self._prev_frame_time = time.perf_counter()
        self._frame_times = []
        self._fps_window_size = 20
        self.current_fps = float(target_fps)

    def start(self) -> bool:
        """
        Initializes and opens the camera stream.
        Tries DirectShow backend first on Windows, falls back to default.
        """
        print(f"[CameraManager] Initializing camera index {self.camera_index}...")

        backend = cv2.CAP_DSHOW if self.backend_str == "DSHOW" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, backend)

        if not self.cap.isOpened():
            print(f"[CameraManager] Failed with DSHOW backend. Trying default backend...")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print(f"[CameraManager] ERROR: Could not open camera {self.camera_index}!")
            self.is_opened = False
            return False

        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read actual acquired dimensions
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CameraManager] Camera ready: Resolution={actual_w}x{actual_h}, Target FPS={self.target_fps} (Device reported: {actual_fps})")

        self.is_opened = True
        self._prev_frame_time = time.perf_counter()
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads a single frame from the camera and computes instantaneous/smoothed FPS.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success_flag, bgr_image_array)
        """
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        if config.CAMERA_MIRROR:
            frame = cv2.flip(frame, 1)

        # Update FPS calculation
        now = time.perf_counter()
        dt = now - self._prev_frame_time
        self._prev_frame_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            self._frame_times.append(instant_fps)
            if len(self._frame_times) > self._fps_window_size:
                self._frame_times.pop(0)
            self.current_fps = sum(self._frame_times) / len(self._frame_times)

        return True, frame

    def get_fps(self) -> float:
        """Returns the current smoothed FPS."""
        return self.current_fps

    def release(self) -> None:
        """Safely releases the webcam capture device."""
        if self.cap is not None:
            print("[CameraManager] Releasing camera capture device...")
            self.cap.release()
            self.cap = None
        self.is_opened = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
