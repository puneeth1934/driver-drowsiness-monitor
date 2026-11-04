"""
=============================================================================
Driver Drowsiness Detection System - Configuration Module
=============================================================================
Central configuration file containing all tunable parameters, camera settings,
thresholds, window sizes, and file paths.

All parameters can be tuned or calibrated for individual drivers and environments
without modifying core business logic.
=============================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env if available
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = DATA_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CSV_LOG_PATH = DATA_DIR / "eye_tracking_data.csv"
CALIBRATION_FILE_PATH = DATA_DIR / "calibration.json"

# =============================================================================
# CAMERA & ACQUISITION SETTINGS
# =============================================================================
CAMERA_INDEX = 0           # Default USB webcam index (0 = default internal/external camera)
CAMERA_MIRROR = True       # Mirror the camera feed (flip horizontally)
FRAME_WIDTH = 640      # Capture width in pixels
FRAME_HEIGHT = 480     # Capture height in pixels
TARGET_FPS = 20            # Target camera frames per second
CAMERA_BACKEND = "DSHOW"     # Windows DirectShow backend ("DSHOW" or "DEFAULT")

# =============================================================================
# MEDIAPIPE FACE MESH CONFIGURATION
# =============================================================================
MAX_NUM_FACES = 2            # Allows detection of primary driver + detecting multiple faces
REFINE_LANDMARKS = True      # Enables iris landmarks and fine eye contour tracking
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# =============================================================================
# EYE ASPECT RATIO (EAR) & EYE CLOSURE CONFIGURATION
# =============================================================================
# Note: Thresholds can be calibrated per user via calibration mode (Press 'C').
EAR_CLOSED_THRESHOLD = 0.21         # EAR below this threshold is treated as closed/closing
EAR_OPEN_THRESHOLD = 0.26           # EAR above this threshold is treated as fully open (hysteresis)

# Temporal duration thresholds (in seconds)
MIN_BLINK_DURATION_SEC = 0.06       # Minimum duration (60 ms) for a valid physiological blink
MAX_BLINK_DURATION_SEC = 0.80       # Maximum duration (450 ms) for a normal blink (longer is closure)
PROLONGED_CLOSURE_THRESHOLD_SEC = 2.0 # Duration (seconds) of continuous closure to flag prolonged closure

# =============================================================================
# PERCLOS (Percentage of Eye Closure) CONFIGURATION
# =============================================================================
# Standard NHTSA / automotive DMS rolling observation window
PERCLOS_WINDOW_SECONDS = 60.0       # Rolling observation window duration in seconds
PERCLOS_ALERT_THRESHOLD = 0.25      # 15% eye closure in rolling window (informative reference)

# =============================================================================
# YAWNING (MOUTH ASPECT RATIO - MAR) CONFIGURATION
# =============================================================================
MAR_YAWN_THRESHOLD = 0.20   # MAR above this threshold indicates significant mouth opening
MIN_YAWN_DURATION_SEC = 1.2    # Sustained opening duration (seconds) required to register a yawn
YAWN_COOLDOWN_SEC = 3.0             # Cooldown period before another distinct yawn can be triggered

# =============================================================================
# HEAD POSE ESTIMATION CONFIGURATION
# =============================================================================
# Head orientation alert angle limits in degrees
PITCH_LOOK_DOWN_THRESHOLD = -18.0   # Nodding down (degrees)
PITCH_LOOK_UP_THRESHOLD = 20.0      # Looking up (degrees)
YAW_LOOK_LEFT_THRESHOLD = -25.0     # Looking left (degrees)
YAW_LOOK_RIGHT_THRESHOLD = 25.0     # Looking right (degrees)
ROLL_TILT_THRESHOLD = 25.0          # Head tilt (degrees)

# =============================================================================
# DATA LOGGING CONFIGURATION
# =============================================================================
LOGGING_FREQUENCY_HZ = 10.0         # CSV logging rate in Hz (10 samples per sec = every 100ms)
LOG_TO_CSV = True                   # Enable or disable background CSV writing

# =============================================================================
# MEDIAPIPE FACIAL LANDMARK INDICES (Standard FaceMesh 468/478 format)
# =============================================================================
# 6-Point Left Eye EAR Landmark Indices: [p1_corner, p2_top1, p3_top2, p4_corner, p5_bot2, p6_bot1]
LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]

# 6-Point Right Eye EAR Landmark Indices: [p1_corner, p2_top1, p3_top2, p4_corner, p5_bot2, p6_bot1]
RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]

# Detailed Eye Contours for drawing
LEFT_EYE_CONTOUR_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# 6-Point Mouth MAR Indices: [corner_left, top_1, top_2, corner_right, bot_2, bot_1]
MOUTH_MAR_INDICES = [78, 81, 311, 308, 402, 178]

# Mouth Contours for drawing
MOUTH_OUTER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
MOUTH_INNER_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# 3D Generic Head Model Reference Points (x, y, z in mm) for solvePnP
HEAD_POSE_3D_MODEL_POINTS = [
    (0.0, 0.0, 0.0),             # Nose tip (index 1)
    (0.0, -330.0, -65.0),        # Chin (index 199 / 152)
    (-225.0, 170.0, -135.0),     # Left eye outer corner (index 263)
    (225.0, 170.0, -135.0),      # Right eye outer corner (index 33)
    (-150.0, -150.0, -125.0),    # Left mouth corner (index 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (index 291)
]

# Corresponding 2D Landmark Indices in MediaPipe FaceMesh
HEAD_POSE_2D_LANDMARK_INDICES = [1, 199, 263, 33, 61, 291]

# =============================================================================
# UI & COLOR THEMES (BGR Format for OpenCV)
# =============================================================================
COLOR_BG_CARD = (25, 25, 30)         # Dark slate panel background
COLOR_TEXT_WHITE = (245, 245, 245)   # Primary text
COLOR_TEXT_MUTED = (160, 160, 160)   # Secondary text
COLOR_NOMINAL = (70, 210, 120)       # Green (nominal / open / ready)
COLOR_WARNING = (30, 180, 255)       # Amber / Yellow (closing / caution)
COLOR_ALERT = (50, 50, 235)          # Red (closed / prolonged / yawn / alert)
COLOR_CYAN = (235, 205, 50)          # Cyan / Blue for accents

# =============================================================================
# PHASE 3: VEHICLE DYNAMICS & GPS CONFIGURATION
# =============================================================================
# MPU6050 Granular Movement Detection Thresholds
MPU_ACCEL_SURGE_THRESHOLD_G = 0.28     # Sudden acceleration threshold (+X)
MPU_BRAKE_THRESHOLD_G = -0.28          # Sudden braking threshold (-X)
MPU_SIDE_JERK_THRESHOLD_G = 0.30       # Lateral side jerk threshold (|Y|)
MPU_TURN_GYRO_THRESHOLD_DPS = 35.0     # Left/Right yaw turn rate threshold (|Z|)
MPU_JERK_THRESHOLD_G = 0.40            # Dynamic jerk/shock threshold
MPU_HIGH_ROTATION_THRESHOLD_DPS = 65.0 # Total rotational velocity magnitude (deg/s)
MPU_IMPACT_THRESHOLD_G = 1.00          # High acceleration spike indicating severe shock/impact
MPU_STAGNANT_ACCEL_MAX_G = 0.08        # Acceleration deviation limit for stagnant state
MPU_STAGNANT_GYRO_MAX_DPS = 8.0        # Gyro limit for stagnant state
MPU_FILTER_ALPHA = 0.35                # EMA low-pass filter factor to damp sensor vibration

# Road Speed Limit Lookup Configuration (OpenStreetMap Overpass API)
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
SPEED_LIMIT_CACHE_TTL_SEC = 30.0       # Cache limit duration in seconds
SPEED_LIMIT_MIN_DISTANCE_M = 50.0      # Minimum distance traveled before refreshing road lookup
SPEED_LIMIT_TIMEOUT_SEC = 2.0          # Non-blocking network timeout for HTTP queries

# =============================================================================
# CLOUD SYNCHRONIZATION CONFIGURATION (Firebase Realtime Database)
# =============================================================================
DEVICE_ID = os.getenv("DEVICE_ID", "DDS-VEM-001")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "")
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
FIREBASE_SYNC_INTERVAL_SEC = float(os.getenv("FIREBASE_SYNC_INTERVAL_SEC", "1.0"))


