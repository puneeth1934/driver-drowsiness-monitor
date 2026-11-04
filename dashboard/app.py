"""
=============================================================================
Driver Drowsiness Detection System — Streamlit Cockpit Dashboard
=============================================================================
Real-time web monitoring interface integrating:
1. Camera & Facial Eye/Pose Tracking (Live Vision Subsystem)
2. Heart Rate & HRV Physiological Sensor Cards (Phase 2 Interface)
3. Steering Angle & Pattern Sensor Cards (Phase 3 Interface)
4. Multimodal Drowsiness State Evaluation
5. Buzzer & Haptic Vibration Actuator Subsystem

Reuses existing CameraManager, MediaPipe FaceMesh, and detection modules.
=============================================================================
"""

import os
import sys
import time
import collections
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import serial
import threading
import requests

# =============================================================================
# ESP32 / MPU6050 & NEO-6M GPS VEHICLE SENSOR SUBSYSTEM
# =============================================================================

VEHICLE_PORT = "COM5"
VEHICLE_BAUD = 115200


@st.cache_resource
def get_vehicle_serial():
    try:
        connection = serial.Serial(
            VEHICLE_PORT,
            VEHICLE_BAUD,
            timeout=0.1,
            write_timeout=0.1
        )
        time.sleep(2)
        return connection
    except Exception as e:
        print("ESP32 connection error:", e)
        return None


vehicle_serial = get_vehicle_serial()
vehicle_sensor_connected = vehicle_serial is not None

# Thread-safe OpenStreetMap Overpass Speed Limit Cache
speed_limit_cache: Dict[str, Any] = {
    "last_lat": None,
    "last_lng": None,
    "last_query_time": 0.0,
    "current_road": None,
    "speed_limit": None,
    "query_in_progress": False
}


def query_osm_speed_limit_async(lat: float, lng: float) -> None:
    """Queries OpenStreetMap Overpass API for road speed limit in a background daemon thread."""
    now = time.time()
    if speed_limit_cache["query_in_progress"]:
        return

    # Rate limiting: only query every 30s or when moved > 50m
    if (now - speed_limit_cache["last_query_time"]) < getattr(config, "SPEED_LIMIT_CACHE_TTL_SEC", 30.0):
        if speed_limit_cache["last_lat"] is not None:
            dist_m = (((lat - speed_limit_cache["last_lat"])**2 + (lng - speed_limit_cache["last_lng"])**2)**0.5) * 111000.0
            if dist_m < getattr(config, "SPEED_LIMIT_MIN_DISTANCE_M", 50.0):
                return

    speed_limit_cache["query_in_progress"] = True
    speed_limit_cache["last_lat"] = lat
    speed_limit_cache["last_lng"] = lng
    speed_limit_cache["last_query_time"] = now

    def worker():
        try:
            url = getattr(config, "OVERPASS_API_URL", "https://overpass-api.de/api/interpreter")
            timeout_sec = getattr(config, "SPEED_LIMIT_TIMEOUT_SEC", 2.0)
            overpass_query = f"""[out:json][timeout:2];way(around:45,{lat},{lng})["highway"]["maxspeed"];out tags 1;"""
            resp = requests.post(url, data={"data": overpass_query}, timeout=timeout_sec)
            if resp.status_code == 200:
                data = resp.json()
                elements = data.get("elements", [])
                if elements and "tags" in elements[0]:
                    tags = elements[0]["tags"]
                    raw_maxspeed = tags.get("maxspeed", "")
                    road_name = tags.get("name", "Identified Roadway")
                    tokens = raw_maxspeed.split()
                    if tokens:
                        val = float(tokens[0])
                        if "mph" in raw_maxspeed.lower():
                            val *= 1.60934
                        speed_limit_cache["speed_limit"] = int(round(val))
                        speed_limit_cache["current_road"] = road_name
                        return
            speed_limit_cache["speed_limit"] = None
        except Exception:
            speed_limit_cache["speed_limit"] = None
        finally:
            speed_limit_cache["query_in_progress"] = False

    threading.Thread(target=worker, daemon=True).start()


# Shared Vehicle Telemetry State
vehicle_telemetry: Dict[str, Any] = {
    "mpu_connected": False,
    "ax": 0.0,
    "ay": 0.0,
    "az": 1.0,
    "gx": 0.0,
    "gy": 0.0,
    "gz": 0.0,
    "filtered_ax": 0.0,
    "filtered_ay": 0.0,
    "filtered_az": 1.0,
    "filtered_gx": 0.0,
    "filtered_gy": 0.0,
    "filtered_gz": 0.0,
    "filtered_accel_mag": 1.0,
    "filtered_gyro_mag": 0.0,
    "movement_state": "STAGNANT (AT REST)",
    "movement_badge": "nominal",
    "movement_detail": "Vehicle at complete standstill",
    "gps_connected": False,
    "latitude": 0.0,
    "longitude": 0.0,
    "speed_kmh": 0.0,
    "satellites": 0,
    "fix_valid": False,
    "current_road": None,
    "speed_limit": None,
    "overspeeding": False,
}


def read_vehicle_sensor() -> Dict[str, Any]:
    """Reads available MPU6050 and NEO-6M GPS packets without blocking Streamlit."""
    global vehicle_serial, vehicle_telemetry

    if vehicle_serial is None or not vehicle_serial.is_open:
        return vehicle_telemetry

    try:
        lines_read = 0
        while vehicle_serial.in_waiting > 0 and lines_read < 10:
            lines_read += 1
            line = vehicle_serial.readline().decode("utf-8", errors="ignore").strip()

            if line.startswith("MPU,"):
                parts = line.split(",")
                if len(parts) >= 7:
                    try:
                        ax = float(parts[1])
                        ay = float(parts[2])
                        az = float(parts[3])
                        gx = float(parts[4])
                        gy = float(parts[5])
                        gz = float(parts[6])

                        # Low-pass EMA filtering across all 3 linear & 3 angular axes
                        alpha = getattr(config, "MPU_FILTER_ALPHA", 0.35)
                        prev_ax = vehicle_telemetry.get("filtered_ax", 0.0)
                        prev_ay = vehicle_telemetry.get("filtered_ay", 0.0)
                        prev_az = vehicle_telemetry.get("filtered_az", 1.0)
                        prev_gx = vehicle_telemetry.get("filtered_gx", 0.0)
                        prev_gy = vehicle_telemetry.get("filtered_gy", 0.0)
                        prev_gz = vehicle_telemetry.get("filtered_gz", 0.0)

                        f_ax = alpha * ax + (1.0 - alpha) * prev_ax
                        f_ay = alpha * ay + (1.0 - alpha) * prev_ay
                        f_az = alpha * az + (1.0 - alpha) * prev_az
                        f_gx = alpha * gx + (1.0 - alpha) * prev_gx
                        f_gy = alpha * gy + (1.0 - alpha) * prev_gy
                        f_gz = alpha * gz + (1.0 - alpha) * prev_gz

                        raw_a_mag = (ax**2 + ay**2 + az**2) ** 0.5
                        raw_g_mag = (gx**2 + gy**2 + gz**2) ** 0.5

                        f_a = alpha * raw_a_mag + (1.0 - alpha) * vehicle_telemetry["filtered_accel_mag"]
                        f_g = alpha * raw_g_mag + (1.0 - alpha) * vehicle_telemetry["filtered_gyro_mag"]

                        # Acceleration deviation from static 1.0g gravity
                        delta_g = abs(f_a - 1.0)

                        # Movement Detection Thresholds from config
                        impact_thresh = getattr(config, "MPU_IMPACT_THRESHOLD_G", 1.00)
                        accel_thresh = getattr(config, "MPU_ACCEL_SURGE_THRESHOLD_G", 0.28)
                        brake_thresh = getattr(config, "MPU_BRAKE_THRESHOLD_G", -0.28)
                        turn_thresh = getattr(config, "MPU_TURN_GYRO_THRESHOLD_DPS", 35.0)
                        side_thresh = getattr(config, "MPU_SIDE_JERK_THRESHOLD_G", 0.30)
                        jerk_thresh = getattr(config, "MPU_JERK_THRESHOLD_G", 0.40)
                        rot_thresh = getattr(config, "MPU_HIGH_ROTATION_THRESHOLD_DPS", 65.0)
                        stag_accel = getattr(config, "MPU_STAGNANT_ACCEL_MAX_G", 0.08)
                        stag_gyro = getattr(config, "MPU_STAGNANT_GYRO_MAX_DPS", 8.0)

                        # Movement Classification:
                        # 1. Collision / High-G Impact
                        if delta_g >= impact_thresh:
                            movement_state = "POSSIBLE IMPACT"
                            movement_badge = "alert"
                            movement_detail = f"High-G collision shock: {f_a:.2f}g"
                        # 2. Sudden Acceleration (Forward surge)
                        elif f_ax >= accel_thresh:
                            movement_state = "SUDDEN ACCELERATION"
                            movement_badge = "warning"
                            movement_detail = f"Forward surge: +{f_ax:.2f}g"
                        # 3. Sudden Braking (Rapid deceleration)
                        elif f_ax <= brake_thresh:
                            movement_state = "SUDDEN BRAKING"
                            movement_badge = "warning"
                            movement_detail = f"Abrupt deceleration: {f_ax:.2f}g"
                        # 4. Sudden Right Turn (Clockwise yaw)
                        elif f_gz >= turn_thresh:
                            movement_state = "SUDDEN RIGHT TURN"
                            movement_badge = "warning"
                            movement_detail = f"Clockwise turn rate: +{f_gz:.1f}°/s"
                        # 5. Sudden Left Turn (Counter-clockwise yaw)
                        elif f_gz <= -turn_thresh:
                            movement_state = "SUDDEN LEFT TURN"
                            movement_badge = "warning"
                            movement_detail = f"Counter-clockwise turn rate: {f_gz:.1f}°/s"
                        # 6. Side Jerk (Lateral sway)
                        elif abs(f_ay) >= side_thresh:
                            side_dir = "RIGHT" if f_ay > 0 else "LEFT"
                            movement_state = f"SIDE JERK ({side_dir})"
                            movement_badge = "warning"
                            movement_detail = f"Lateral sway jolt: {abs(f_ay):.2f}g"
                        # 7. Forward / Backward Jerk (Multiaxial longitudinal shock)
                        elif delta_g >= jerk_thresh:
                            movement_state = "SUDDEN FORWARD/BACKWARD JERK"
                            movement_badge = "warning"
                            movement_detail = f"Longitudinal shock: {delta_g:.2f}g"
                        # 8. High Rotational Movement
                        elif f_g >= rot_thresh:
                            movement_state = "HIGH ROTATIONAL MOVEMENT"
                            movement_badge = "warning"
                            movement_detail = f"High angular rate: {f_g:.1f}°/s"
                        # 9. Stagnant (At rest)
                        elif delta_g < stag_accel and f_g < stag_gyro:
                            movement_state = "STAGNANT (AT REST)"
                            movement_badge = "nominal"
                            movement_detail = "Vehicle at complete standstill"
                        # 10. Normal Vehicle Movement
                        else:
                            movement_state = "NORMAL VEHICLE MOVEMENT"
                            movement_badge = "nominal"
                            movement_detail = "Smooth driving profile"

                        # Dynamic speed estimation:
                        # 1. Primary: Real NEO-6M GPS ground speed when 3D satellite fix is established
                        # 2. Inertial fallback: Computes velocity dynamically from physical MPU motion when testing or indoors
                        now_mpu = time.time()
                        last_t = vehicle_telemetry.get("last_mpu_time", now_mpu)
                        dt = min(max(now_mpu - last_t, 0.02), 0.3)
                        vehicle_telemetry["last_mpu_time"] = now_mpu

                        current_speed = vehicle_telemetry.get("speed_kmh", 0.0)

                        if vehicle_telemetry.get("fix_valid", False) and vehicle_telemetry.get("raw_gps_speed", 0.0) > 1.0:
                            current_speed = vehicle_telemetry["raw_gps_speed"]
                        else:
                            # Inertial velocity response from physical sensor motion
                            if movement_state == "STAGNANT (AT REST)":
                                current_speed = max(0.0, current_speed * 0.70 - 1.5)
                                if current_speed < 0.5:
                                    current_speed = 0.0
                            elif movement_state == "SUDDEN BRAKING":
                                current_speed = max(0.0, current_speed - abs(f_ax) * 45.0 * dt)
                            elif movement_state == "SUDDEN ACCELERATION":
                                current_speed = min(120.0, current_speed + abs(f_ax) * 40.0 * dt)
                            elif delta_g > 0.12 or f_g > 15.0:
                                motion_factor = max(delta_g, f_g / 45.0)
                                current_speed = min(95.0, max(current_speed, motion_factor * 32.0))
                            else:
                                current_speed = max(0.0, current_speed - 10.0 * dt)

                        current_speed = round(current_speed, 1)
                        curr_limit = speed_limit_cache["speed_limit"]
                        overspeeding = (curr_limit is not None and current_speed > curr_limit)

                        vehicle_telemetry.update({
                            "mpu_connected": True,
                            "ax": ax, "ay": ay, "az": az,
                            "gx": gx, "gy": gy, "gz": gz,
                            "filtered_ax": f_ax,
                            "filtered_ay": f_ay,
                            "filtered_az": f_az,
                            "filtered_gx": f_gx,
                            "filtered_gy": f_gy,
                            "filtered_gz": f_gz,
                            "filtered_accel_mag": f_a,
                            "filtered_gyro_mag": f_g,
                            "speed_kmh": current_speed,
                            "overspeeding": overspeeding,
                            "movement_state": movement_state,
                            "movement_badge": movement_badge,
                            "movement_detail": movement_detail,
                        })
                    except ValueError:
                        pass

            elif line.startswith("GPS,"):
                parts = line.split(",")
                if len(parts) >= 6:
                    try:
                        lat = float(parts[1])
                        lng = float(parts[2])
                        speed = float(parts[3])
                        sats = int(parts[4])
                        valid_flag = int(parts[5])

                        fix_valid = (valid_flag == 1 and abs(lat) > 0.0001 and abs(lng) > 0.0001)

                        if fix_valid:
                            query_osm_speed_limit_async(lat, lng)

                        curr_limit = speed_limit_cache["speed_limit"]
                        curr_road = speed_limit_cache["current_road"]

                        vehicle_telemetry["raw_gps_speed"] = speed
                        if fix_valid and speed > 1.0:
                            vehicle_telemetry["speed_kmh"] = round(speed, 1)

                        overspeeding = (curr_limit is not None and vehicle_telemetry["speed_kmh"] > curr_limit)

                        vehicle_telemetry.update({
                            "gps_connected": True,
                            "latitude": lat,
                            "longitude": lng,
                            "satellites": sats,
                            "fix_valid": fix_valid,
                            "current_road": curr_road,
                            "speed_limit": curr_limit,
                            "overspeeding": overspeeding,
                        })
                    except ValueError:
                        pass

        # Discard stale buffer overflow if excessive lines accumulated
        if vehicle_serial.in_waiting > 150:
            vehicle_serial.reset_input_buffer()

        return vehicle_telemetry

    except (serial.SerialException, OSError) as e:
        print("Vehicle sensor serial reading error:", e)
        return vehicle_telemetry


# Add project root directory to path for clean modular imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from cloud.firebase_sync import FirebaseCloudPublisher


@st.cache_resource
def get_cloud_publisher():
    try:
        publisher = FirebaseCloudPublisher(
            device_id=getattr(config, "DEVICE_ID", "DDS-VEM-001"),
            database_url=getattr(config, "FIREBASE_DATABASE_URL", ""),
            credentials_path=getattr(config, "FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json"),
            sync_interval_sec=getattr(config, "FIREBASE_SYNC_INTERVAL_SEC", 1.0)
        )
        publisher.start()
        return publisher
    except Exception as e:
        print("Firebase publisher initialization warning:", e)
        return None


cloud_publisher = get_cloud_publisher()


# =============================================================================
# STREAMLIT PAGE CONFIGURATION & INJECTED STYLES
# =============================================================================
st.set_page_config(
    page_title="Driver Drowsiness Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Base Dark Cockpit Theme */
.stApp {
    background-color: #0b0e14;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Glassmorphism Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(20, 26, 38, 0.85), rgba(13, 17, 23, 0.95));
    border: 1px solid rgba(70, 85, 110, 0.35);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(0, 212, 255, 0.5);
    transform: translateY(-2px);
}

.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
}

.metric-unit {
    font-size: 0.85rem;
    font-weight: 500;
    color: #64748b;
    margin-left: 4px;
}

.metric-sub {
    font-size: 0.74rem;
    color: #64748b;
    margin-top: 4px;
}

/* Status Badges */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-nominal {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.badge-alert {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.5);
    animation: pulse 1.5s infinite;
}

.badge-standby {
    background: rgba(148, 163, 184, 0.12);
    color: #94a3b8;
    border: 1px solid rgba(148, 163, 184, 0.3);
}

.badge-cyan {
    background: rgba(6, 182, 212, 0.15);
    color: #06b6d4;
    border: 1px solid rgba(6, 182, 212, 0.4);
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* Main Drowsiness Banner */
.drowsiness-banner {
    border-radius: 12px;
    padding: 16px 22px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 25px rgba(0, 0, 0, 0.4);
}

.banner-nominal {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.2), rgba(6, 78, 59, 0.4));
    border-left: 6px solid #10b981;
}

.banner-warning {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.2), rgba(120, 53, 15, 0.4));
    border-left: 6px solid #f59e0b;
}

.banner-alert {
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.25), rgba(127, 29, 29, 0.5));
    border-left: 6px solid #ef4444;
    animation: pulse 1.2s infinite;
}

.banner-standby {
    background: linear-gradient(90deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.8));
    border-left: 6px solid #64748b;
}

/* Video Frame Container */
.video-frame-container {
    background: #000000;
    border-radius: 14px;
    border: 2px solid rgba(70, 85, 110, 0.4);
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
    margin-bottom: 16px;
}

/* Section Header */
.section-header {
    font-size: 0.95rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #38bdf8;
    margin-top: 14px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Sensor Placeholder Box */
.sensor-placeholder-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px dashed rgba(100, 116, 139, 0.4);
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS FOR HTML RENDERING
# =============================================================================
def render_metric_card(
    label: str,
    value: str,
    unit: str = "",
    status_badge: Optional[str] = None,
    badge_type: str = "nominal",
    sub_text: Optional[str] = None
) -> str:
    """Generates an HTML string for a telemetry card."""
    badge_html = ""
    if status_badge:
        badge_html = f'<span class="badge badge-{badge_type}">{status_badge}</span>'

    sub_html = ""
    if sub_text:
        sub_html = f'<div class="metric-sub">{sub_text}</div>'

    unit_html = f'<span class="metric-unit">{unit}</span>' if unit else ""

    return f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div class="metric-label">{label}</div>
            {badge_html}
        </div>
        <div class="metric-value">{value}{unit_html}</div>
        {sub_html}
    </div>
    """


def render_drowsiness_banner(status_text: str, severity: str, sub_message: str = "") -> str:
    """Generates an HTML banner for the master drowsiness decision."""
    badge_type = {
        "nominal": "badge-nominal",
        "warning": "badge-warning",
        "alert": "badge-alert",
        "standby": "badge-standby"
    }.get(severity, "badge-standby")

    banner_class = f"banner-{severity}"

    return f"""
    <div class="drowsiness-banner {banner_class}">
        <div>
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; font-weight: 600;">
                Master System Evaluation
            </div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #ffffff; margin-top: 2px;">
                {status_text}
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 2px;">
                {sub_message}
            </div>
        </div>
        <div>
            <span class="badge {badge_type}" style="font-size: 0.85rem; padding: 6px 14px;">
                {status_text}
            </span>
        </div>
    </div>
    """


# =============================================================================
# SIDEBAR CONTROLS & CONFIGURATION
# =============================================================================
with st.sidebar:
    st.image(
        "https://img.shields.io/badge/DRIVER_DROWSINESS-DMS_SYSTEM-00d4ff?style=for-the-badge&logo=shield",
        use_container_width=True
    )
    st.markdown("### 🎛️ System Control Panel")

    # Camera Configuration
    st.markdown("#### 📷 Video Acquisition")
    camera_idx_option = st.selectbox(
        "Camera Index (USB Webcam)",
        options=[1, 0, 2, 3],
        index=0,  # Default: 1 (USB Webcam)
        help="Select the camera index. Default is 1 for external USB webcam."
    )

    frame_width = st.selectbox("Resolution Width", [640, 1280], index=0)
    frame_height = st.selectbox("Resolution Height", [480, 720], index=0)

    # Visualization Options
    st.markdown("#### 🎨 Display Overlays")
    show_hud = st.checkbox("Render OpenCV HUD Overlay", value=True, help="Renders telemetry HUD directly on video stream")
    show_mesh = st.checkbox("Draw Facial Landmarks & Contours", value=True)
    show_pose = st.checkbox("Draw 3D Head Pose Vector", value=True)

    # Threshold Customization
    st.markdown("#### ⚙️ Detection Thresholds")
    ear_thresh = st.slider("EAR Closed Threshold", 0.15, 0.35, float(config.EAR_CLOSED_THRESHOLD), 0.01)
    closure_time_thresh = st.slider("Prolonged Closure Limit (s)", 0.8, 3.0, float(config.PROLONGED_CLOSURE_THRESHOLD_SEC), 0.1)
    perclos_alert_thresh = st.slider("PERCLOS Alert Threshold (%)", 5.0, 30.0, float(config.PERCLOS_ALERT_THRESHOLD * 100.0), 1.0)
    enable_csv_logging = st.checkbox("Log Telemetry to CSV", value=False)

    st.markdown("---")
    st.markdown("#### 🔌 Hardware & Cloud Peripherals")
    cloud_status_color = "#10b981" if (cloud_publisher and cloud_publisher.is_connected) else "#94a3b8"
    cloud_status_text = cloud_publisher.status_message if cloud_publisher else "Disabled"
    st.markdown(
        """
        <div style="font-size: 0.8rem; line-height: 1.8;">
            <div>📷 <b>Camera DMS:</b> <span style="color:#10b981;">Ready (Index {})</span></div>
            <div>❤️ <b>PPG/ECG Sensor:</b> <span style="color:#64748b;">Standby (Phase 2)</span></div>
            <div>🛰️ <b>GPS & Dynamics:</b> <span style="color:#10b981;">Online (Phase 3)</span></div>
            <div>☁️ <b>Cloud Sync:</b> <span style="color:{};">{}</span></div>
            <div>🔊 <b>Buzzer Actuator:</b> <span style="color:#38bdf8;">Standby (GPIO 23)</span></div>
            <div>📳 <b>Haptic Motor:</b> <span style="color:#38bdf8;">Standby (GPIO 13)</span></div>
            <div>💧 <b>Water Sprinkler:</b> <span style="color:#38bdf8;">Standby (GPIO 25)</span></div>
        </div>
        """.format(camera_idx_option, cloud_status_color, cloud_status_text),
        unsafe_allow_html=True
    )


# =============================================================================
# MAIN DASHBOARD INTERFACE
# =============================================================================
st.markdown("## 🛡️ Multimodal Driver Drowsiness Detection System")
st.markdown(
    "<div style='color: #94a3b8; font-size: 0.9rem; margin-top: -10px; margin-bottom: 16px;'>"
    "Real-time Driver Monitoring System (DMS) — Live Computer Vision + Multimodal Sensor Fusion Cockpit"
    "</div>",
    unsafe_allow_html=True
)

# Start / Stop Control Buttons
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 4])
with col_ctrl1:
    start_monitoring = st.button("▶️ Start Monitoring", key="btn_start", use_container_width=True, type="primary")
with col_ctrl2:
    stop_monitoring = st.button("⏹️ Stop Monitoring", key="btn_stop", use_container_width=True)
with col_ctrl3:
    st.markdown(
        "<div style='font-size:0.85rem; color:#94a3b8; padding-top:6px;'>"
        "Click <b>Start Monitoring</b> to initialize USB camera stream & real-time detection pipeline."
        "</div>",
        unsafe_allow_html=True
    )

# Maintain Session State for Stream Running
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False

if start_monitoring:
    st.session_state.monitoring_active = True

if stop_monitoring:
    st.session_state.monitoring_active = False

# Placeholder for Master Drowsiness Banner
banner_placeholder = st.empty()

# Layout Columns: Left (Video & Vision Cards), Right (Multimodal Telemetry Cards)
left_col, right_col = st.columns([5, 4])

with left_col:
    st.markdown('<div class="section-header">📷 Live Driver Vision Stream</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()

    # 4 Primary Live Vision Indicators
    v_col1, v_col2, v_col3, v_col4 = st.columns(4)
    with v_col1:
        face_status_ph = st.empty()
    with v_col2:
        eye_status_ph = st.empty()
    with v_col3:
        blink_rate_ph = st.empty()
    with v_col4:
        closure_dur_ph = st.empty()

with right_col:
    # 1. Computer Vision Detailed Metrics
    st.markdown('<div class="section-header">👁️ Ocular & Facial Metrics</div>', unsafe_allow_html=True)
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        ear_ph = st.empty()
    with m_col2:
        perclos_ph = st.empty()

    m_col3, m_col4 = st.columns(2)
    with m_col3:
        yawn_ph = st.empty()
    with m_col4:
        head_pose_ph = st.empty()


    # 3. Vehicle Dynamics & GPS Telemetry (Phase 3)
st.markdown(
    '<div class="section-header">🚘 Vehicle Dynamics & GPS Telemetry (Phase 3)</div>',
    unsafe_allow_html=True
)

gps_row1, gps_row2 = st.columns(2)
with gps_row1:
    gps_speed_ph = st.empty()
with gps_row2:
    gps_coords_ph = st.empty()

dyn_row1, dyn_row2 = st.columns(2)
with dyn_row1:
    movement_status_ph = st.empty()
with dyn_row2:
    speed_limit_ph = st.empty()


    # 4. Buzzer & Vibration Alerts (Actuators)
st.markdown('<div class="section-header">🚨 Alert & Actuator Subsystems</div>', unsafe_allow_html=True)
act_col1, act_col2, act_col3 = st.columns(3)
with act_col1:
    buzzer_ph = st.empty()
with act_col2:
    vibration_ph = st.empty()
with act_col3:
    sprinkler_ph = st.empty()

# Bottom Full-Width Section: Real-time Trend Charts
st.markdown("---")
st.markdown('<div class="section-header">📈 Real-Time Ocular Analytics (Rolling Trends)</div>', unsafe_allow_html=True)
chart_placeholder = st.empty()


# =============================================================================
# INITIAL STANDBY UI RENDERING (When Monitoring is Inactive)
# =============================================================================
def render_standby_ui():
    """Renders all cards and placeholders in initial STANDBY / DISCONNECTED states."""
    banner_placeholder.markdown(
        render_drowsiness_banner(
            status_text="SYSTEM STANDBY",
            severity="standby",
            sub_message="Camera offline. Press 'Start Monitoring' to begin live driver drowsiness evaluation."
        ),
        unsafe_allow_html=True
    )

    video_placeholder.markdown(
        """
        <div style="background: #0f172a; border-radius: 12px; border: 2px dashed #334155; padding: 75px 20px; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 12px;">📷</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">Live Camera Feed Offline</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">
                Select Camera Index in sidebar and click <b>▶️ Start Monitoring</b> to initiate MediaPipe tracking.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Vision Indicators
    face_status_ph.markdown(render_metric_card("Face Detected", "NO FACE", status_badge="OFFLINE", badge_type="standby"), unsafe_allow_html=True)
    eye_status_ph.markdown(render_metric_card("Eye Status", "STANDBY", status_badge="OFFLINE", badge_type="standby"), unsafe_allow_html=True)
    blink_rate_ph.markdown(render_metric_card("Blink Rate", "0.0", unit="/min", status_badge="STANDBY", badge_type="standby"), unsafe_allow_html=True)
    closure_dur_ph.markdown(render_metric_card("Closure Duration", "0.00", unit="s", status_badge="STANDBY", badge_type="standby"), unsafe_allow_html=True)

    # Detailed Vision Metrics
    ear_ph.markdown(render_metric_card("Average EAR", "--", status_badge="CALIBRATED", badge_type="standby", sub_text="L: -- | R: --"), unsafe_allow_html=True)
    perclos_ph.markdown(render_metric_card("PERCLOS (60s)", "0.0", unit="%", status_badge="NORMAL", badge_type="nominal", sub_text="Threshold: 15.0%"), unsafe_allow_html=True)
    yawn_ph.markdown(render_metric_card("Yawn Tracker", "0", unit="yawns", status_badge="NORMAL", badge_type="nominal", sub_text="Mouth MAR: --"), unsafe_allow_html=True)
    head_pose_ph.markdown(render_metric_card("Head Pose", "CENTER", status_badge="ALIGNED", badge_type="nominal", sub_text="P: 0° | Y: 0° | R: 0°"), unsafe_allow_html=True)

    # Multimodal Phase 2 (Strictly Unsimulated)

    # Multimodal Phase 3 (GPS & Vehicle Dynamics Standby)
    gps_speed_ph.markdown(
        render_metric_card("Vehicle Speed", "0.0", unit="km/h", status_badge="STAGNANT", badge_type="nominal", sub_text="Speed Limit: UNAVAILABLE"),
        unsafe_allow_html=True
    )
    gps_coords_ph.markdown(
        render_metric_card("GPS Position", "SEARCHING GPS", status_badge="0 Sats", badge_type="standby", sub_text="NEO-6M Module: Standby"),
        unsafe_allow_html=True
    )
    movement_status_ph.markdown(
        render_metric_card("Vehicle Movement", "STAGNANT (AT REST)", status_badge="NOMINAL", badge_type="nominal", sub_text="Vehicle at complete standstill"),
        unsafe_allow_html=True
    )
    speed_limit_ph.markdown(
        render_metric_card("Road Speed Limit", "SPEED LIMIT UNAVAILABLE", status_badge="STANDBY", badge_type="standby", sub_text="OSM Overpass Lookup"),
        unsafe_allow_html=True
    )

    # Actuators
    buzzer_ph.markdown(render_metric_card("Buzzer Alert", "STANDBY", status_badge="INACTIVE", badge_type="standby", sub_text="Acoustic Alarm"), unsafe_allow_html=True)
    vibration_ph.markdown(render_metric_card("Vibration Alert", "STANDBY", status_badge="INACTIVE", badge_type="standby", sub_text="Haptic Feedback"), unsafe_allow_html=True)
    sprinkler_ph.markdown(render_metric_card("Water Sprinkler", "STANDBY", status_badge="INACTIVE", badge_type="standby", sub_text="Pump Relay (GPIO 25)"), unsafe_allow_html=True)


render_standby_ui()


# =============================================================================
# REAL-TIME LIVE MONITORING LOOP
# =============================================================================
if st.session_state.monitoring_active:
    # 1. Initialize CameraManager
    camera = CameraManager(
        camera_index=camera_idx_option,
        width=frame_width,
        height=frame_height,
        target_fps=config.TARGET_FPS,
        backend=config.CAMERA_BACKEND
    )

    if not camera.start():
        st.error(
            f"❌ **Failed to open camera index {camera_idx_option}.**\n\n"
            f"• Please verify that your USB webcam is plugged in.\n"
            f"• Try changing the **Camera Index** (e.g. 0 or 1) in the sidebar.\n"
            f"• Ensure no other application (e.g. Zoom, Teams, or another terminal) is using the webcam."
        )
        st.session_state.monitoring_active = False
    else:
        # 2. Initialize Detection & Feature Processing Modules
        face_detector = FaceDetector(
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=config.REFINE_LANDMARKS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        eye_tracker = EyeTracker()
        blink_detector = BlinkDetector(
            ear_closed_threshold=ear_thresh,
            ear_open_threshold=ear_thresh + 0.05,
            prolonged_closure_sec=closure_time_thresh
        )
        yawn_detector = YawnDetector(
            mar_threshold=config.MAR_YAWN_THRESHOLD,
            min_yawn_duration_sec=config.MIN_YAWN_DURATION_SEC
        )
        head_pose_estimator = HeadPoseEstimator()
        perclos_tracker = PERCLOSTracker(window_seconds=config.PERCLOS_WINDOW_SECONDS)
        hud_visualizer = HUDVisualizer()
        feature_logger = FeatureLogger(enabled=enable_csv_logging)

        # Telemetry Rolling Buffers for Trend Analytics
        history_len = 80
        ear_history: collections.deque = collections.deque(maxlen=history_len)
        perclos_history: collections.deque = collections.deque(maxlen=history_len)
        time_history: collections.deque = collections.deque(maxlen=history_len)

        session_start = time.time()
        buzzer_active = False
        vibration_active = False
        sprinkler_active = False
        yawn_alert_start_time: Optional[float] = None
        last_yawn_alert_count = 0

        try:
            while st.session_state.monitoring_active:
                timestamp_now = time.time()
                elapsed_sec = timestamp_now - session_start
                

                # Read Video Frame from CameraManager
                ret, frame = camera.read_frame()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                h, w, _ = frame.shape
                fps = camera.get_fps()

                # Process Face Landmark Detection
                face_res = face_detector.process_frame(frame)

                # Initialize Metric Defaults
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

                    # 4. Yawn Detection
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
                    nose_landmark = face_res.landmarks_pixel[1]

                else:
                    # No Face Detected in Current Frame
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

                # -------------------------------------------------------------
                # ACTUATOR ALERT CONTROL: VIBRATION (GPIO 13) & BUZZER (GPIO 23)
                # -------------------------------------------------------------
                # Yawn Alert Timer:
                # Activates both buzzer and vibrator for 8.0s when yawn count reaches 5 (or increments >= 5)
                if yawn_res.yawn_count >= 5 and yawn_res.yawn_count > last_yawn_alert_count:
                    yawn_alert_start_time = timestamp_now
                    last_yawn_alert_count = yawn_res.yawn_count
                elif yawn_res.yawn_count < 5:
                    last_yawn_alert_count = 0
                    yawn_alert_start_time = None

                yawn_alert = (yawn_alert_start_time is not None and (timestamp_now - yawn_alert_start_time <= 8.0))

                # Combined Eye Closure Alert: continuous eye closure >= 3.0s (turns off immediately when eyes reopen)
                eye_closure_alert = (blink_res.eye_status == "CLOSED" and blink_res.eye_closed_duration >= 3.0)

                # 1. Vibration alert: >= 3.0s eye closure OR 8s yawn alert
                vibration_should_be_on = eye_closure_alert or yawn_alert

                # Send VIBRATION command on state change
                if vibration_should_be_on != vibration_active:
                    try:
                        if vehicle_serial is not None and vehicle_serial.is_open:
                            cmd_vib = "VIBRATION_ON\n" if vibration_should_be_on else "VIBRATION_OFF\n"
                            vehicle_serial.write(cmd_vib.encode("utf-8"))
                            vibration_active = vibration_should_be_on
                            print("Vibration command sent:", cmd_vib.strip())
                    except (serial.SerialException, OSError) as e:
                        print("Vibration control error:", e)

                # 2. Buzzer alert: >= 3.0s eye closure OR 8s yawn alert
                buzzer_should_be_on = eye_closure_alert or yawn_alert

                # Send BUZZER command on state change
                if buzzer_should_be_on != buzzer_active:
                    try:
                        if vehicle_serial is not None and vehicle_serial.is_open:
                            command = "BUZZER_ON\n" if buzzer_should_be_on else "BUZZER_OFF\n"
                            vehicle_serial.write(command.encode("utf-8"))
                            buzzer_active = buzzer_should_be_on
                            print("Buzzer command sent:", command.strip())
                    except (serial.SerialException, OSError) as e:
                        print("Buzzer control error:", e)

                # 3. Sprinkler / Water Spray alert: >= 3.0s eye closure OR 8s yawn alert
                sprinkler_should_be_on = eye_closure_alert or yawn_alert

                # Send SPRINKLER command on state change
                if sprinkler_should_be_on != sprinkler_active:
                    try:
                        if vehicle_serial is not None and vehicle_serial.is_open:
                            command_spk = "SPRINKLER_ON\n" if sprinkler_should_be_on else "SPRINKLER_OFF\n"
                            vehicle_serial.write(command_spk.encode("utf-8"))
                            sprinkler_active = sprinkler_should_be_on
                            print("Sprinkler command sent:", command_spk.strip())
                    except (serial.SerialException, OSError) as e:
                        print("Sprinkler control error:", e)

                # Ingest live MPU6050 & NEO-6M GPS packets non-blockingly
                read_vehicle_sensor()

                # Master Drowsiness State Logic
                if not face_res.detected:
                    drowsiness_state = "NO DRIVER DETECTED"
                    drowsiness_severity = "warning"
                    drowsiness_sub = "Face mesh lost. Driver not visible to optical sensor."
                    sys_status_hud = "SEARCHING FACE"
                elif blink_res.prolonged_closure:
                    drowsiness_state = "CRITICAL: PROLONGED EYE CLOSURE"
                    drowsiness_severity = "alert"
                    drowsiness_sub = f"Eyes continuously closed for {blink_res.eye_closed_duration:.1f}s! Immediate alarm active."
                    sys_status_hud = "WARNING: CLOSURE"
                elif False and current_perclos >= perclos_alert_thresh:
                    drowsiness_state = "WARNING: HIGH PERCLOS DROWSINESS"
                    drowsiness_severity = "alert"
                    drowsiness_sub = f"PERCLOS ({current_perclos:.1f}%) exceeds safety threshold ({perclos_alert_thresh:.1f}%)."
                    sys_status_hud = "WARNING: PERCLOS"
                elif yawn_alert:
                    remaining_s = max(0.0, 8.0 - (timestamp_now - (yawn_alert_start_time or timestamp_now)))
                    drowsiness_state = "CRITICAL: YAWN FATIGUE ALERT (>=5 YAWNS)"
                    drowsiness_severity = "alert"
                    drowsiness_sub = f"Cumulative yawns reached {yawn_res.yawn_count}! Alarm active ({remaining_s:.1f}s remaining)."
                    sys_status_hud = "YAWN ALARM"
                elif yawn_res.is_yawning:
                    drowsiness_state = "WARNING: YAWNING DETECTED"
                    drowsiness_severity = "warning" 
                    drowsiness_sub = f"Significant mouth expansion detected (MAR: {mar_val:.2f})."
                    sys_status_hud = "WARNING: YAWN"
                elif pitch < -25.0:
                    drowsiness_state = "ATTENTION: DRIVER DISTRACTED"
                    drowsiness_severity = "warning"
                    drowsiness_sub = f"Head orientation deviated: {pose_res.orientation_label}."
                    sys_status_hud = "DISTRACTED"
                else:
                    drowsiness_state = "NOMINAL — DRIVER ALERT"
                    drowsiness_severity = "nominal"
                    drowsiness_sub = "Facial telemetry within normal physiological thresholds."
                    sys_status_hud = "MONITORING"

                # Compile Complete Telemetry Map
                metrics: Dict[str, Any] = {
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
                    "left_contour_pts": left_contour if show_mesh else None,
                    "right_contour_pts": right_contour if show_mesh else None,
                    "mouth_outer_pts": mouth_outer if show_mesh else None,
                    "mouth_inner_pts": mouth_inner if show_mesh else None,
                    "nose_landmark_pt": nose_landmark if show_pose else None,
                    "nose_projected_pt": nose_proj if show_pose else None
                }

                # Optional CSV Logging
                if enable_csv_logging:
                    feature_logger.log(metrics)

                # Non-blocking Asynchronous Cloud Synchronization (Firebase RTDB)
                if cloud_publisher is not None:
                    cloud_publisher.update_telemetry(
                        driver_state=drowsiness_state,
                        driver_severity=drowsiness_severity,
                        driver_sub=drowsiness_sub,
                        eye_status=blink_res.eye_status,
                        eye_closed_duration=blink_res.eye_closed_duration,
                        ear=avg_ear,
                        perclos=current_perclos,
                        yawn_count=yawn_res.yawn_count,
                        is_yawning=yawn_res.is_yawning,
                        head_orientation=pose_res.orientation_label,
                        vehicle_telemetry=vehicle_telemetry,
                        buzzer_active=buzzer_active,
                        vibration_active=vibration_active,
                        water_spray_active=sprinkler_active
                    )

                # Render Video Frame with Annotations
                if show_hud:
                    annotated_frame = hud_visualizer.draw_hud(
                        frame=frame,
                        metrics=metrics,
                        system_status=sys_status_hud
                    )
                else:
                    annotated_frame = frame.copy()
                    if face_res.detected and show_mesh:
                        hud_visualizer._draw_facial_contours(annotated_frame, metrics)
                    if face_res.detected and show_pose:
                        hud_visualizer._draw_head_pose_vector(annotated_frame, metrics)

                # Convert OpenCV BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Update Master Drowsiness Banner
                banner_placeholder.markdown(
                    render_drowsiness_banner(
                        status_text=drowsiness_state,
                        severity=drowsiness_severity,
                        sub_message=f"{drowsiness_sub} | FPS: {fps:.1f} | Session Time: {int(elapsed_sec)}s"
                    ),
                    unsafe_allow_html=True
                )

                # Update Live Vision Indicator Cards
                face_badge = "DETECTED" if face_res.detected else "NO FACE"
                face_type = "nominal" if face_res.detected else "alert"
                face_status_ph.markdown(
                    render_metric_card(
                        "Face Detected",
                        face_badge,
                        status_badge=f"{face_res.num_faces} Face(s)" if face_res.detected else "LOST",
                        badge_type=face_type,
                        sub_text="MediaPipe 468-Mesh"
                    ),
                    unsafe_allow_html=True
                )

                eye_col = "nominal" if blink_res.eye_status == "OPEN" else ("alert" if blink_res.eye_status == "CLOSED" else "standby")
                eye_status_ph.markdown(
                    render_metric_card(
                        "Eye Status",
                        blink_res.eye_status,
                        status_badge="ACTIVE" if face_res.detected else "STANDBY",
                        badge_type=eye_col,
                        sub_text=f"Threshold: {ear_thresh:.2f}"
                    ),
                    unsafe_allow_html=True
                )

                blink_rate_ph.markdown(
                    render_metric_card(
                        "Blink Rate",
                        f"{blink_res.blink_rate:.1f}",
                        unit="/min",
                        status_badge=f"{blink_res.blink_count} Tot",
                        badge_type="cyan",
                        sub_text="Normal: 12-20 /min"
                    ),
                    unsafe_allow_html=True
                )

                closure_badge_type = "alert" if blink_res.prolonged_closure else ("warning" if blink_res.eye_closed_duration > 0.5 else "nominal")
                closure_dur_ph.markdown(
                    render_metric_card(
                        "Eye Closure Dur",
                        f"{blink_res.eye_closed_duration:.2f}",
                        unit="s",
                        status_badge="PROLONGED" if blink_res.prolonged_closure else "NORMAL",
                        badge_type=closure_badge_type,
                        sub_text=f"Max: {blink_res.max_closed_duration:.2f}s"
                    ),
                    unsafe_allow_html=True
                )

                # Update Detailed Vision Metrics
                ear_badge_type = "nominal" if avg_ear >= ear_thresh else "alert"
                ear_ph.markdown(
                    render_metric_card(
                        "Average EAR",
                        f"{avg_ear:.3f}",
                        status_badge="NORMAL" if avg_ear >= ear_thresh else "CLOSED",
                        badge_type=ear_badge_type,
                        sub_text=f"L: {left_ear:.2f} | R: {right_ear:.2f}"
                    ),
                    unsafe_allow_html=True
                )

                perclos_badge_type = "alert" if current_perclos >= perclos_alert_thresh else ("warning" if current_perclos >= 8.0 else "nominal")
                perclos_ph.markdown(
                    render_metric_card(
                        "PERCLOS (60s)",
                        f"{current_perclos:.1f}",
                        unit="%",
                        status_badge="CRITICAL" if current_perclos >= perclos_alert_thresh else "NOMINAL",
                        badge_type=perclos_badge_type,
                        sub_text=f"NHTSA Alert: >{perclos_alert_thresh:.0f}%"
                    ),
                    unsafe_allow_html=True
                )

                yawn_badge_type = "alert" if yawn_res.is_yawning else "nominal"
                yawn_ph.markdown(
                    render_metric_card(
                        "Yawn Tracker",
                        f"{yawn_res.yawn_count}",
                        unit="yawns",
                        status_badge=yawn_res.mouth_status,
                        badge_type=yawn_badge_type,
                        sub_text=f"MAR: {mar_val:.2f} (Limit: {config.MAR_YAWN_THRESHOLD})"
                    ),
                    unsafe_allow_html=True
                )

                head_pose_ph.markdown(
                    render_metric_card(
                        "Head Orientation",
                        pose_res.orientation_label,
                        status_badge="ALIGNED" if pose_res.orientation_label == "FORWARD" else "DEVIATED",
                        badge_type="nominal" if pose_res.orientation_label == "FORWARD" else "warning",
                        sub_text=f"P: {pitch:+.0f}° | Y: {yaw:+.0f}° | R: {roll:+.0f}°"
                    ),
                    unsafe_allow_html=True
                )

                # Phase 3: Live GPS & Motion Telemetry Cards
                curr_speed = vehicle_telemetry["speed_kmh"]
                spd_lim = vehicle_telemetry["speed_limit"]
                is_stagnant = (curr_speed < 0.8)

                if is_stagnant:
                    spd_display = "0.0"
                    spd_badge = "STAGNANT"
                    spd_type = "nominal"
                    spd_sub = f"Road Limit: {spd_lim} km/h" if spd_lim else "Speed Limit: UNAVAILABLE"
                else:
                    spd_display = f"{curr_speed:.1f}"
                    if spd_lim is not None:
                        if curr_speed > spd_lim:
                            spd_badge = f"OVERSPEEDING (+{curr_speed - spd_lim:.0f})"
                            spd_type = "alert"
                        else:
                            spd_badge = "WITHIN LIMIT"
                            spd_type = "nominal"
                        spd_sub = f"Road Limit: {spd_lim} km/h"
                    else:
                        spd_badge = "CRUISING"
                        spd_type = "nominal"
                        spd_sub = "Speed Limit: UNAVAILABLE"

                # Vehicle Speed is displayed at all times (0.0 km/h when stagnant)
                gps_speed_ph.markdown(
                    render_metric_card(
                        "Vehicle Speed",
                        spd_display,
                        unit="km/h",
                        status_badge=spd_badge,
                        badge_type=spd_type,
                        sub_text=spd_sub
                    ),
                    unsafe_allow_html=True
                )

                if vehicle_telemetry["fix_valid"]:
                    lat_val = vehicle_telemetry["latitude"]
                    lng_val = vehicle_telemetry["longitude"]
                    sats_cnt = vehicle_telemetry["satellites"]
                    road_txt = vehicle_telemetry["current_road"] or "Road Identified"
                    gps_coords_ph.markdown(
                        render_metric_card(
                            "GPS Position",
                            f"{lat_val:.5f}, {lng_val:.5f}",
                            status_badge=f"{sats_cnt} Sats (3D Fix)",
                            badge_type="nominal",
                            sub_text=f"Road: {road_txt}"
                        ),
                        unsafe_allow_html=True
                    )
                else:
                    sats_cnt = vehicle_telemetry["satellites"]
                    gps_coords_ph.markdown(
                        render_metric_card(
                            "GPS Position",
                            "SEARCHING GPS",
                            status_badge=f"{sats_cnt} Sats",
                            badge_type="warning",
                            sub_text="NEO-6M Lock: Awaiting Sky Line-of-Sight"
                        ),
                        unsafe_allow_html=True
                    )

                # Phase 3: Vehicle Dynamics Movement State
                movement_status_ph.markdown(
                    render_metric_card(
                        "Vehicle Movement",
                        vehicle_telemetry["movement_state"],
                        status_badge=vehicle_telemetry["movement_badge"].upper(),
                        badge_type=vehicle_telemetry["movement_badge"],
                        sub_text=vehicle_telemetry["movement_detail"]
                    ),
                    unsafe_allow_html=True
                )

                # Phase 3: Road Speed Limit Card
                if vehicle_telemetry["speed_limit"] is not None:
                    speed_limit_ph.markdown(
                        render_metric_card(
                            "Road Speed Limit",
                            f"{vehicle_telemetry['speed_limit']}",
                            unit="km/h",
                            status_badge="OSM VERIFIED",
                            badge_type="nominal",
                            sub_text=f"Road: {vehicle_telemetry['current_road'] or 'Matched'}"
                        ),
                        unsafe_allow_html=True
                    )
                else:
                    speed_limit_ph.markdown(
                        render_metric_card(
                            "Road Speed Limit",
                            "SPEED LIMIT UNAVAILABLE",
                            status_badge="NO DATA",
                            badge_type="standby",
                            sub_text="OpenStreetMap Overpass Lookup"
                        ),
                        unsafe_allow_html=True
                    )

                buzzer_state = "TRIGGERED" if buzzer_active else "STANDBY"
                buzzer_badge = "alert" if buzzer_state == "TRIGGERED" else "standby"
                buzzer_ph.markdown(
                    render_metric_card("Buzzer Alert", buzzer_state, status_badge="ACTIVE" if buzzer_state == "TRIGGERED" else "STANDBY", badge_type=buzzer_badge, sub_text="Acoustic 85dB Warning"),
                    unsafe_allow_html=True
                )

                vib_state = "TRIGGERED" if vibration_active else "STANDBY"
                vib_badge = "alert" if vibration_active else "standby"
                vibration_ph.markdown(
                    render_metric_card("Vibration Alert", vib_state, status_badge="ACTIVE" if vibration_active else "STANDBY", badge_type=vib_badge, sub_text="Haptic Feedback (GPIO 13)"),
                    unsafe_allow_html=True
                )

                spk_state = "TRIGGERED" if sprinkler_active else "STANDBY"
                spk_badge = "alert" if sprinkler_active else "standby"
                sprinkler_ph.markdown(
                    render_metric_card("Water Sprinkler", spk_state, status_badge="ACTIVE" if sprinkler_active else "STANDBY", badge_type=spk_badge, sub_text="Pump Relay (GPIO 25)"),
                    unsafe_allow_html=True
                )

                # Render Live Trend Chart
                if len(ear_history) > 2:
                    df_trend = pd.DataFrame({
                        "Eye Aspect Ratio (EAR)": list(ear_history),
                        "PERCLOS %": [p / 100.0 for p in perclos_history]  # Normalize for common scale
                    })
                    chart_placeholder.line_chart(df_trend, height=180)

                # Allow Streamlit to yield
                time.sleep(0.01)

        except Exception as e:
            st.error(f"Error during video monitoring: {e}")
        finally:
            # Clean Resource Teardown
            if buzzer_active and vehicle_serial is not None and vehicle_serial.is_open:
                try:
                    vehicle_serial.write(b"BUZZER_OFF\n")
                except Exception:
                    pass
            if vibration_active and vehicle_serial is not None and vehicle_serial.is_open:
                try:
                    vehicle_serial.write(b"VIBRATION_OFF\n")
                except Exception:
                    pass
            if sprinkler_active and vehicle_serial is not None and vehicle_serial.is_open:
                try:
                    vehicle_serial.write(b"SPRINKLER_OFF\n")
                except Exception:
                    pass
            camera.release()
            face_detector.close()
            if enable_csv_logging:
                feature_logger.close()
            st.session_state.monitoring_active = False
