"""
=============================================================================
Driver Drowsiness Detection System — Mobile Telemetry Client
=============================================================================
A responsive, lightweight mobile web application for monitoring driver
drowsiness, vehicle dynamics, GPS location, and actuator alert states in
real time via Firebase Realtime Database.

Designed for smartphones (Android & iOS) and tablet viewports.
=============================================================================
"""

import os
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment configuration
BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(PARENT_DIR / ".env")

# =============================================================================
# PAGE & VIEWPORT CONFIGURATION (Mobile Optimized)
# =============================================================================
st.set_page_config(
    page_title="Driver Drowsiness Fleet Monitor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Mobile-responsive CSS and Theme styling
MOBILE_CSS = """
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="theme-color" content="#0b0e14">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<link rel="manifest" href="static/manifest.json">

<style>
/* Base Dark Mobile Cockpit Theme */
.stApp {
    background-color: #0b0e14;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

/* Remove desktop padding for full mobile view */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    padding-left: 0.8rem;
    padding-right: 0.8rem;
    max-width: 540px;
}

/* Mobile Card Styling */
.mobile-card {
    background: #141a26;
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
}

.mobile-card-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 4px;
}

.mobile-card-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1.2;
}

.mobile-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-nominal { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
.badge-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
.badge-alert   { background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.5); }
.badge-standby { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.3); }

/* Actuator Pills */
.actuator-pill {
    padding: 8px 10px;
    border-radius: 10px;
    text-align: center;
    font-weight: 700;
    font-size: 0.8rem;
    transition: all 0.3s ease;
}
.actuator-off {
    background: #1a2233;
    border: 1px solid #334155;
    color: #64748b;
}
.actuator-on {
    background: rgba(239, 68, 68, 0.25);
    border: 1px solid #ef4444;
    color: #fca5a5;
    animation: pulse 1s infinite alternate;
}
@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 4px rgba(239, 68, 68, 0.4); }
    100% { transform: scale(1.02); box-shadow: 0 0 14px rgba(239, 68, 68, 0.8); }
}

/* Master Alert Banner */
.status-banner {
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
}
.status-nominal {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.85), rgba(15, 23, 42, 0.95));
    border: 1px solid #059669;
}
.status-warning {
    background: linear-gradient(135deg, rgba(120, 53, 15, 0.85), rgba(15, 23, 42, 0.95));
    border: 1px solid #d97706;
}
.status-alert {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.90), rgba(15, 23, 42, 0.95));
    border: 2px solid #ef4444;
    animation: alertGlow 1.5s infinite alternate;
}
@keyframes alertGlow {
    from { box-shadow: 0 0 8px rgba(239, 68, 68, 0.5); }
    to { box-shadow: 0 0 20px rgba(239, 68, 68, 0.9); }
}
.status-standby {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
    border: 1px solid #475569;
}
</style>
"""
st.markdown(MOBILE_CSS, unsafe_allow_html=True)


# =============================================================================
# FIREBASE REST CLIENT (Pure Client, No Secrets Required on Mobile)
# =============================================================================
def get_firebase_url() -> str:
    """Retrieves Firebase Realtime Database URL from secrets, env, or sidebar."""
    # 1. Check Streamlit Secrets (for Streamlit Cloud deployment)
    if hasattr(st, "secrets") and "FIREBASE_DATABASE_URL" in st.secrets:
        return st.secrets["FIREBASE_DATABASE_URL"].rstrip("/")
    # 2. Check local environment
    env_url = os.getenv("FIREBASE_DATABASE_URL", "").strip().rstrip("/")
    if env_url:
        return env_url
    # 3. Default fallback
    return "https://driver-drowsiness-monito-244f3-default-rtdb.asia-southeast1.firebasedatabase.app"


def fetch_vehicle_telemetry(db_url: str, device_id: str) -> Optional[Dict[str, Any]]:
    """Fetches the latest vehicle telemetry record via Firebase REST API."""
    if not db_url or not device_id:
        return None
    try:
        url = f"{db_url}/vehicles/{device_id.strip()}.json"
        resp = requests.get(url, timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# =============================================================================
# SIDEBAR / SETTINGS (Mobile Friendly)
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Monitor Settings")
    default_dev = os.getenv("DEVICE_ID", "DDS-VEM-001")
    target_device = st.text_input("Vehicle Device ID", value=default_dev, help="Enter the unique ID of the vehicle to monitor").strip()
    
    custom_db_url = st.text_input(
        "Firebase RTDB URL",
        value=get_firebase_url(),
        help="Firebase Realtime Database endpoint"
    ).rstrip("/")

    auto_refresh = st.checkbox("Auto-Refresh (Live Feed)", value=True)
    refresh_rate = st.slider("Refresh Interval (s)", 1, 5, 2)
    
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #94a3b8; line-height: 1.6;">
            <b>📲 Add to Home Screen:</b><br>
            • In Chrome for Android: tap <b>⋮</b> &gt; <b>Add to Home screen</b>.<br>
            • In Safari for iOS: tap <b>Share</b> &gt; <b>Add to Home Screen</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

# Use target_device
device_id = target_device or "DDS-VEM-001"
db_url = custom_db_url or get_firebase_url()


# =============================================================================
# LIVE REFRESH FRAGMENT
# =============================================================================
@st.fragment(run_every=f"{refresh_rate}s" if auto_refresh else None)
def render_live_telemetry():
    """Renders the live mobile interface with auto-refreshing fragment."""
    data = fetch_vehicle_telemetry(db_url, device_id)
    now = time.time()

    # 1. Header Bar: Device ID & Connectivity
    is_online = False
    last_seen_str = "Offline"
    if data and "meta" in data:
        t_delta = now - float(data.get("meta", {}).get("timestamp", 0.0))
        is_online = t_delta <= 12.0  # Online if telemetry received within last 12s
        last_seen_str = f"{int(t_delta)}s ago" if t_delta < 60 else "Inactive"

    conn_badge = "badge-nominal" if is_online else "badge-standby"
    conn_text = "🟢 LIVE" if is_online else "⚪ OFFLINE"

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <span style="font-size: 1.15rem; font-weight: 800; color: #f8fafc;">🚗 {device_id}</span>
            </div>
            <div>
                <span class="mobile-badge {conn_badge}">{conn_text} ({last_seen_str})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not data:
        st.markdown(
            f"""
            <div class="status-banner status-standby">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🛰️</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Waiting for Vehicle Signal</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    Ensure the laptop Streamlit dashboard is running and connected to Firebase for Device <b>{device_id}</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Extract Subsystems
    driver = data.get("driver_status", {})
    vehicle = data.get("vehicle_telemetry", {})
    actuators = data.get("actuators", {})
    gps = vehicle.get("gps", {})
    mpu = vehicle.get("mpu", {})

    status_str = driver.get("status", "NORMAL").upper()
    raw_state = driver.get("raw_state", "NOMINAL — DRIVER ALERT")
    desc = driver.get("description", "Telemetry within nominal limits")

    # 2. Master Driver Status Banner
    if status_str == "DROWSY" or "PROLONGED" in raw_state or "YAWN FATIGUE" in raw_state:
        b_class = "status-alert"
        b_icon = "🚨"
        b_title = "CRITICAL: DROWSINESS ALERT"
    elif status_str == "DISTRACTED" or "DISTRACTED" in raw_state:
        b_class = "status-warning"
        b_icon = "⚠️"
        b_title = "ATTENTION: DRIVER DISTRACTED"
    elif status_str == "NO_DRIVER":
        b_class = "status-standby"
        b_icon = "👤"
        b_title = "NO DRIVER DETECTED"
    else:
        b_class = "status-nominal"
        b_icon = "🛡️"
        b_title = "DRIVER ALERT & ATTENTIVE"

    st.markdown(
        f"""
        <div class="status-banner {b_class}">
            <div style="font-size: 1.8rem; margin-bottom: 2px;">{b_icon}</div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #f8fafc; letter-spacing: 0.02em;">{b_title}</div>
            <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 4px;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. Actuator Alerts Row (Buzzer, Vibration, Sprinkler)
    buzzer_on = bool(actuators.get("buzzer", False))
    vib_on = bool(actuators.get("vibration", False))
    spray_on = bool(actuators.get("water_spray", False))

    act_col1, act_col2, act_col3 = st.columns(3)
    with act_col1:
        b_style = "actuator-on" if buzzer_on else "actuator-off"
        b_lbl = "ACTIVE" if buzzer_on else "INACTIVE"
        st.markdown(f'<div class="actuator-pill {b_style}">🔊 Buzzer<br><span style="font-size:0.68rem;">{b_lbl}</span></div>', unsafe_allow_html=True)
    with act_col2:
        v_style = "actuator-on" if vib_on else "actuator-off"
        v_lbl = "ACTIVE" if vib_on else "INACTIVE"
        st.markdown(f'<div class="actuator-pill {v_style}">📳 Vibration<br><span style="font-size:0.68rem;">{v_lbl}</span></div>', unsafe_allow_html=True)
    with act_col3:
        s_style = "actuator-on" if spray_on else "actuator-off"
        s_lbl = "ACTIVE" if spray_on else "INACTIVE"
        st.markdown(f'<div class="actuator-pill {s_style}">💧 Sprinkler<br><span style="font-size:0.68rem;">{s_lbl}</span></div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 4. Vehicle Speed & Movement Card
    speed_kmh = float(vehicle.get("speed_kmh", 0.0))
    speed_lim = vehicle.get("speed_limit")
    overspeeding = bool(vehicle.get("overspeeding", False))
    mv_state = vehicle.get("movement_state", "STAGNANT (AT REST)")
    mv_badge = vehicle.get("movement_badge", "nominal")

    sp_color = "#ef4444" if overspeeding else "#f8fafc"
    limit_txt = f"Road Limit: {speed_lim} km/h" if speed_lim else "Limit: Unavailable"

    st.markdown(
        f"""
        <div class="mobile-card">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div class="mobile-card-title">Vehicle Velocity</div>
                <div class="mobile-badge {'badge-alert' if overspeeding else 'badge-nominal'}">
                    {'OVERSPEEDING' if overspeeding else 'WITHIN LIMIT'}
                </div>
            </div>
            <div style="display: flex; align-items: baseline; gap: 6px; margin-top: 2px;">
                <span class="mobile-card-value" style="color: {sp_color}; font-size: 2.1rem;">{speed_kmh:.1f}</span>
                <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 700;">km/h</span>
                <span style="font-size: 0.75rem; color: #64748b; margin-left: auto;">{limit_txt}</span>
            </div>
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #1e293b; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.75rem; color: #94a3b8;">Movement:</span>
                <span class="mobile-badge badge-{mv_badge}">{mv_state}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 5. Ocular & Fatigue Metrics Grid
    c1, c2 = st.columns(2)
    with c1:
        eye_status = driver.get("eye_status", "OPEN")
        c_dur = float(driver.get("eye_closed_duration", 0.0))
        e_badge = "badge-alert" if eye_status == "CLOSED" else "badge-nominal"
        st.markdown(
            f"""
            <div class="mobile-card">
                <div class="mobile-card-title">Eyes & Closure</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.2rem; font-weight: 800;">{c_dur:.1f}s</span>
                    <span class="mobile-badge {e_badge}">{eye_status}</span>
                </div>
                <div style="font-size: 0.7rem; color: #64748b; margin-top: 4px;">EAR: {driver.get('ear', 0.0):.3f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        yawns = driver.get("yawn_count", 0)
        perclos = float(driver.get("perclos", 0.0))
        y_badge = "badge-alert" if yawns >= 5 else ("badge-warning" if yawns >= 3 else "badge-nominal")
        st.markdown(
            f"""
            <div class="mobile-card">
                <div class="mobile-card-title">Yawns & PERCLOS</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.2rem; font-weight: 800;">{yawns} yawns</span>
                    <span class="mobile-badge {y_badge}">{perclos:.0f}%</span>
                </div>
                <div style="font-size: 0.7rem; color: #64748b; margin-top: 4px;">Head: {driver.get('head_orientation', 'FORWARD')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 6. GPS Location & Live Map
    lat = float(gps.get("latitude", 0.0))
    lng = float(gps.get("longitude", 0.0))
    sats = int(gps.get("satellites", 0))
    fix_valid = bool(gps.get("fix_valid", False)) and abs(lat) > 0.0001 and abs(lng) > 0.0001
    road_name = gps.get("current_road") or "Identified Roadway"

    st.markdown(
        f"""
        <div class="mobile-card" style="margin-bottom: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div class="mobile-card-title">GPS Positioning</div>
                <div class="mobile-badge {'badge-nominal' if fix_valid else 'badge-warning'}">
                    {f'{sats} Sats (3D Fix)' if fix_valid else f'{sats} Sats (Searching)'}
                </div>
            </div>
            <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc; margin-top: 2px;">
                {f'{lat:.5f}, {lng:.5f}' if fix_valid else 'Awaiting Sky Line-of-Sight'}
            </div>
            <div style="font-size: 0.7rem; color: #64748b; margin-top: 2px;">Road: {road_name}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if fix_valid:
        df_map = pd.DataFrame([{"lat": lat, "lon": lng}])
        st.map(df_map, zoom=14, use_container_width=True)

    # 7. MPU6050 Accelerometer & Gyroscope Telemetry
    with st.expander("📊 MPU6050 Motion Sensors", expanded=False):
        ax = mpu.get("ax", 0.0)
        ay = mpu.get("ay", 0.0)
        az = mpu.get("az", 1.0)
        gx = mpu.get("gx", 0.0)
        gy = mpu.get("gy", 0.0)
        gz = mpu.get("gz", 0.0)
        detail = vehicle.get("movement_detail", "Normal")

        st.markdown(
            f"""
            <div style="font-size: 0.78rem; line-height: 1.7; color: #94a3b8;">
                <div><b>Linear Accel (g):</b> X: <code>{ax:+.2f}</code> | Y: <code>{ay:+.2f}</code> | Z: <code>{az:+.2f}</code></div>
                <div><b>Gyroscope (°/s):</b> X: <code>{gx:+.1f}</code> | Y: <code>{gy:+.1f}</code> | Z: <code>{gz:+.1f}</code></div>
                <div style="margin-top: 4px; color: #38bdf8;"><b>Motion Diagnostic:</b> {detail}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# Render Fragment
render_live_telemetry()

