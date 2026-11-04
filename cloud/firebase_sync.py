"""
=============================================================================
Firebase Realtime Database Cloud Synchronization Module
=============================================================================
Provides asynchronous, non-blocking telemetry synchronization between the
local Driver Drowsiness Detection System and Firebase Realtime Database.

Key Features:
- Runs in a dedicated background daemon thread (never degrades DMS frame rate).
- Pushes continuous telemetry at configurable intervals (default: 1.0s heartbeat).
- Triggers immediate push on critical events (drowsiness alerts, impacts, overspeeding).
- Handles internet loss and reconnects gracefully with exponential backoff.
- Thread-safe payload caching.
- Safe standalone fallback if Firebase credentials are unconfigured.
=============================================================================
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logger
logger = logging.getLogger("FirebaseSync")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [FirebaseSync] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Optional Firebase import guard
try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logger.warning("firebase-admin package not found. Cloud synchronization disabled.")


class FirebaseCloudPublisher:
    """
    Manages non-blocking cloud telemetry publishing to Firebase Realtime Database.
    """

    def __init__(
        self,
        device_id: str = "DDS-VEM-001",
        database_url: str = "",
        credentials_path: str = "serviceAccountKey.json",
        sync_interval_sec: float = 1.0
    ):
        self.device_id = device_id.strip() if device_id else "DDS-VEM-001"
        self.database_url = database_url.strip() if database_url else ""
        self.credentials_path = credentials_path
        self.sync_interval = max(0.5, float(sync_interval_sec))

        self._initialized = False
        self._connected = False
        self._status_message = "Initializing..."
        self._last_sync_time = 0.0

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._immediate_event = threading.Event()

        # Thread-safe telemetry snapshot
        self._latest_payload: Optional[Dict[str, Any]] = None
        self._last_pushed_state: Optional[str] = None

        # Resolve credentials path relative to project root
        self._resolved_cred_path = self._resolve_credentials_path(self.credentials_path)

        # Initialize Firebase App
        self._init_firebase()

    def _resolve_credentials_path(self, path_str: str) -> Optional[Path]:
        """Resolves the credentials path against absolute path or project root."""
        if not path_str:
            return None
        p = Path(path_str)
        if p.is_file():
            return p.resolve()
        
        # Check relative to base directory (one level up from cloud/)
        base_dir = Path(__file__).resolve().parent.parent
        candidate = base_dir / path_str
        if candidate.is_file():
            return candidate.resolve()
        return None

    def _init_firebase(self) -> None:
        """Initializes the Firebase Admin SDK app safely."""
        if not FIREBASE_AVAILABLE:
            self._status_message = "Disabled (firebase-admin not installed)"
            return

        if not self.database_url:
            self._status_message = "Standby (FIREBASE_DATABASE_URL not configured)"
            logger.info("Firebase Realtime Database URL not set in .env. Running in offline mode.")
            return

        if not self._resolved_cred_path or not self._resolved_cred_path.exists():
            self._status_message = f"Standby (Missing {self.credentials_path})"
            logger.info(f"Firebase credentials JSON not found at '{self.credentials_path}'. Running in offline mode.")
            return

        try:
            # Check if default app is already initialized (e.g. across Streamlit reruns)
            app_initialized = False
            try:
                firebase_admin.get_app()
                app_initialized = True
            except ValueError:
                app_initialized = False

            if not app_initialized:
                cred = credentials.Certificate(str(self._resolved_cred_path))
                firebase_admin.initialize_app(cred, {
                    "databaseURL": self.database_url
                })
                logger.info(f"Firebase Admin SDK initialized successfully for device '{self.device_id}'.")

            self._initialized = True
            self._connected = True
            self._status_message = "Connected to Firebase"

            # Set device initial online status
            self._set_device_status(online=True)

        except Exception as e:
            self._initialized = False
            self._connected = False
            self._status_message = f"Init Error: {type(e).__name__}"
            logger.error(f"Failed to initialize Firebase Admin SDK: {e}")

    def _set_device_status(self, online: bool) -> None:
        """Updates device connectivity status in Firebase."""
        if not self._initialized:
            return
        try:
            ref = db.reference(f"vehicles/{self.device_id}/device_info")
            ref.update({
                "device_id": self.device_id,
                "online": online,
                "last_heartbeat": time.time(),
                "last_updated": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.debug(f"Failed to update device online status: {e}")

    def start(self) -> bool:
        """Starts the background publishing daemon thread."""
        if self._running:
            return True

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._publish_worker_loop,
            daemon=True,
            name=f"FirebaseSyncWorker-{self.device_id}"
        )
        self._worker_thread.start()
        logger.info(f"Firebase cloud sync background worker started (interval: {self.sync_interval}s).")
        return True

    def stop(self) -> None:
        """Stops the publishing thread and marks the device as offline."""
        if not self._running:
            return

        self._running = False
        self._immediate_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)

        self._set_device_status(online=False)
        self._status_message = "Stopped"
        logger.info("Firebase cloud sync worker stopped.")

    def update_telemetry(
        self,
        driver_state: str,
        driver_severity: str,
        driver_sub: str,
        eye_status: str,
        eye_closed_duration: float,
        ear: float,
        perclos: float,
        yawn_count: int,
        is_yawning: bool,
        head_orientation: str,
        vehicle_telemetry: Dict[str, Any],
        buzzer_active: bool,
        vibration_active: bool,
        water_spray_active: bool = False
    ) -> None:
        """
        Thread-safe ingestion of the latest detection metrics and sensor packet.
        Called once per camera frame from dashboard/app.py without blocking.
        """
        now = time.time()
        iso_str = datetime.now(timezone.utc).isoformat()

        # Simplify driver classification into standard categories: NORMAL, DROWSY, DISTRACTED, NO_DRIVER
        state_upper = driver_state.upper()
        if "PROLONGED" in state_upper or "YAWN FATIGUE" in state_upper or "DROWSY" in state_upper:
            normalized_status = "DROWSY"
        elif "DISTRACTED" in state_upper:
            normalized_status = "DISTRACTED"
        elif "NO DRIVER" in state_upper:
            normalized_status = "NO_DRIVER"
        elif "WARNING: YAWNING" in state_upper:
            normalized_status = "DROWSY"
        else:
            normalized_status = "NORMAL"

        payload = {
            "device_info": {
                "device_id": self.device_id,
                "online": True,
                "last_heartbeat": now
            },
            "driver_status": {
                "status": normalized_status,
                "raw_state": driver_state,
                "severity": driver_severity,
                "description": driver_sub,
                "eye_status": eye_status,
                "eye_closed_duration": round(float(eye_closed_duration), 2),
                "ear": round(float(ear), 3),
                "perclos": round(float(perclos), 1),
                "yawn_count": int(yawn_count),
                "is_yawning": bool(is_yawning),
                "head_orientation": str(head_orientation)
            },
            "vehicle_telemetry": {
                "speed_kmh": round(float(vehicle_telemetry.get("speed_kmh", 0.0)), 1),
                "speed_limit": vehicle_telemetry.get("speed_limit"),
                "overspeeding": bool(vehicle_telemetry.get("overspeeding", False)),
                "movement_state": str(vehicle_telemetry.get("movement_state", "STAGNANT (AT REST)")),
                "movement_badge": str(vehicle_telemetry.get("movement_badge", "nominal")),
                "movement_detail": str(vehicle_telemetry.get("movement_detail", "")),
                "gps": {
                    "latitude": round(float(vehicle_telemetry.get("latitude", 0.0)), 6),
                    "longitude": round(float(vehicle_telemetry.get("longitude", 0.0)), 6),
                    "satellites": int(vehicle_telemetry.get("satellites", 0)),
                    "fix_valid": bool(vehicle_telemetry.get("fix_valid", False)),
                    "current_road": vehicle_telemetry.get("current_road")
                },
                "mpu": {
                    "ax": round(float(vehicle_telemetry.get("filtered_ax", 0.0)), 3),
                    "ay": round(float(vehicle_telemetry.get("filtered_ay", 0.0)), 3),
                    "az": round(float(vehicle_telemetry.get("filtered_az", 1.0)), 3),
                    "gx": round(float(vehicle_telemetry.get("filtered_gx", 0.0)), 2),
                    "gy": round(float(vehicle_telemetry.get("filtered_gy", 0.0)), 2),
                    "gz": round(float(vehicle_telemetry.get("filtered_gz", 0.0)), 2)
                }
            },
            "actuators": {
                "buzzer": bool(buzzer_active),
                "vibration": bool(vibration_active),
                "water_spray": bool(water_spray_active)
            },
            "meta": {
                "timestamp": now,
                "last_updated": iso_str
            }
        }

        with self._lock:
            self._latest_payload = payload

        # Immediate push trigger on critical transitions (e.g. DROWSY or buzzer turning on)
        is_critical = (
            normalized_status == "DROWSY"
            or buzzer_active
            or vibration_active
            or water_spray_active
            or vehicle_telemetry.get("overspeeding", False)
            or "IMPACT" in vehicle_telemetry.get("movement_state", "")
        )
        if is_critical and self._last_pushed_state != normalized_status:
            self._immediate_event.set()

    def _publish_worker_loop(self) -> None:
        """Background worker thread loop that pushes telemetry to Firebase RTDB."""
        backoff_sec = 1.0
        while self._running:
            # Wait for sync_interval or an immediate event
            triggered = self._immediate_event.wait(timeout=self.sync_interval)
            if triggered:
                self._immediate_event.clear()

            if not self._running:
                break

            if not self._initialized:
                # Retry initialization periodically (e.g. if user adds credentials while app is running)
                if self.database_url and self._resolved_cred_path and self._resolved_cred_path.exists():
                    self._init_firebase()
                time.sleep(2.0)
                continue

            # Fetch snapshot to upload
            with self._lock:
                snapshot = self._latest_payload

            if snapshot is None:
                continue

            try:
                # Update entire vehicle record in Firebase Realtime Database
                ref = db.reference(f"vehicles/{self.device_id}")
                ref.update(snapshot)

                self._connected = True
                self._status_message = "Sync Active (Online)"
                self._last_sync_time = time.time()
                self._last_pushed_state = snapshot["driver_status"]["status"]
                backoff_sec = 1.0  # Reset backoff on success

            except Exception as e:
                self._connected = False
                self._status_message = f"Sync Warning: {type(e).__name__}"
                logger.debug(f"Firebase publish error (backing off {backoff_sec:.1f}s): {e}")
                # Safe exponential backoff on network drop (max 10s)
                time.sleep(backoff_sec)
                backoff_sec = min(10.0, backoff_sec * 1.5)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def last_sync_time(self) -> float:
        return self._last_sync_time

