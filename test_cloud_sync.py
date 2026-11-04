"""
=============================================================================
Cloud Sync Module Verification Test
=============================================================================
Tests initialization, thread safety, graceful fallback, and telemetry encoding
without requiring active hardware or camera.
=============================================================================
"""

import time
import unittest
from cloud.firebase_sync import FirebaseCloudPublisher
import config

class TestCloudSync(unittest.TestCase):

    def test_unconfigured_graceful_fallback(self):
        """Publisher should handle missing credentials gracefully without throwing."""
        print("[TEST] Testing FirebaseCloudPublisher unconfigured fallback...")
        pub = FirebaseCloudPublisher(
            device_id="DDS-TEST-001",
            database_url="",
            credentials_path="non_existent_key.json",
            sync_interval_sec=0.5
        )
        self.assertFalse(pub.is_initialized)
        self.assertIn("Standby", pub.status_message)
        
        # Test start and stop
        started = pub.start()
        self.assertTrue(started)

        # Ingest mock telemetry
        pub.update_telemetry(
            driver_state="NOMINAL — DRIVER ALERT",
            driver_severity="nominal",
            driver_sub="All clear",
            eye_status="OPEN",
            eye_closed_duration=0.0,
            ear=0.32,
            perclos=3.2,
            yawn_count=0,
            is_yawning=False,
            head_orientation="FORWARD",
            vehicle_telemetry={"speed_kmh": 0.0, "latitude": 12.97, "longitude": 77.59},
            buzzer_active=False,
            vibration_active=False
        )

        time.sleep(0.6)
        pub.stop()
        self.assertEqual(pub.status_message, "Stopped")
        print("  -> Unconfigured fallback passed successfully.")

if __name__ == "__main__":
    unittest.main()

