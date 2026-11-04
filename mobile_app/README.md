# Driver Drowsiness System — Mobile Telemetry App

A mobile-friendly, responsive live monitoring web client for the Driver Drowsiness Detection System.

Reads real-time vehicle and driver telemetry directly from Firebase Realtime Database. Does **not** require USB webcams, serial COM ports, or ESP32 hardware to run.

---

## Features
- **Responsive Mobile Cockpit**: Custom viewport, touch-friendly dark UI, and PWA manifest.
- **Master Alert Banner**: Live color-coded status (`NORMAL`, `DROWSY`, `DISTRACTED`, `NO DRIVER`).
- **Actuator Alert Status**: Real-time indicators for **Buzzer** (GPIO 23), **Vibration Motor** (GPIO 13), and **Water Sprinkler / Pump** (GPIO 25).
- **Vehicle Dynamics & Speed**: Live speed in km/h, road speed limits, and overspeeding alerts.
- **GPS Map Tracking**: Pinpoint vehicle location on an interactive map.
- **Ocular Telemetry**: Eye closure duration, EAR, PERCLOS, and yawn count.
- **MPU6050 Diagnostics**: Linear acceleration ($g$) and angular turn rates (°/s).

---

## Option 1: Run Locally & View on Mobile Phone (Same Wi-Fi)

1. Make sure your laptop and your smartphone are connected to the same Wi-Fi router or mobile hotspot.
2. Start the mobile app on your laptop bound to all network interfaces (`0.0.0.0`):
   ```powershell
   .\venv\Scripts\streamlit.exe run mobile_app/app.py --server.address 0.0.0.0 --server.port 8502
   ```
3. Check your laptop's local IP address (e.g., run `ipconfig`). Suppose your laptop IP is `192.168.1.45`.
4. Open your phone's browser (Chrome on Android or Safari on iPhone) and navigate to:
   ```text
   http://192.168.1.45:8502
   ```

---

## Option 2: Deploy to Streamlit Community Cloud (Free Online Access Anywhere)

To access your vehicle from anywhere in the world over cellular data:

1. Push your repository to GitHub (credentials in `.env` and `ServiceAccountKey.json` are automatically protected by `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select your repository, set the **Main file path** to:
   ```text
   mobile_app/app.py
   ```
4. In **Advanced settings** $\rightarrow$ **Secrets**, add:
   ```toml
   FIREBASE_DATABASE_URL = "https://driver-drowsiness-monito-244f3-default-rtdb.asia-southeast1.firebasedatabase.app"
   ```
5. Click **Deploy**. You will get a permanent public URL (e.g. `https://your-monitor.streamlit.app`).

---

## How to Add to Mobile Home Screen (PWA)

- **Android (Chrome)**: Tap the three dots (⋮) in the top-right corner $\rightarrow$ tap **Add to Home screen** (or **Install app**).
- **iOS (Safari)**: Tap the **Share** button (box with an upward arrow) $\rightarrow$ scroll down and tap **Add to Home Screen**.

