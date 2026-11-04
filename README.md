# AI-Based Multimodal Driver Drowsiness Detection System
## Phase 1: Real-Time Webcam Eye-Tracking & Head Pose System

---

## 1. Project Objective

Driver drowsiness and micro-sleep episodes are among the leading causes of fatal highway and transport collisions worldwide. This final-year engineering project develops an **AI-Based Multimodal Driver Drowsiness Detection System** combining:

1. **Webcam-based Computer Vision** (Eye Aspect Ratio, PERCLOS, Blink Rate, Yawning MAR, and 3D Head Pose)
2. **Physiological Monitoring** (Heart Rate & Heart Rate Variability via MAX30102 PPG sensor)
3. **Vehicle Steering Dynamics** (Steering angle from AS5600 magnetic encoder and angular velocity from MPU6050 6-DOF IMU)
4. **Multimodal Machine Learning Classifier** (Awake vs. Drowsy classification)
5. **Driver Alert System** (ESP32 micro-controller controlling an active buzzer, vibration motor, and OLED screen)

> [!IMPORTANT]
> **Phase 1 Focus**: This module builds and thoroughly tests the **Vision-Based Driver Monitoring System** using a standard USB webcam. It computes objective, scientific numerical features without premature heuristic labeling, establishing a reliable ground-truth telemetry stream for later fusion with the ESP32 hardware sensors.

---

## 2. Hardware Requirements (Phase 1)

- **Computer**: Windows 10 / 11 (or Linux / macOS) PC with USB 2.0/3.0 port
- **Camera**: Standard USB Webcam (720p / 1080p, 30 FPS recommended) or integrated laptop camera
- **Lighting**: Normal ambient indoor or vehicle cabin lighting

---

## 3. Technology Stack & Python Version

- **Python Version**: Python 3.10 or 3.11 (64-bit recommended)
- **Computer Vision**: OpenCV (`opencv-python`), MediaPipe FaceMesh (`mediapipe`)
- **Scientific Computing**: NumPy, SciPy
- **Data Engineering**: Pandas
- **Visualization**: Matplotlib (for offline analysis)

---

## 4. Installation & Virtual Environment Setup (Windows)

Open **PowerShell** or **Command Prompt** in Windows and navigate to the project directory:

```powershell
# 1. Navigate to the project directory
cd C:\Users\Puneeth\.gemini\antigravity\scratch\driver_drowsiness

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1
# (If using CMD: venv\Scripts\activate.bat)

# 4. Upgrade pip and install all required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> If PowerShell shows an `Execution_Policies` script restriction error when activating the virtual environment, run:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run `.\venv\Scripts\Activate.ps1` again.

---

## 5. How to Run the System

### A. Run Real-Time Monitoring
```powershell
python main.py
```

### B. Run with Specific Camera Index
```powershell
python main.py --camera 1
```

### C. Run with Initial Guided Baseline Calibration
```powershell
python main.py --calibrate
```

### D. Run Automated Algorithm Tests (No camera required)
```powershell
python test_system.py
```

### E. Generate Offline Analysis Plots from Recorded CSV
```powershell
python analyze_data.py
```

---

## 6. Real-Time Controls & Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` or `ESC` | Cleanly shuts down camera, closes windows, and flushes CSV log |
| `C` | Starts the interactive 4-step user baseline calibration routine |
| `R` | Resets cumulative blink count, yawn count, and rolling PERCLOS buffer |

---

## 7. Project Architecture & Directory Layout

```
driver_drowsiness/
│
├── main.py                  # Real-time application entry point & acquisition loop
├── config.py                # Centralized threshold & camera configuration
├── requirements.txt         # Pinned project dependencies
├── README.md                # Complete documentation & test suite guide
├── analyze_data.py          # Offline multi-panel time-series visualization
├── test_system.py           # Automated unit & mathematical integration tests
│
├── camera/                  # Camera acquisition & FPS tracking
│   ├── __init__.py
│   └── camera_manager.py
│
├── detection/               # Landmark extraction & temporal state machines
│   ├── __init__.py
│   ├── face_detector.py     # MediaPipe FaceMesh wrapper & multi-face handler
│   ├── eye_tracker.py       # Eye landmark extraction & EAR calculation
│   ├── blink_detector.py    # Physiological blink FSM & closure tracker
│   ├── yawn_detector.py     # Mouth Aspect Ratio (MAR) & sustained yawn FSM
│   └── head_pose.py         # solvePnP 3D Euler angles (Pitch, Yaw, Roll)
│
├── features/                # Pure mathematical feature algorithms
│   ├── __init__.py
│   ├── ear.py               # Euclidean EAR & MAR formulas
│   ├── perclos.py           # Rolling-window timestamp-weighted PERCLOS
│   └── feature_logger.py    # Rate-limited CSV logger (5-10 Hz)
│
├── utils/                   # Display & calibration utilities
│   ├── __init__.py
│   ├── drawing.py           # Automotive HUD dashboard & contour overlays
│   └── calibration.py       # Interactive baseline calibration helper
│
├── data/
│   ├── eye_tracking_data.csv# Real-time logged telemetry data
│   ├── calibration.json     # Saved user baseline calibration profile
│   └── plots/               # Saved high-resolution figures from analyze_data.py
│
└── models/
    └── README.md            # Roadmap for multimodal AI sensor fusion with ESP32
```

---

## 8. Mathematical Formulations & Feature Descriptions

### 1. Eye Aspect Ratio (EAR)
Defined by Soukupová and Čech (2016), EAR measures eye opening using 6 distinct eyelid landmarks:

$$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

- $p_1, p_4$ are the inner/outer eye corners.
- $(p_2, p_6)$ and $(p_3, p_5)$ are vertical pairs on the upper and lower eyelids.
- **Interpretation**: Open eye $\approx 0.28 - 0.38$; Closed eye $< 0.21$.

### 2. Physiological Blink Detection (FSM)
A robust 4-stage Finite State Machine avoids counting jitter or sustained closure as blinks:
$$\text{OPEN} \longrightarrow \text{CLOSING} \longrightarrow \text{CLOSED} \longrightarrow \text{OPEN}$$
- A blink is only registered if closure duration is within valid human physiological blink times ($60\text{ ms} \le t_{\text{blink}} \le 450\text{ ms}$).
- Eyelid closure exceeding $450\text{ ms}$ transitions directly into **Eye Closure Duration** tracking.

### 3. Prolonged Eye Closure
Tracks the continuous duration for which eyes remain below the closure threshold. When $t_{\text{closed}} \ge 1.5\text{ s}$, the system flags **PROLONGED EYE CLOSURE**.

### 4. PERCLOS (Percentage of Eye Closure)
Standardized by NHTSA and ISO 15007 as a key ocular fatigue metric:

$$\text{PERCLOS} = \frac{\sum \Delta t_{\text{closed, valid}}}{\sum \Delta t_{\text{valid}}} \times 100\%$$

- Evaluated over a rolling 60-second observation window.
- Excludes missing/occluded face intervals to prevent measurement distortion.

### 5. Mouth Aspect Ratio (MAR) & Yawning
Mouth opening is quantified using 6 inner/outer lip landmarks:

$$\text{MAR} = \frac{||m_2 - m_6|| + ||m_3 - m_5||}{2 \cdot ||m_1 - m_4||}$$

- Normal resting mouth: $\text{MAR} \approx 0.15 - 0.35$.
- Yawning: $\text{MAR} \ge 0.60$ sustained continuously for $\ge 1.8\text{ seconds}$. This temporal constraint filters out talking, singing, and brief smiles.

### 6. 3D Head Pose Estimation (Perspective-n-Point)
Uses OpenCV `cv2.solvePnP` mapping 2D facial landmarks (nose tip, chin, eye corners, mouth corners) to a 3D anthropometric facial model:
- **Pitch**: Nodding up/down (head droop detection)
- **Yaw**: Looking left/right (distraction detection)
- **Roll**: Lateral head tilt

---

## 9. CSV Telemetry Schema (`data/eye_tracking_data.csv`)

| Column Name | Type | Example | Description |
|---|---|---|---|
| `timestamp` | string | `2026-08-14 09:15:30.120` | Timestamp with millisecond precision |
| `elapsed_time` | float | `12.450` | Seconds elapsed since session start |
| `face_detected` | int | `1` | 1 = Face Present, 0 = Face Not Found |
| `left_ear` | float | `0.3012` | Left Eye Aspect Ratio |
| `right_ear` | float | `0.2985` | Right Eye Aspect Ratio |
| `average_ear` | float | `0.2998` | Mean Eye Aspect Ratio |
| `eye_status` | string | `OPEN` | `OPEN` or `CLOSED` |
| `eye_closed_duration`| float | `0.000` | Current ongoing closure duration (seconds) |
| `blink_event` | int | `0` | Instantaneous 1-frame impulse upon blink completion |
| `blink_count` | int | `7` | Cumulative total blinks |
| `blink_rate` | float | `14.20` | Rolling blinks per minute |
| `perclos` | float | `3.40` | Rolling 60s eye closure percentage |
| `mouth_ratio` | float | `0.2240` | Mouth Aspect Ratio (MAR) |
| `mouth_status` | string | `NORMAL` | `NORMAL` or `OPEN` |
| `yawn_event` | int | `0` | Instantaneous 1-frame impulse upon confirmed yawn |
| `yawn_count` | int | `1` | Cumulative confirmed yawns |
| `head_pitch` | float | `-2.40` | Head Pitch in degrees |
| `head_yaw` | float | `1.10` | Head Yaw in degrees |
| `head_roll` | float | `0.30` | Head Roll in degrees |
| `fps` | float | `29.8` | Real-time acquisition framerate |

---

## 10. Comprehensive 10-Point Verification Test Plan

| Test ID | Test Scenario | Procedure | Expected System Response |
|---|---|---|---|
| **TEST 1** | **Face Detection** | Position face in front of webcam. | Face status shows `DETECTED`, green eye contours and mouth mesh are drawn. |
| **TEST 2** | **Eyes Open** | Look straight at camera with eyes naturally open. | Left/Right EAR display values between `0.28` and `0.36`. Eye Status = `OPEN`. |
| **TEST 3** | **Normal Blinking** | Blink naturally at regular intervals. | `Blink Count` increments by 1 per blink. `Blink Rate` shows ~10–20/min. |
| **TEST 4** | **Eyes Closed** | Close eyes for 0.5 seconds. | Eye Status immediately switches to `CLOSED`. EAR drops below `0.21`. |
| **TEST 5** | **Prolonged Eye Closure** | Close eyes continuously for > 1.5 seconds. | Red banner **"PROLONGED EYE CLOSURE"** appears at the bottom. |
| **TEST 6** | **Yawning** | Open mouth wide and hold for 2 seconds. | `Mouth Status` switches to `OPEN`, MAR rises > 0.60, and `Yawn Count` increments by 1. |
| **TEST 7** | **Head Movement** | Turn head left/right, nod up/down. | 3D nose vector rotates dynamically; Pitch, Yaw, Roll angles track movements. |
| **TEST 8** | **Face Leaving Camera** | Move face out of camera frame. | Top red banner displays **"FACE NOT DETECTED"**. Eye calculations freeze safely without crashing. |
| **TEST 9** | **Poor Lighting / Occlusion** | Dim ambient light or cover half of face. | System gracefully handles landmark dropouts; recovers immediately when visibility returns. |
| **TEST 10** | **CSV Logging & Offline Plots** | Run for 30s, press `Q`, then run `python analyze_data.py`. | Valid CSV file created in `data/eye_tracking_data.csv` and multi-panel plot saved to `data/plots/drowsiness_features_analysis.png`. |

---

## 11. Troubleshooting & FAQs

- **Camera fails to open (`Could not open camera 0`)**:
  - Check if another application (Zoom, Teams, Camera App) is using the webcam.
  - In `config.py`, change `CAMERA_INDEX = 1` or `CAMERA_INDEX = 2` if using an external USB camera.
- **Low FPS (< 20 FPS)**:
  - In `config.py`, verify `FRAME_WIDTH = 640` and `FRAME_HEIGHT = 480`. Lowering resolution maintains full landmark accuracy while significantly boosting framerate on standard CPUs.
- **MediaPipe import error**:
  - Run `pip install --upgrade mediapipe opencv-python numpy`.

---

## 12. Next Steps: Roadmap to Multimodal ESP32 Integration (Phase 2)

```
       +-------------------------+
       |   Phase 1: Eye Tracker  |  (Completed)
       +------------+------------+
                    |
                    v
       +-------------------------+
       | Phase 2: ESP32 Sensors  |  (MAX30102 + MPU6050 + AS5600 Serial Ingestion)
       +------------+------------+
                    |
                    v
       +-------------------------+
       | Phase 3: AI Classifier  |  (Multimodal Feature Model: Awake vs Drowsy)
       +------------+------------+
                    |
                    v
       +-------------------------+
       | Phase 4: Alert Actuator |  (ESP32 Buzzer + Vibration Motor + OLED)
       +-------------------------+
```
