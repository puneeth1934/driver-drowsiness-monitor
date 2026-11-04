# Multimodal Machine Learning & Sensor Fusion Architecture (Phase 2 Roadmap)

This directory will host the trained AI/ML classification models and multimodal feature fusion pipelines for the **AI-Based Multimodal Driver Drowsiness Detection System**.

---

## 1. System Overview & Future Sensor Fusion

In the final system, the driver monitoring pipeline will combine physiological, behavioral, and vehicular telemetry into a unified multimodal feature vector.

```
+-------------------------------------------------------------------------------+
|                             MULTIMODAL SENSOR INPUTS                          |
+-----------------------+-------------------------------+-----------------------+
|  VISION (Webcam)      |  PHYSIOLOGY (MAX30102)        |  VEHICLE DYNAMICS     |
|  - Left/Right EAR     |  - Heart Rate (BPM)           |  - Steering Angle     |
|  - PERCLOS (60s)      |  - RMSSD (HRV)                |    (AS5600 Magnetic)  |
|  - Blink Rate (/min)  |  - SDNN (HRV)                 |  - Steering Angular   |
|  - Closure Duration   |  - LF/HF Ratio                |    Velocity (MPU6050) |
|  - Yawn Count & MAR   |                               |  - Micro-corrections  |
|  - Head Pitch/Yaw/Roll|                               |    frequency          |
+-----------------------+-------------------------------+-----------------------+
                                    |
                                    v
            +-----------------------------------------------+
            |    Feature Fusion & Time-Series Alignment     |
            |     (Synchronized by UTC / Epoch Millis)      |
            +-----------------------------------------------+
                                    |
                                    v
            +-----------------------------------------------+
            |      AI / ML Binary Classifier (LightGBM /    |
            |            Random Forest / LSTM)              |
            +-----------------------------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
             [ AWAKE (Class 0) ]            [ DROWSY (Class 1) ]
                                                    |
                                                    v
                                    +-------------------------------+
                                    |    ESP32 Actuator Feedback    |
                                    |    - Active Buzzer (PWM)      |
                                    |    - Vibration Motor Alert    |
                                    |    - I2C OLED Status Display  |
                                    +-------------------------------+
```

---

## 2. Multimodal Data Schema Contract

When the ESP32 serial bridge is implemented in Phase 2, telemetry streams will merge using timestamp matching into the following unified schema:

| Feature Group | Column Name | Type | Description |
|---|---|---|---|
| **Temporal** | `timestamp` | string | ISO 8601 UTC timestamp (`YYYY-MM-DD HH:MM:SS.mmm`) |
| | `elapsed_time` | float | Elapsed seconds from session start |
| **Vision (Ocular)** | `left_ear` | float | Left Eye Aspect Ratio |
| | `right_ear` | float | Right Eye Aspect Ratio |
| | `average_ear` | float | Average EAR |
| | `perclos` | float | 60-second rolling percentage of eye closure |
| | `blink_count` | int | Total cumulative blinks |
| | `blink_rate` | float | Blinks per minute |
| | `eye_closed_duration` | float | Current continuous closure duration (s) |
| **Vision (Oral/Head)**| `mouth_ratio` | float | Mouth Aspect Ratio (MAR) |
| | `yawn_count` | int | Total confirmed yawns |
| | `head_pitch` | float | Nodding angle in degrees (solvePnP) |
| | `head_yaw` | float | Looking left/right in degrees |
| | `head_roll` | float | Head tilt in degrees |
| **Physiological** | `heart_rate_bpm` | float | Pulse rate from MAX30102 PPG sensor |
| | `hrv_rmssd` | float | Root Mean Square of Successive RR differences (ms) |
| | `hrv_sdnn` | float | Standard deviation of NN intervals (ms) |
| **Steering / IMU** | `steering_angle_deg` | float | Absolute steering wheel angle from AS5600 (0-360°) |
| | `steering_velocity` | float | Angular velocity from MPU6050 gyro (°/s) |
| | `steering_reversals` | int | Count of micro-steering adjustments per minute |
| **Ground Truth** | `drowsiness_label` | int | 0 = Alert / Awake, 1 = Drowsy |

---

## 3. Machine Learning Model Planning

1. **Baseline Model**: LightGBM / XGBoost classifier for fast, low-latency edge inference (~1-2 ms per prediction).
2. **Sequential Model**: 1D-CNN or Bidirectional LSTM evaluated over 10-second sliding windows to capture temporal degradation patterns before micro-sleep episodes occur.
3. **Hardware Deployment**: Export trained weights to ONNX or TensorFlow Lite for Microcontrollers (TFLite-Micro) or host locally in a dedicated Python inference thread.
