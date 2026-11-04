
import serial
import time
import threading

PORT = "COM5"
BAUD_RATE = 115200

ser = None
sensor_thread = None
latest_data = {
    "status": "NO DATA",
    "acceleration": 0.0,
    "rotation": 0.0,
    "movement": 0.0,
    "ax": 0.0,
    "ay": 0.0,
    "az": 0.0,
    "gx": 0.0,
    "gy": 0.0,
    "gz": 0.0,
}


def connect_sensor():
    global ser

    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"ESP32 connected on {PORT}")
        return True

    except serial.SerialException as e:
        print("ESP32 connection error:", e)
        ser = None
        latest_data["status"] = "DISCONNECTED"
        return False


def read_sensor():
    global latest_data

    if ser is None:
        return

    while True:
        try:
            line = ser.readline().decode(
                "utf-8", errors="ignore"
            ).strip()

            if not line:
                continue

            print("ESP32:", line)
            parts = line.split(",")

            if len(parts) >= 7 and parts[0] == "MPU":
                try:
                    ax = float(parts[1])
                    ay = float(parts[2])
                    az = float(parts[3])
                    gx = float(parts[4])
                    gy = float(parts[5])
                    gz = float(parts[6])

                    latest_data.update({
                        "status": "CONNECTED",
                        "ax": ax,
                        "ay": ay,
                        "az": az,
                        "gx": gx,
                        "gy": gy,
                        "gz": gz,
                        "acceleration": (
                            ax**2 + ay**2 + az**2
                        ) ** 0.5,
                        "rotation": (
                            gx**2 + gy**2 + gz**2
                        ) ** 0.5,
                        "movement": abs(
                            (ax**2 + ay**2 + az**2) ** 0.5 - 1.0
                        ),
                    })

                except ValueError:
                    print("Invalid MPU data:", line)

        except (serial.SerialException, OSError) as e:
            print("Sensor reading error:", e)
            latest_data["status"] = "ERROR"
            break


def start_sensor():
    global sensor_thread

    if sensor_thread is not None and sensor_thread.is_alive():
        return

    if connect_sensor():
        sensor_thread = threading.Thread(
            target=read_sensor,
            daemon=True
        )
        sensor_thread.start()


def get_vehicle_data():
    return latest_data.copy()
