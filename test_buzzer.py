import serial
import time

try:
    esp = serial.Serial("COM5", 115200, timeout=1)
    time.sleep(2)

    print("Sending BUZZER_ON")
    esp.write(b"BUZZER_ON\n")
    esp.flush()

    start = time.time()
    time.sleep(3)
    print(f"Waited {time.time() - start:.1f} seconds")

    print("Sending BUZZER_OFF")
    esp.write(b"BUZZER_OFF\n")
    esp.flush()

    esp.close()
    print("Test complete")

except Exception as e:
    print("Error:", e)