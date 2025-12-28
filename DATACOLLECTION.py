import serial
import csv
import time
from datetime import datetime

# ==============================
# USER SETTINGS
# ==============================
PORT = "COM5"      # 🔹 Change to your ESP32 COM port
BAUD_RATE = 115200
CSV_FILENAME = "sensor_data.csv"
# ==============================

def main():
    try:
        # Open serial connection
        ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
        print(f"✅ Connected to {PORT} at {BAUD_RATE} baud.")
        time.sleep(2)  # wait for ESP32 to boot

        # Check if CSV file already exists
        file_exists = False
        try:
            with open(CSV_FILENAME, "r"):
                file_exists = True
        except FileNotFoundError:
            pass

        with open(CSV_FILENAME, "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write header if file is new
            if not file_exists:
                header = ["date", "time", "ax", "ay", "az", "temp", "hum", "gas", "heart", "ir", "flame", "status"]
                csvwriter.writerow(header)

            print(f"📁 Writing data to '{CSV_FILENAME}' ... Press Ctrl+C to stop.\n")

            while True:
                try:
                    line = ser.readline().decode('utf-8').strip()

                    # Skip empty or invalid lines
                    if not line or "," not in line:
                        continue

                    # Skip ESP32 header line
                    if line.startswith("ax") or line.startswith("ay"):
                        continue

                    parts = line.split(",")

                    # Expecting 10 values (ax..status)
                    if len(parts) == 10:
                        now = datetime.now()
                        date_str = now.strftime("%Y-%m-%d")
                        time_str = now.strftime("%H:%M:%S")

                        row = [date_str, time_str] + parts
                        csvwriter.writerow(row)
                        csvfile.flush()

                        print("b ", row)

                except UnicodeDecodeError:
                    continue  # skip bad serial characters

    except serial.SerialException:
        print(f" Could not open port {PORT}. Check COM number or cable connection.")
    except KeyboardInterrupt:
        print("\n Data collection stopped by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        print(f" Data saved to '{CSV_FILENAME}' successfully.")

if __name__ == "__main__":
    main()
