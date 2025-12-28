# ===============================================================
# SMART SAFETY ANALYZER (v7)
# Custom time range (HH:MM AM/PM) + Email + Alerts + GUI Enhancements
# ===============================================================

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd, numpy as np, joblib, tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt, os, smtplib
from email.mime.text import MIMEText
from datetime import datetime, time
import winsound

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ------------------ CONFIG ------------------
DATA_FILE = "sensor_data.csv"
SVM_MODEL = "svm_safe_unsafe_model.pkl"
ANN_MODEL = "ann_safe_unsafe.keras"
CNN_MODEL = "cnn_safe_unsafe.keras"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER = "label_encoder.pkl"

SENDER_EMAIL = "goat76305@gmail.com"
SENDER_PASSWORD = "zeyyhvmbkcgonhuy"  # Google App password
RECEIVER_EMAIL = "sanjaisaravanan8870@gmail.com"


# ===============================================================
# EMAIL ALERT (Formatted)
# ===============================================================
def send_email_alert(unsafe_rows):
    try:
        if not unsafe_rows:
            return
        body_lines = [
            "⚠️ UNSAFE CONDITION ALERT ⚠️",
            "Detected unsafe conditions during selected time range:\n",
        ]
        for idx, timestamp, reasons, urgency in unsafe_rows:
            body_lines.append(f"➡️ Row {idx} | {urgency} | Time: {timestamp}")
            for r in reasons:
                body_lines.append(f"   - {r}")
            body_lines.append("")
        body_lines.append("✅ Please take corrective action immediately.\n\nSmart Safety Analyzer v7")

        body = "\n".join(body_lines)
        msg = MIMEText(body)
        msg["Subject"] = "⚠️ UNSAFE Condition Detected"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print("✅ Email alert sent successfully!")
    except Exception as e:
        print("❌ Email send failed:", e)


# ===============================================================
# DATA + MODEL
# ===============================================================
def load_csv():
    try:
        df = pd.read_csv(DATA_FILE)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        return df
    except FileNotFoundError:
        messagebox.showerror("Error", f"Dataset '{DATA_FILE}' not found!")
        return None


def preprocess_data(df, model_type):
    df = df.copy()
    if model_type == "SVM":
        cols = ["ax", "ay", "az", "temp", "hum", "gas", "heart", "ir", "flame"]
    else:
        df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
        df["gas_by_temp"] = df["gas"] / (df["temp"].replace(0, 1))
        df["heart_by_gas"] = df["heart"] / (df["gas"].replace(0, 1))
        for c in ["ax", "ay", "az", "temp", "hum", "gas", "heart", "acc_mag"]:
            df[f"{c}_z"] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)
        cols = [
            "ax","ay","az","temp","hum","gas","heart","ir","flame",
            "acc_mag","gas_by_temp","heart_by_gas",
            "ax_z","ay_z","az_z","temp_z","hum_z","gas_z","heart_z","acc_mag_z"
        ]
    return df[[c for c in cols if c in df.columns]]


def load_model(algo):
    try:
        scaler = joblib.load(SCALER_FILE)
        encoder = joblib.load(LABEL_ENCODER)
        if algo == "SVM":
            model = joblib.load(SVM_MODEL)
            model_type = "SVM"
        elif algo == "ANN":
            model = tf.keras.models.load_model(ANN_MODEL)
            model_type = "ANN"
        else:
            model = tf.keras.models.load_model(CNN_MODEL)
            model_type = "CNN"
        return model, scaler, encoder, model_type
    except Exception as e:
        messagebox.showerror("Error", f"Model load failed: {e}")
        return None, None, None, None


# ===============================================================
# ALERT + SOUND
# ===============================================================
def play_alert_sound():
    winsound.Beep(1200, 400)
    winsound.Beep(800, 400)


def show_alert_banner(status):
    if status == "UNSAFE":
        alert_label.config(bg="#b91c1c", text="🚨 UNSAFE DETECTED! 🚨")
    else:
        alert_label.config(bg="#16a34a", text="✅ SYSTEM SAFE ✅")


# ===============================================================
# COLORFUL UNSAFE FRAMES
# ===============================================================
def display_unsafe_frames(unsafe_rows):
    for widget in unsafe_frame.winfo_children():
        widget.destroy()
    if not unsafe_rows:
        return
    for (idx, timestamp, reasons, urgency) in unsafe_rows:
        color = {
            "Critical": "#ef4444",
            "Warning": "#f97316",
            "Moderate": "#facc15"
        }.get(urgency, "#9ca3af")
        frame = tk.Frame(unsafe_frame, bg=color, bd=3, relief="ridge")
        frame.pack(fill="x", padx=12, pady=5)
        tk.Label(frame, text=f"Row {idx} | {urgency} | {timestamp}",
                 bg=color, fg="white", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(5, 0))
        for r in reasons:
            tk.Label(frame, text=f"• {r}", bg=color, fg="white", font=("Consolas", 10)).pack(anchor="w", padx=25)


# ===============================================================
# TIME RANGE PREDICTION
# ===============================================================
def predict_by_timerange():
    algo = algo_var.get()
    df = load_csv()
    if df is None or "time" not in df.columns:
        messagebox.showerror("Error", "Dataset missing 'time' column.")
        return

    try:
        start_time = datetime.strptime(start_time_entry.get(), "%I:%M %p").time()
        end_time = datetime.strptime(end_time_entry.get(), "%I:%M %p").time()
    except ValueError:
        messagebox.showerror("Error", "Please enter time in HH:MM AM/PM format.")
        return

    df = df.dropna(subset=["time"])
    df["t"] = df["time"].dt.time
    df_filtered = df[df["t"].between(start_time, end_time)]

    if df_filtered.empty:
        messagebox.showinfo("Info", f"No data between {start_time_entry.get()} and {end_time_entry.get()}.")
        return

    model, scaler, encoder, model_type = load_model(algo)
    if model is None:
        return

    X = preprocess_data(df_filtered, model_type)
    X_scaled = scaler.transform(X)
    if model_type == "CNN":
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        y_pred = np.argmax(model.predict(X_scaled), axis=1)
    elif model_type == "ANN":
        y_pred = np.argmax(model.predict(X_scaled), axis=1)
    else:
        y_pred = model.predict(X_scaled)

    y_label = encoder.inverse_transform(y_pred.astype(int))
    df_filtered["Predicted_Status"] = y_label

    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"Algorithm: {algo}\n")
    output_text.insert(tk.END, f"Time Range: {start_time_entry.get()} - {end_time_entry.get()}\n")
    output_text.insert(tk.END, f"Rows Scanned: {len(df_filtered)}\n")
    output_text.insert(tk.END, "-" * 60 + "\n")

    unsafe_detected = False
    unsafe_rows = []
    for i, row in enumerate(df_filtered.itertuples(index=False)):
        current_time = datetime.now().strftime("%I:%M:%S %p")
        if row.Predicted_Status == "SAFE":
            output_text.insert(tk.END, f"Row {i}: SAFE ✅\n")
        else:
            unsafe_detected = True
            reasons, urgency = [], "Moderate"
            if getattr(row, "temp", 0) > 45:
                reasons.append("🔥 High Temperature (>45°C)")
                urgency = "Critical"
            if getattr(row, "gas", 0) > 300:
                reasons.append("☠️ Gas Leak Detected")
                urgency = "Critical"
            if getattr(row, "heart", 0) < 60:
                reasons.append("💓 Low Heart Rate (<60 bpm)")
                urgency = "Warning"
            if getattr(row, "ir", 0) > 600:
                reasons.append("👋 IR Obstacle Detected")
            if getattr(row, "flame", 0) > 0:
                reasons.append("🔥 Flame Sensor Triggered")
                urgency = "Critical"

            output_text.insert(tk.END, f"Row {i}: UNSAFE ⚠️ ({urgency}) at {current_time}\n")
            for r in reasons:
                output_text.insert(tk.END, f" - {r}\n")
            unsafe_rows.append((i, current_time, reasons, urgency))

    show_alert_banner("UNSAFE" if unsafe_detected else "SAFE")
    display_unsafe_frames(unsafe_rows)
    output_text.config(state=tk.DISABLED)
    if unsafe_detected:
        play_alert_sound()
        send_email_alert(unsafe_rows)


# ===============================================================
# GUI DESIGN
# ===============================================================
root = tk.Tk()
root.title("Smart Safety Analyzer v7")
root.geometry("1020x950")
root.configure(bg="#f1f5f9")

header = tk.Canvas(root, height=90, bg="#1e3a8a", highlightthickness=0)
header.pack(fill="x")
header.create_text(510, 45, text="SMART SAFETY ANALYZER v7",
                   fill="white", font=("Helvetica", 22, "bold"))

alert_label = tk.Label(root, text="✅ SYSTEM SAFE ✅",
                       bg="#16a34a", fg="white", font=("Arial", 15, "bold"))
alert_label.pack(fill="x")

frame_top = tk.Frame(root, bg="#f1f5f9")
frame_top.pack(pady=20)

tk.Label(frame_top, text="Select Algorithm:", font=("Arial", 12, "bold"), bg="#f1f5f9").grid(row=0, column=0, padx=10)
algo_var = tk.StringVar(value="SVM")
algo_menu = ttk.Combobox(frame_top, textvariable=algo_var, values=["SVM", "ANN", "CNN"], width=15, state="readonly")
algo_menu.grid(row=0, column=1, padx=10)

tk.Label(frame_top, text="Start Time (HH:MM AM/PM):", font=("Arial", 12, "bold"), bg="#f1f5f9").grid(row=0, column=2, padx=10)
start_time_entry = tk.Entry(frame_top, width=12, font=("Arial", 12))
start_time_entry.grid(row=0, column=3)
start_time_entry.insert(0, "08:00 AM")

tk.Label(frame_top, text="End Time (HH:MM AM/PM):", font=("Arial", 12, "bold"), bg="#f1f5f9").grid(row=0, column=4, padx=10)
end_time_entry = tk.Entry(frame_top, width=12, font=("Arial", 12))
end_time_entry.grid(row=0, column=5)
end_time_entry.insert(0, "06:00 PM")

btn_frame = tk.Frame(root, bg="#f1f5f9")
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="🕒 Predict by Time Range", bg="#2563eb", fg="white", font=("Arial", 13, "bold"),
          command=predict_by_timerange, width=22).grid(row=0, column=0, padx=10)

output_frame = tk.Frame(root, bg="white", bd=2, relief="groove")
output_frame.pack(pady=10, padx=15, fill="both", expand=True)
output_text = tk.Text(output_frame, height=18, width=95, font=("Consolas", 11), wrap="word", bg="#f9fafb")
output_text.pack(side="left", fill="both", expand=True)
scrollbar = tk.Scrollbar(output_frame, command=output_text.yview)
scrollbar.pack(side="right", fill="y")
output_text.config(yscrollcommand=scrollbar.set, state=tk.DISABLED)

tk.Label(root, text="🚨 UNSAFE DETECTIONS SUMMARY 🚨", font=("Arial", 14, "bold"),
         bg="#f1f5f9", fg="#1e3a8a").pack(pady=(10, 0))
unsafe_frame = tk.Frame(root, bg="#f1f5f9")
unsafe_frame.pack(pady=5, padx=10, fill="both", expand=False)

root.mainloop()
