# ===============================================================
# SMART SAFETY ANALYZER (v17)
# - 12-hour Time Picker
# - Email Alert + Confusion Matrix + PDF Export (Times New Roman)
# - PDF Button next to Confusion Matrix
# ===============================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import winsound
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ------------------ CONFIG ------------------
DATA_FILE = "sensor_data.csv"
SVM_MODEL = "svm_safe_unsafe_model.pkl"
ANN_MODEL = "ann_safe_unsafe.keras"
CNN_MODEL = "cnn_safe_unsafe.keras"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER = "label_encoder.pkl"

SENDER_EMAIL = "goat76305@gmail.com"
SENDER_PASSWORD = "zeyyhvmbkcgonhuy"
RECEIVER_EMAIL = "sanjaisaravanan8870@gmail.com"

last_predicted_df = None


# ------------------ EMAIL ALERT ------------------
def send_email_alert(unsafe_rows):
    try:
        if not unsafe_rows:
            return
        body_lines = [
            "⚠️ UNSAFE CONDITION ALERT ⚠️",
            "",
            "The following unsafe conditions were detected:",
            "",
        ]
        for idx, timestamp, reasons, urgency in unsafe_rows:
            body_lines.append(f"➡ Row {idx} | {urgency} | Time: {timestamp}")
            for r in reasons:
                body_lines.append(f"   - {r}")
            body_lines.append("")
        body_lines.append("✅ Please take immediate corrective action.\n\nSmart Safety Analyzer v17")

        body = "\n".join(body_lines)
        msg = MIMEText(body)
        msg["Subject"] = "⚠️ Smart Safety Analyzer - UNSAFE Condition Detected"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        messagebox.showinfo("Email Notification", "✅ Email alert sent successfully!")
    except Exception as e:
        messagebox.showerror("Email failed", f"{e}")


# ------------------ LOAD + MODEL ------------------
def load_csv():
    try:
        df = pd.read_csv(DATA_FILE)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
        return df
    except FileNotFoundError:
        messagebox.showerror("Error", f"Dataset '{DATA_FILE}' not found!")
        return None


def preprocess_data(df, model_type):
    df = df.copy()
    if model_type == "SVM":
        cols = ["ax", "ay", "az", "temp", "hum", "gas", "heart", "ir", "flame"]
    else:
        df["acc_mag"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
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
        messagebox.showerror("Error", f"Failed to load model: {e}")
        return None, None, None, None


# ------------------ PDF EXPORT ------------------
def download_time_range_pdf():
    global last_predicted_df
    if last_predicted_df is None or last_predicted_df.empty:
        messagebox.showinfo("Info", "No prediction data available to export.")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF Files", "*.pdf")],
        title="Save Time Range Prediction Report"
    )
    if not save_path:
        return

    styles = getSampleStyleSheet()
    style = ParagraphStyle("Times", parent=styles["Normal"], fontName="Times-Roman", fontSize=12, leading=14)
    doc = SimpleDocTemplate(save_path, pagesize=A4)
    story = [
        Paragraph("<b>SMART SAFETY ANALYZER - TIME RANGE PREDICTION REPORT</b>", style),
        Spacer(1, 10),
        Paragraph(f"Generated on: {datetime.now().strftime('%d-%m-%Y %I:%M:%S %p')}", style),
        Spacer(1, 15)
    ]

    for _, row in last_predicted_df.iterrows():
        time_str = row["time"].strftime("%I:%M:%S %p")
        story.append(Paragraph(f"<b>Time:</b> {time_str} | <b>Status:</b> {row['Predicted_Status']}", style))
    doc.build(story)
    messagebox.showinfo("PDF Saved", f"✅ Report saved successfully:\n{save_path}")


# ------------------ CONFUSION MATRIX ------------------
def show_confusion_matrix():
    algo = algo_var.get()
    df = load_csv()
    if df is None or "status" not in df.columns:
        messagebox.showerror("Error", "Dataset must include 'status' column.")
        return
    model, scaler, encoder, model_type = load_model(algo)
    if model is None:
        return

    X = preprocess_data(df, model_type)
    X_scaled = scaler.transform(X)
    if model_type == "CNN":
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        y_pred = np.argmax(model.predict(X_scaled), axis=1)
    elif model_type == "ANN":
        y_pred = np.argmax(model.predict(X_scaled), axis=1)
    else:
        y_pred = model.predict(X_scaled)

    y_true_enc = encoder.transform(df["status"].values)
    cm = confusion_matrix(y_true_enc, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap="coolwarm", fmt="d",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title(f"{algo} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# ------------------ ALERT ------------------
def play_alert_sound():
    winsound.Beep(1200, 400)
    winsound.Beep(800, 400)


def show_alert_banner(status):
    if status == "UNSAFE":
        alert_label.config(bg="#b91c1c", text="🚨 UNSAFE DETECTED! 🚨")
    else:
        alert_label.config(bg="#16a34a", text="✅ SYSTEM SAFE ✅")


# ------------------ PREDICTION ------------------
def predict_time_range():
    global last_predicted_df
    algo = algo_var.get()
    try:
        sh, sm, ss = int(hour_start.get()), int(min_start.get()), int(sec_start.get())
        eh, em, es = int(hour_end.get()), int(min_end.get()), int(sec_end.get())
    except:
        messagebox.showerror("Error", "Please select valid time values.")
        return

    ampm = time_period.get()
    def to_24h(h, ampm_val):
        if ampm_val == "PM" and h != 12: return h + 12
        if ampm_val == "AM" and h == 12: return 0
        return h

    sh24, eh24 = to_24h(sh, ampm), to_24h(eh, ampm)
    df = load_csv()
    if df is None or "time" not in df.columns:
        return

    model, scaler, encoder, model_type = load_model(algo)
    if model is None:
        return

    df = df.dropna(subset=["time"])
    tstart = datetime(2000, 1, 1, sh24, sm, ss).time()
    tend = datetime(2000, 1, 1, eh24, em, es).time()

    if tstart <= tend:
        df_filtered = df[(df["time"].dt.time >= tstart) & (df["time"].dt.time <= tend)].copy()
    else:
        df_filtered = df[(df["time"].dt.time >= tstart) | (df["time"].dt.time <= tend)].copy()

    if df_filtered.empty:
        messagebox.showinfo("Info", "No data found for the selected time range.")
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
    last_predicted_df = df_filtered.copy()

    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    unsafe_detected = False
    unsafe_rows = []

    for idx, row in enumerate(df_filtered.itertuples(index=False)):
        time_str = row.time.strftime("%I:%M:%S %p")
        if row.Predicted_Status == "SAFE":
            output_text.insert(tk.END, f"{time_str}: SAFE ✅\n")
        else:
            unsafe_detected = True
            reasons, urgency = [], "Moderate"
            if getattr(row, "temp", 0) > 45:
                reasons.append("🔥 High Temperature (>45°C)")
                urgency = "Critical"
            if getattr(row, "gas", 0) > 300:
                reasons.append("☠️ Gas Leak — toxic environment")
                urgency = "Critical"
            if getattr(row, "flame", 0) > 0:
                reasons.append("🔥 Flame Sensor Triggered — fire hazard")
                urgency = "Critical"
            output_text.insert(tk.END, f"{time_str}: UNSAFE ⚠️ ({urgency})\n")
            for r in reasons:
                output_text.insert(tk.END, f" - {r}\n")
            unsafe_rows.append((idx, time_str, reasons, urgency))

    output_text.config(state=tk.DISABLED)
    show_alert_banner("UNSAFE" if unsafe_detected else "SAFE")

    if unsafe_detected:
        play_alert_sound()
        send_email_alert(unsafe_rows)


# ------------------ GUI ------------------
root = tk.Tk()
root.title("Smart Safety Analyzer v17")
root.geometry("1080x900")
root.configure(bg="#f1f5f9")

header = tk.Canvas(root, height=90, bg="#1e3a8a", highlightthickness=0)
header.pack(fill="x")
header.create_text(540, 45, text="SMART SAFETY HELMET CLASSIFIER", fill="white", font=("Helvetica", 22, "bold"))

alert_label = tk.Label(root, text="✅ SYSTEM SAFE ✅", bg="#16a34a", fg="white", font=("Arial", 15, "bold"))
alert_label.pack(fill="x", pady=(0, 6))

frame_top = tk.Frame(root, bg="#f1f5f9")
frame_top.pack(pady=12)

tk.Label(frame_top, text="Algorithm:", font=("Arial", 12, "bold"), bg="#f1f5f9").grid(row=0, column=0, padx=8)
algo_var = tk.StringVar(value="SVM")
algo_menu = ttk.Combobox(frame_top, textvariable=algo_var, values=["SVM", "ANN", "CNN"], width=12, state="readonly")
algo_menu.grid(row=0, column=1, padx=8)

# Time pickers
tk.Label(frame_top, text="Start Time:", font=("Arial", 12, "bold"), bg="#f1f5f9").grid(row=0, column=2)
hour_start = ttk.Combobox(frame_top, values=[f"{i:02}" for i in range(1, 13)], width=3)
min_start = ttk.Combobox(frame_top, values=[f"{i:02}" for i in range(0, 60)], width=3)
sec_start = ttk.Combobox(frame_top, values=[f"{i:02}" for i in range(0, 60)], width=3)
hour_start.set("08"); min_start.set("50"); sec_start.set("00")
hour_start.grid(row=0, column=3); min_start.grid(row=0, column=4); sec_start.grid(row=0, column=5)

tk.Label(frame_top, text="End Time:", font=("Arial", 12, "bold"), bg="#f1f5f9").grid(row=0, column=6)
hour_end = ttk.Combobox(frame_top, values=[f"{i:02}" for i in range(1, 13)], width=3)
min_end = ttk.Combobox(frame_top, values=[f"{i:02}" for i in range(0, 60)], width=3)
sec_end = ttk.Combobox(frame_top, values=[f"{i:02}" for i in range(0, 60)], width=3)
hour_end.set("09"); min_end.set("00"); sec_end.set("00")
hour_end.grid(row=0, column=7); min_end.grid(row=0, column=8); sec_end.grid(row=0, column=9)

time_period = tk.StringVar(value="PM")
tk.Radiobutton(frame_top, text="AM", variable=time_period, value="AM", bg="#f1f5f9", font=("Arial", 11)).grid(row=0, column=10)
tk.Radiobutton(frame_top, text="PM", variable=time_period, value="PM", bg="#f1f5f9", font=("Arial", 11)).grid(row=0, column=11)

# Buttons in one row
btn_frame = tk.Frame(root, bg="#f1f5f9")
btn_frame.pack(pady=12)
tk.Button(btn_frame, text="⏰ Classification Time Window", bg="#2563eb", fg="white", font=("Arial", 13, "bold"),
          command=predict_time_range, width=30).grid(row=0, column=0, padx=8)
tk.Button(btn_frame, text="📊 Confusion Matrix", bg="#0ea5e9", fg="white", font=("Arial", 13, "bold"),
          command=show_confusion_matrix, width=20).grid(row=0, column=1, padx=8)
tk.Button(btn_frame, text="📄 Download Report (PDF)", bg="#7c3aed", fg="white", font=("Arial", 13, "bold"),
          command=download_time_range_pdf, width=25).grid(row=0, column=2, padx=8)

output_text = tk.Text(root, height=26, width=120, font=("Consolas", 11))
output_text.pack(pady=8, padx=20, fill="both", expand=True)

root.mainloop()
