# =====================================================
# 1D CNN SAFE / UNSAFE CLASSIFIER (Train-Test Comparison)
# Structured for performance comparison with SVM & ANN
# =====================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib

# -----------------------------
# USER SETTINGS
# -----------------------------
CSV_PATH = "sensor_data.csv"
MODEL_OUT = "cnn_safe_unsafe.keras"
SCALER_OUT = "scaler_cnn.pkl"
ENCODER_OUT = "label_encoder_cnn.pkl"
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SPLIT = 0.2
EPOCHS = 150
BATCH_SIZE = 32
# -----------------------------

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =====================================================
# 1️⃣ Load & Prepare Dataset
# =====================================================
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(columns=["date", "time"], errors="ignore", inplace=True)

    # Ensure numeric types for ir/flame
    for col in ["ir", "flame"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Derived features
    df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    df["gas_by_temp"] = df["gas"] / (df["temp"].replace(0, 1))
    df["heart_by_gas"] = df["heart"] / (df["gas"].replace(0, 1))

    # Normalized versions
    for c in ["ax","ay","az","temp","hum","gas","heart","acc_mag"]:
        df[f"{c}_z"] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)

    features = [
        "ax","ay","az","temp","hum","gas","heart","ir","flame",
        "acc_mag","gas_by_temp","heart_by_gas",
        "ax_z","ay_z","az_z","temp_z","hum_z","gas_z","heart_z","acc_mag_z"
    ]
    features = [f for f in features if f in df.columns]

    X = df[features].astype(float)
    y = df["status"].astype(str).str.strip()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le, features

# =====================================================
# 2️⃣ Build 1D CNN Model
# =====================================================
def build_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =====================================================
# 3️⃣ Metrics Utility
# =====================================================
def compute_metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"\n📊 {name} Metrics:")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    return acc, prec, rec, f1

# =====================================================
# 4️⃣ Main Training + Comparison
# =====================================================
def main():
    print("🚀 Loading and preparing data...")
    X, y, le, features = load_and_prepare(CSV_PATH)
    print(f"✅ Loaded {len(X)} samples with {len(features)} features.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    cw = class_weight.compute_class_weight(class_weight="balanced",
                                           classes=np.unique(y_train), y=y_train)
    cw = {i: w for i, w in enumerate(cw)}

    print(f"\n📈 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"⚖️ Class Weights: {cw}")

    model = build_cnn((X_train.shape[1], 1))
    model.summary()

    cb = [
        callbacks.ModelCheckpoint("best_cnn_model.keras", monitor="val_loss", save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
    ]

    print("\n🎯 Training CNN...")
    history = model.fit(
        X_train, y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=cb,
        verbose=1
    )

    # Save model & preprocessors
    model.save(MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(le, ENCODER_OUT)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # Metrics
    train_metrics = compute_metrics(y_train, y_train_pred, "Training")
    test_metrics = compute_metrics(y_test, y_test_pred, "Testing")

    # Overfitting check
    acc_gap = train_metrics[0] - test_metrics[0]
    print(f"\n📈 Overfitting Check: Accuracy Gap = {acc_gap:.4f}")
    if acc_gap > 0.1:
        print("⚠️ High overfitting!")
    elif acc_gap > 0.05:
        print("ℹ️ Moderate gap.")
    else:
        print("✅ Minimal overfitting.")

    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
    axes[0].set_title("Training Confusion Matrix")
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
    axes[1].set_title("Testing Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Metrics bar chart
    metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, train_metrics, width, label="Training", color="blue", alpha=0.7)
    plt.bar(x + width/2, test_metrics, width, label="Testing", color="red", alpha=0.7)
    plt.xticks(x, metrics_names)
    plt.ylabel("Score")
    plt.title("1D CNN Training vs Testing Metrics Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Save results for comparison
    results_df = pd.DataFrame({
        "Dataset": ["Training", "Testing"],
        "Accuracy": [train_metrics[0], test_metrics[0]],
        "Precision": [train_metrics[1], test_metrics[1]],
        "Recall": [train_metrics[2], test_metrics[2]],
        "F1_Score": [train_metrics[3], test_metrics[3]],
        "Samples": [len(y_train), len(y_test)],
        "Misclassifications": [
            len(np.where(y_train != y_train_pred)[0]),
            len(np.where(y_test != y_test_pred)[0])
        ]
    })
    results_df.to_csv("cnn_performance_comparison.csv", index=False)
    print("\n💾 Results saved to 'cnn_performance_comparison.csv'")
    print(f"💾 Model saved as '{MODEL_OUT}'")

    print("\n✅ 1D CNN Train-Test Comparison Completed Successfully!")

if __name__ == "__main__":
    main()
