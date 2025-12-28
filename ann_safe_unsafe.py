# =====================================================
#  ANN SAFE / UNSAFE CLASSIFIER (Train vs Test Comparison)
#  Complete performance and visualization version
# =====================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# USER SETTINGS
# -----------------------------
CSV_PATH = "sensor_data.csv"
MODEL_OUT = "ann_safe_unsafe.keras"
SCALER_OUT = "scaler_ann.pkl"
ENCODER_OUT = "label_encoder_ann.pkl"
RANDOM_SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 100
# -----------------------------

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =====================================================
# 1. LOAD AND PREPARE DATA
# =====================================================
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(columns=["date", "time"], errors="ignore", inplace=True)

    # Convert possible string columns to numeric
    for col in ["ir", "flame"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Derived features
    df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    df["gas_by_temp"] = df["gas"] / (df["temp"].replace(0, 1))
    df["heart_by_gas"] = df["heart"] / (df["gas"].replace(0, 1))

    features = [
        "ax","ay","az","temp","hum","gas","heart","ir","flame",
        "acc_mag","gas_by_temp","heart_by_gas"
    ]
    X = df[features].astype(float)
    y = df["status"].astype(str).str.strip()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le, features

# =====================================================
# 2. BUILD ANN MODEL
# =====================================================
def build_ann(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu", kernel_initializer="he_uniform"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(32, activation="relu", kernel_initializer="he_uniform"),
        Dropout(0.25),

        Dense(2, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =====================================================
# 3. MAIN FUNCTION
# =====================================================
def main():
    print("🚀 Loading dataset...")
    X, y, le, features = load_and_prepare(CSV_PATH)
    print(f"✅ Features: {len(features)} | Samples: {len(X)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\n📊 Data Split: {X_train.shape[0]} train | {X_test.shape[0]} test")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Class weights
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = {i: w for i, w in enumerate(cw)}

    # Build + Train model
    model = build_ann(X_train.shape[1])
    print("\n🏗️ Model Summary:")
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=0),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
    ]

    print("\n🎯 Training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )

    # Save preprocessing
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(le, ENCODER_OUT)

    # -----------------------------
    #  Evaluate both Train & Test
    # -----------------------------
    print("\n🧠 Evaluating model...")
    y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # Metrics
    def calculate_metrics(y_true, y_pred, name):
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

    train_metrics = calculate_metrics(y_train, y_train_pred, "Training")
    test_metrics = calculate_metrics(y_test, y_test_pred, "Testing")

    # -----------------------------
    #  Confusion Matrices
    # -----------------------------
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
    axes[0].set_title("Training Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
    axes[1].set_title("Testing Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    #  Metrics Comparison Plot
    # -----------------------------
    metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
    train_values = train_metrics
    test_values = test_metrics

    plt.figure(figsize=(8, 5))
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.bar(x - width/2, train_values, width, label="Training", color="blue", alpha=0.7)
    plt.bar(x + width/2, test_values, width, label="Testing", color="green", alpha=0.7)
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Training vs Testing Metrics Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    #  Overfitting Analysis
    # -----------------------------
    acc_gap = train_metrics[0] - test_metrics[0]
    print(f"\n📈 Overfitting Analysis:")
    print(f"   Training Accuracy: {train_metrics[0]:.4f}")
    print(f"   Testing Accuracy : {test_metrics[0]:.4f}")
    print(f"   Accuracy Gap     : {acc_gap:.4f}")

    if acc_gap > 0.1:
        print("   ⚠️ Potential overfitting detected.")
    elif acc_gap > 0.05:
        print("   ℹ️ Moderate gap.")
    else:
        print("   ✅ Minimal overfitting (Good).")

    # -----------------------------
    #  Save CSV Results
    # -----------------------------
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
    results_df.to_csv("ann_performance_comparison.csv", index=False)
    print("\n💾 Results saved to 'ann_performance_comparison.csv'")

    print(f"\n💾 Model saved as {MODEL_OUT}")
    print("🎉 ANN Train-Test comparison completed successfully!")

if __name__ == "__main__":
    main()
