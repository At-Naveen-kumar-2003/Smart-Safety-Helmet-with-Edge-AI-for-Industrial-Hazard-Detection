# svm_safe_unsafe.py
# =====================================================
# SVM CLASSIFIER FOR SENSOR DATA (Safe vs Unsafe)
# =====================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 1️⃣ Load Data
df = pd.read_csv("sensor_data.csv")

print("📄 Sample Data:")
print(df.head())

# 2️⃣ Preprocess
df = df.drop(columns=["date", "time"], errors='ignore')
X = df.drop(columns=["status"])
y = df["status"]

# Encode 'SAFE' / 'UNSAFE' → numeric (0/1)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4️⃣ Train SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)

# 5️⃣ Evaluate Model
y_pred = svm_model.predict(X_test)

print("\n✅ Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n📈 Classification Report:\n",
      classification_report(y_test, y_pred, target_names=encoder.classes_))

# 6️⃣ Visualize Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.tight_layout()
plt.show()

# 7️⃣ Save Trained Model + Scaler + Encoder
joblib.dump(svm_model, "svm_safe_unsafe_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "label_encoder.pkl")
print("\n💾 Model saved as 'svm_safe_unsafe_model.pkl'")

# 8️⃣ Predict Specific Row (optional)
row_number = 5  # 🔹 change this index as you like
sample = df.drop(columns=["status"]).iloc[[row_number]]
sample_scaled = scaler.transform(sample)
pred = svm_model.predict(sample_scaled)
label = encoder.inverse_transform(pred)
print(f"\nRow {row_number} predicted as: {label[0]}")
# svm_safe_unsafe_comparison.py
# =====================================================
# SVM CLASSIFIER WITH TRAIN-TEST COMPARISON
# =====================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
    f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 1️⃣ Load Data
df = pd.read_csv("sensor_data.csv")

print("📄 Sample Data:")
print(df.head())
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"🔢 Class Distribution:\n{df['status'].value_counts()}")

# 2️⃣ Preprocess
df = df.drop(columns=["date", "time"], errors='ignore')
X = df.drop(columns=["status"])
y = df["status"]

# Encode 'SAFE' / 'UNSAFE' → numeric (0/1)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(f"\n🎯 Label Encoding: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n📈 Data Split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Testing class distribution: {np.bincount(y_test)}")

# 4️⃣ Train SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)

# 5️⃣ PREDICTIONS & COMPARISON
print("\n" + "=" * 60)
print("🎯 TRAIN-TEST PERFORMANCE COMPARISON")
print("=" * 60)

# Train predictions
y_train_pred = svm_model.predict(X_train)
y_train_proba = svm_model.predict_proba(X_train)

# Test predictions
y_test_pred = svm_model.predict(X_test)
y_test_proba = svm_model.predict_proba(X_test)


# 6️⃣ METRICS COMPARISON
def calculate_metrics(y_true, y_pred, dataset_name):
    """Calculate comprehensive metrics for a dataset"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n📊 {dataset_name} Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")

    return accuracy, precision, recall, f1


# Calculate metrics for both sets
train_metrics = calculate_metrics(y_train, y_train_pred, "TRAINING")
test_metrics = calculate_metrics(y_test, y_test_pred, "TESTING")

# 7️⃣ CONFUSION MATRICES COMPARISON
print("\n" + "=" * 60)
print("📊 CONFUSION MATRICES COMPARISON")
print("=" * 60)

# Calculate confusion matrices
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print(f"\nTraining Confusion Matrix:")
print(cm_train)
print(f"\nTesting Confusion Matrix:")
print(cm_test)

# 8️⃣ VISUALIZATION COMPARISON
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Training Confusion Matrix
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_,
            ax=axes[0, 0])
axes[0, 0].set_title('Training Set - Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# Testing Confusion Matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_,
            ax=axes[0, 1])
axes[0, 1].set_title('Testing Set - Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# 9️⃣ METRICS COMPARISON PLOT
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_values = train_metrics
test_values = test_metrics

x_pos = np.arange(len(metrics_names))
width = 0.35

axes[1, 0].bar(x_pos - width / 2, train_values, width, label='Training', alpha=0.7, color='blue')
axes[1, 0].bar(x_pos + width / 2, test_values, width, label='Testing', alpha=0.7, color='red')
axes[1, 0].set_xlabel('Metrics')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Training vs Testing Metrics Comparison')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(metrics_names)
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 1)

# 🔟 CONFIDENCE DISTRIBUTION COMPARISON
train_confidences = np.max(y_train_proba, axis=1)
test_confidences = np.max(y_test_proba, axis=1)

axes[1, 1].hist(train_confidences, bins=20, alpha=0.7, label='Training', color='blue', density=True)
axes[1, 1].hist(test_confidences, bins=20, alpha=0.7, label='Testing', color='red', density=True)
axes[1, 1].set_xlabel('Prediction Confidence')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Confidence Distribution Comparison')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# 1️⃣1️⃣ DETAILED PERFORMANCE ANALYSIS
print("\n" + "=" * 60)
print("🔍 DETAILED PERFORMANCE ANALYSIS")
print("=" * 60)

# Overfitting analysis
train_accuracy = train_metrics[0]
test_accuracy = test_metrics[0]
accuracy_gap = train_accuracy - test_accuracy

print(f"\n📈 Overfitting Analysis:")
print(f"   Training Accuracy: {train_accuracy:.4f}")
print(f"   Testing Accuracy:  {test_accuracy:.4f}")
print(f"   Accuracy Gap:      {accuracy_gap:.4f}")

if accuracy_gap > 0.1:
    print("   ⚠️  WARNING: Potential overfitting detected (gap > 0.1)")
elif accuracy_gap > 0.05:
    print("   ℹ️  NOTE: Moderate gap (0.05 < gap ≤ 0.1)")
else:
    print("   ✅ GOOD: Minimal overfitting (gap ≤ 0.05)")

# Classification reports
print(f"\n📋 Training Set Classification Report:")
print(classification_report(y_train, y_train_pred, target_names=encoder.classes_))

print(f"\n📋 Testing Set Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=encoder.classes_))

# 1️⃣2️⃣ MISCLASSIFICATION ANALYSIS
print("\n" + "=" * 60)
print("❌ MISCLASSIFICATION ANALYSIS")
print("=" * 60)

# Training misclassifications
train_misclassified = np.where(y_train != y_train_pred)[0]
test_misclassified = np.where(y_test != y_test_pred)[0]

print(
    f"\nTraining Misclassifications: {len(train_misclassified)}/{len(y_train)} ({len(train_misclassified) / len(y_train) * 100:.2f}%)")
print(
    f"Testing Misclassifications:  {len(test_misclassified)}/{len(y_test)} ({len(test_misclassified) / len(y_test) * 100:.2f}%)")

# Confidence analysis for misclassified samples
if len(train_misclassified) > 0:
    train_misconf = train_confidences[train_misclassified]
    print(f"Avg confidence for training misclassifications: {np.mean(train_misconf):.3f}")

if len(test_misclassified) > 0:
    test_misconf = test_confidences[test_misclassified]
    print(f"Avg confidence for testing misclassifications:  {np.mean(test_misconf):.3f}")

# 1️⃣3️⃣ SAVE MODEL AND RESULTS
print("\n" + "=" * 60)
print("💾 SAVING MODEL AND RESULTS")
print("=" * 60)

# Save model and preprocessing objects
joblib.dump(svm_model, "svm_safe_unsafe_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "label_encoder.pkl")

# Save performance results to CSV
results_df = pd.DataFrame({
    'Dataset': ['Training', 'Testing'],
    'Accuracy': [train_metrics[0], test_metrics[0]],
    'Precision': [train_metrics[1], test_metrics[1]],
    'Recall': [train_metrics[2], test_metrics[2]],
    'F1_Score': [train_metrics[3], test_metrics[3]],
    'Samples': [len(y_train), len(y_test)],
    'Misclassifications': [len(train_misclassified), len(test_misclassified)]
})

results_df.to_csv('svm_performance_comparison.csv', index=False)
print("✅ Performance results saved to 'svm_performance_comparison.csv'")

print("\n💾 Model saved as 'svm_safe_unsafe_model.pkl'")

# 1️⃣4️⃣ SAMPLE PREDICTIONS FROM BOTH SETS
print("\n" + "=" * 60)
print("🎯 SAMPLE PREDICTIONS COMPARISON")
print("=" * 60)


def display_sample_predictions(X, y_true, y_pred, y_proba, dataset_name, num_samples=3):
    """Display sample predictions from a dataset"""
    print(f"\n{dataset_name} Set Sample Predictions:")
    indices = np.random.choice(len(y_true), min(num_samples, len(y_true)), replace=False)

    for i, idx in enumerate(indices):
        actual_label = encoder.inverse_transform([y_true[idx]])[0]
        pred_label = encoder.inverse_transform([y_pred[idx]])[0]
        confidence = np.max(y_proba[idx]) * 100
        correct = "✅" if y_true[idx] == y_pred[idx] else "❌"

        print(f"  Sample {i + 1}: Actual: {actual_label}, Predicted: {pred_label}, "
              f"Confidence: {confidence:.2f}% {correct}")


# Display samples from both sets
display_sample_predictions(X_train, y_train, y_train_pred, y_train_proba, "TRAINING")
display_sample_predictions(X_test, y_test, y_test_pred, y_test_proba, "TESTING")

print("\n" + "=" * 60)
print("🎉 SVM TRAIN-TEST COMPARISON COMPLETED!")
print("=" * 60)
