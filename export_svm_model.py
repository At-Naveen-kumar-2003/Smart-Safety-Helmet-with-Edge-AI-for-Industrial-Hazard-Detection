# -------------------------------
# File: export_svm_model.py
# -------------------------------
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import joblib, json
import numpy as np

# Example training data (replace with your actual dataset)
X = np.array([
    [1000, 2000, 3000, 25, 50, 400, 70, 1, 1],
    [1500, 2500, 3500, 40, 30, 800, 65, 0, 1],
    [2000, 3000, 4000, 45, 70, 1200, 90, 1, 0],
    [1200, 2200, 3200, 20, 55, 500, 75, 1, 1]
])
y = np.array([0, 1, 1, 0])  # 0 = SAFE, 1 = UNSAFE

# 1️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2️⃣ Train SVM model (RBF kernel)
model = svm.SVC(kernel='rbf', gamma=0.05)
model.fit(X_scaled, y)

# 3️⃣ Export parameters
params = {
    "support_vectors": model.support_vectors_.tolist(),
    "dual_coef": model.dual_coef_.tolist(),
    "intercept": model.intercept_.tolist(),
    "gamma": model._gamma,
    "feature_mean": scaler.mean_.tolist(),
    "feature_std": scaler.scale_.tolist()
}

# 4️⃣ Save to JSON
with open("svm_model.json", "w") as f:
    json.dump(params, f, indent=4)

print("✅ SVM model exported to svm_model.json")
