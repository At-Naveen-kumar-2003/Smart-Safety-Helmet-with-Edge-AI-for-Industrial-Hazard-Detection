# quick_test_prediction.py
# =====================================================
# QUICK TEST PREDICTION FUNCTION
# =====================================================

import joblib
import numpy as np
import pandas as pd


def quick_svm_prediction(input_data, model_path="svm_safe_unsafe_model.pkl"):
    """
    Quick prediction function for SVM model

    Parameters:
    input_data: list, numpy array, or pandas DataFrame of sensor values
    model_path: path to saved SVM model

    Returns:
    prediction: 'SAFE' or 'UNSAFE'
    confidence: probability percentage
    """

    # Load model and preprocessing objects
    svm_model = joblib.load(model_path)
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")

    # Convert input to numpy array
    if isinstance(input_data, pd.DataFrame):
        input_array = input_data.values
    else:
        input_array = np.array(input_data)

    # Reshape if single sample
    if len(input_array.shape) == 1:
        input_array = input_array.reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict
    predictions = svm_model.predict(input_scaled)
    probabilities = svm_model.predict_proba(input_scaled)

    # Convert to labels
    predicted_labels = encoder.inverse_transform(predictions)

    # Calculate confidence
    confidences = np.max(probabilities, axis=1) * 100

    return predicted_labels, confidences, probabilities


# Example usage:
if __name__ == "__main__":
    # Test with single sample
    sample_input = [25.5, 65.2, 1013.2, 45.1, 0.8]

    labels, confidences, probabilities = quick_svm_prediction(sample_input)

    print(f"Input: {sample_input}")
    print(f"Prediction: {labels[0]}")
    print(f"Confidence: {confidences[0]:.2f}%")
    print(f"Probabilities - Safe: {probabilities[0][1] * 100:.2f}%, Unsafe: {probabilities[0][0] * 100:.2f}%")