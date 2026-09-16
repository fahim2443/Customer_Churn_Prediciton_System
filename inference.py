"""
Inference script for the Telco Customer Churn model.

Loads model.pkl and encoder.pkl produced by the training notebook,
and predicts churn (0 = No, 1 = Yes) for one or more new customers.

Usage:
    python inference.py
"""

import pickle
import pandas as pd


def load_artifacts(model_path="model.pkl", encoder_path="encoder.pkl"):
    with open(model_path, "rb") as f:
        model_data = f and pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)
    return model_data["model"], model_data["features_name"], encoders


def preprocess(customer: dict, feature_names: list, encoders: dict) -> pd.DataFrame:
    """
    Turns a single raw customer record (same columns/format as the
    original CSV, minus customerID and Churn) into the encoded row
    the model expects.
    """
    df = pd.DataFrame([customer])

    # Apply the same LabelEncoders used at training time, column by column.
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Make sure column order matches what the model was trained on.
    df = df[feature_names]
    return df


def predict(customer: dict, model, feature_names: list, encoders: dict):
    row = preprocess(customer, feature_names, encoders)
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]  # probability of churn=1
    return pred, proba


if __name__ == "__main__":
    model, feature_names, encoders = load_artifacts()

    # Example customer — same fields as the raw dataset, minus customerID/Churn.
    # Replace with real input (from a form, API request, CSV row, etc).
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    pred, proba = predict(sample_customer, model, feature_names, encoders)
    label = "Churn" if pred == 1 else "No Churn"
    print(f"Prediction: {label}  (churn probability: {proba:.2%})")