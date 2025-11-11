import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -------------------------------
# Load trained model, scaler, and encoders
# -------------------------------
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Telecom Customer Churn Prediction Dashboard")

# -------------------------------
# Create input fields for categorical features
# -------------------------------
inputs = {}

st.sidebar.header("Customer Inputs")

for col, le in encoders.items():
    inputs[col] = st.sidebar.selectbox(f"{col}", le.classes_)

# -------------------------------
# Create input fields for numeric features
# -------------------------------
# Numeric columns = everything in the dataset that is NOT in encoders
# We save numeric column names in encoders dict for simplicity
# (you can adjust if you know which ones are numeric)
all_features = pickle.load(open("encoders.pkl", "rb"))
numeric_cols = [col for col in model.feature_names_in_ if col not in encoders] \
    if hasattr(model, 'feature_names_in_') else []  # fallback empty

# Optional: Let’s assume numeric columns are the rest (you can list manually)
# For example, if your CSV has these numeric columns:
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']  # <- adjust according to your dataset

for col in numeric_cols:
    inputs[col] = st.sidebar.number_input(f"{col}", value=0.0)

# -------------------------------
# Predict button
# -------------------------------
if st.button("Predict Churn"):
    # Encode categorical inputs
    for col, le in encoders.items():
        inputs[col] = le.transform([inputs[col]])[0]

    # Create DataFrame for prediction
    X_new = pd.DataFrame([inputs])

    # Scale numeric features
    X_new[numeric_cols] = scaler.transform(X_new[numeric_cols])

    # Predict
    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0][1]

    st.subheader("### 🧠 Prediction Result")
    st.write("Churn" if pred == 1 else "No Churn")
    st.write(f"### 🔢 Confidence: {prob:.2%}")
