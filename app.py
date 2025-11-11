import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Telecom Customer Churn Prediction Dashboard")

# === Load model & helpers ===
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
numeric_cols = pickle.load(open("numeric_cols.pkl", "rb"))
training_columns = pickle.load(open("training_columns.pkl", "rb"))

# === Sidebar inputs ===
st.sidebar.header("Customer Inputs")
inputs = {}

# Categorical inputs
for col, le in encoders.items():
    inputs[col] = st.sidebar.selectbox(f"{col}", le.classes_)

# Numeric inputs
for col in numeric_cols:
    inputs[col] = st.sidebar.number_input(f"{col}", value=0.0)

# === Predict button ===
if st.button("Predict Churn"):
    # Encode categorical features
    for col, le in encoders.items():
        inputs[col] = le.transform([inputs[col]])[0]

    # Create DataFrame with correct column order
    X_new = pd.DataFrame([inputs])
    X_new = X_new.reindex(columns=training_columns)

    # Scale numeric columns
    X_new[numeric_cols] = scaler.transform(X_new[numeric_cols])

    # Predict
    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0][1]

    st.subheader("### 🧠 Prediction Result")
    st.write("Churn" if pred == 1 else "No Churn")
    st.write(f"### 🔢 Confidence: {prob:.2%}")
