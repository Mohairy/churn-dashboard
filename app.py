import streamlit as st
import pandas as pd
import pickle

# === Load model, scaler, and encoders ===
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict if a customer will churn based on input features.")

# === Create input fields dynamically ===
inputs = {}
for col, enc in encoders.items():
    options = enc.classes_
    inputs[col] = st.selectbox(f"{col}", options)

# For numeric columns (you can adjust these manually)
numeric_cols = [col for col in model.feature_names_in_ if col not in encoders]
for col in numeric_cols:
    inputs[col] = st.number_input(f"{col}", value=0.0)

# === Predict button ===
if st.button("Predict Churn"):
    # Convert categorical to encoded
    for col, enc in encoders.items():
        inputs[col] = enc.transform([inputs[col]])[0]

    # Create dataframe
    X = pd.DataFrame([inputs])

    # Scale numeric features
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.write("### 🧠 Prediction:", "Churn" if prediction == 1 else "No Churn")
    st.write(f"### 🔢 Confidence: {prob:.2%}")
