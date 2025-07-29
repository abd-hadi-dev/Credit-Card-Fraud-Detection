import streamlit as st
import numpy as np
import pickle

# Load model
with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# App title
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter the transaction details to check if it's **fraudulent** or **legitimate**.")

# Create input fields
st.header("🔢 Transaction Features")
input_values = []

for i in range(1, 29):  # V1 to V28
    val = st.number_input(f"V{i}", step=0.01, format="%.4f")
    input_values.append(val)

# Scaled amount
amount = st.number_input("Transaction Amount (scaled)", step=0.01, format="%.4f")
input_values.append(amount)

# Prediction
if st.button("🔍 Predict"):
    input_array = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    
    if prediction == 1:
        st.error("🚨 Alert! This transaction is **FRAUDULENT**.")
    else:
        st.success("✅ This transaction is **LEGITIMATE**.")

# Footer
st.markdown("---")
st.markdown("Developed using **XGBoost & Streamlit** | [@YourName](https://github.com/yourname)")
