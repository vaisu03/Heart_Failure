import streamlit as st
import pandas as pd
import joblib

# Load trained Type II model
model = joblib.load("heart_failure_model.pkl")

# The exact feature order from training
FEATURES = ['age', 'anaemia', 'ejection_fraction', 'high_blood_pressure',
            'serum_creatinine', 'serum_sodium', 'time']

st.title("💓 Heart Failure Prediction (Type II - AIW-PSO GBM)")

st.write("Enter patient details:")

# Collect inputs in same order as FEATURES
age = st.number_input("Age", min_value=1, max_value=120, value=60)
anaemia = st.selectbox("Anaemia (0 = No, 1 = Yes)", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=1, max_value=100, value=38)
high_blood_pressure = st.selectbox("High Blood Pressure (0 = No, 1 = Yes)", [0, 1])
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.2)
serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=200, value=138)
time = st.number_input("Follow-up Time (days)", min_value=1, max_value=300, value=120)

# Prepare data for prediction
data = pd.DataFrame([[age, anaemia, ejection_fraction, high_blood_pressure,
                      serum_creatinine, serum_sodium, time]],
                    columns=FEATURES)

if st.button("Predict Survival"):
    prediction = model.predict(data)[0]
    if prediction == 1:
        st.error("❌ Patient is at high risk of death")
    else:
        st.success("✅ Patient is likely to survive")
