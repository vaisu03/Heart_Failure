import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("heart_failure_model.pkl")

# Title
st.title("💓 Heart Failure Prediction System")
st.write("Predict the survival of heart failure patients using an AIW-PSO optimized Ensemble Model.")

# Input options
option = st.radio("Choose Input Method:", ["Enter Manually", "Upload CSV"])

# Manual Entry Form
if option == "Enter Manually":
    age = st.number_input("Age", 20, 100, 50)
    anaemia = st.selectbox("Anaemia (0 = No, 1 = Yes)", [0, 1])
    creatinine_phosphokinase = st.number_input("CPK Level", 20, 8000, 500)
    diabetes = st.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1])
    ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 40)
    high_blood_pressure = st.selectbox("High Blood Pressure (0 = No, 1 = Yes)", [0, 1])
    platelets = st.number_input("Platelets", 100000, 800000, 250000)
    serum_creatinine = st.number_input("Serum Creatinine", 0.1, 10.0, 1.2)
    serum_sodium = st.number_input("Serum Sodium", 100, 150, 138)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
    time = st.number_input("Follow-up Time (days)", 1, 300, 130)

    if st.button("Predict"):
        data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                          high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                          sex, smoking, time]])
        prediction = model.predict(data)[0]
        st.success("✅ Prediction: **Survived**" if prediction == 0 else "❌ Prediction: **Not Survived**")

# CSV Upload
else:
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("Uploaded Data Preview:", df.head())
        preds = model.predict(df)
        df["Prediction"] = ["Survived" if p == 0 else "Not Survived" for p in preds]
        st.write("Predictions:", df)
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")


