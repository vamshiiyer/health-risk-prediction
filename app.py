import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("health_risk_model.pkl")

st.set_page_config(
    page_title="AI Health Risk Prediction",
    layout="centered"
)

# -----------------------------
# Title & Description
# -----------------------------
st.title("🩺 AI Health Risk Prediction System")

st.write("""
This **web-based AI system** supports **early risk screening** for:

• **Diabetes**  
• **Blood Pressure (Hypertension)**  
• **General Lifestyle Health**

Users must **enter their own health values** to receive predictions.
""")

st.divider()

# -----------------------------
# Input Section (NO DEFAULTS)
# -----------------------------
st.subheader("🧾 Enter Health Details")

age = st.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=None,
    placeholder="Enter age"
)

height_cm = st.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=250.0,
    value=None,
    placeholder="Enter height in cm"
)

weight_kg = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=None,
    placeholder="Enter weight in kg"
)

bp = st.number_input(
    "Blood Pressure (Systolic mmHg)",
    min_value=80,
    max_value=200,
    value=None,
    placeholder="Enter systolic BP"
)

sugar = st.number_input(
    "Blood Sugar Level (mg/dL)",
    min_value=60,
    max_value=300,
    value=None,
    placeholder="Enter blood sugar"
)

smoker = st.selectbox(
    "Smoking Status",
    ["Select option", "Non-Smoker", "Smoker"]
)

activity = st.selectbox(
    "Physical Activity",
    ["Select option", "Active", "Not Active"]
)

# -----------------------------
# Buttons
# -----------------------------
col1, col2 = st.columns(2)
predict_clicked = col1.button("🔍 Predict Health Risk")
reset_clicked = col2.button("🔄 Reset")

if reset_clicked:
    st.experimental_rerun()

# -----------------------------
# Prediction Logic
# -----------------------------
if predict_clicked:

    # Validation: ensure all fields are filled
    if None in [age, height_cm, weight_kg, bp, sugar] \
       or smoker == "Select option" \
       or activity == "Select option":

        st.error("❗ Please fill in all fields before prediction.")
        st.stop()

    # BMI Calculation
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 2)

    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    smoker_val = 1 if smoker == "Smoker" else 0
    activity_val = 1 if activity == "Active" else 0

    user_data = pd.DataFrame(
        [[age, bmi, bp, sugar, smoker_val, activity_val]],
        columns=["age", "bmi", "bp", "sugar", "smoker", "activity"]
    )

    risk = model.predict(user_data)[0]
    risk_prob = model.predict_proba(user_data).max() * 100

    st.divider()

    # -----------------------------
    # Result
    # -----------------------------
    if risk == 0:
        st.success("🟢 Overall Health Risk: LOW")
    elif risk == 1:
        st.warning("🟡 Overall Health Risk: MEDIUM")
    else:
        st.error("🔴 Overall Health Risk: HIGH")

    st.metric("AI Risk Confidence", f"{risk_prob:.1f}%")
    st.progress(risk_prob / 100)

    # -----------------------------
    # Dashboard
    # -----------------------------
    st.subheader("📊 Health Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("BMI", bmi, bmi_category)
    c2.metric("Blood Pressure", bp, "mmHg")
    c3.metric("Blood Sugar", sugar, "mg/dL")

    # -----------------------------
    # Explainable AI
    # -----------------------------
    st.subheader("🤖 Why this result?")

    reasons = []
    if bmi_category != "Normal":
        reasons.append(f"BMI is {bmi_category}")
    if bp > 130:
        reasons.append("Blood pressure is elevated")
    if sugar > 140:
        reasons.append("Blood sugar level is high")
    if smoker_val == 1:
        reasons.append("Smoking habit detected")
    if activity_val == 0:
        reasons.append("Low physical activity")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("• All indicators are within healthy ranges")

    st.divider()
    st.caption(
        "⚠️ This tool provides health risk awareness only and does not replace medical diagnosis."
    )
