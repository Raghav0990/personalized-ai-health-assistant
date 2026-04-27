import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.preprocessing import preprocess_input
from utils.model import load_model, predict
from utils.explain import explain_prediction

# Page setup
st.set_page_config(page_title="AI Health Assistant", layout="wide")

st.title("🧠 Personalized AI Health Assistant")
st.markdown("AI-powered health risk prediction with visual analytics")

# Sidebar Inputs
st.sidebar.header("Patient Inputs")

age = st.sidebar.slider("Age", 1, 100, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
heart_rate = st.sidebar.slider("Heart Rate", 40, 180, 80)
steps = st.sidebar.slider("Daily Steps", 0, 20000, 5000)

# Load model
model = load_model()

# Simulated time-series data (for visualization)
days = np.arange(1, 8)
heart_rate_series = heart_rate + np.random.randint(-5, 5, size=7)
steps_series = steps + np.random.randint(-1000, 1000, size=7)

# Predict
if st.button("🔍 Predict Health Risk"):

    features = preprocess_input(age, gender, heart_rate, steps)
    result, prob = predict(model, features)

    # Layout
    col1, col2 = st.columns(2)

    # Result
    with col1:
        st.subheader("Prediction Result")
        if result == "HIGH RISK":
            st.error(f"⚠️ {result}")
        else:
            st.success(f"✅ {result}")

    with col2:
        st.subheader("Risk Probability")
        st.metric("Probability", f"{prob:.2f}")

    # 📊 Graph Section
    st.subheader("📊 Health Trends (Last 7 Days)")

    fig, ax = plt.subplots()
    ax.plot(days, heart_rate_series, label="Heart Rate")
    ax.plot(days, steps_series / 100, label="Steps (scaled)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Values")
    ax.legend()
    st.pyplot(fig)

    # Explanation
    st.subheader("🧾 AI Explanation")
    explanation = explain_prediction(features)

    for point in explanation:
        st.write(f"- {point}")

    # 🤖 AI Recommendations (Dynamic)
    st.subheader("🤖 AI Recommendations")

    recommendations = []

    if age > 50:
        recommendations.append("Focus on regular health checkups")
    if heart_rate > 100:
        recommendations.append("High heart rate detected — consider stress management")
    if steps < 4000:
        recommendations.append("Increase daily activity to at least 7000 steps")
    if result == "HIGH RISK":
        recommendations.append("Consult a healthcare professional immediately")

    if not recommendations:
        recommendations.append("Maintain your current healthy lifestyle")

    for rec in recommendations:
        st.write(f"✔ {rec}")
