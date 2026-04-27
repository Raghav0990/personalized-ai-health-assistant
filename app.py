import streamlit as st
from utils.preprocessing import preprocess_input
from utils.model import load_model, predict
from utils.explain import explain_prediction

st.title("Personalized AI Health Assistant")

age = st.number_input("Age", 1, 100)
gender = st.selectbox("Gender", ["Male", "Female"])
heart_rate = st.number_input("Heart Rate")
steps = st.number_input("Steps")

model = load_model()

if st.button("Predict"):
    features = preprocess_input(age, gender, heart_rate, steps)
    result, prob = predict(model, features)
    st.write(result, prob)
