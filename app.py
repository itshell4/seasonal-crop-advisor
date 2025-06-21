import streamlit as st
import numpy as np
import pickle

# Load trained model
with open('model/crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("🌾 AI-Based Seasonal Crop Advisor")
st.markdown("Get the best crop recommendation based on soil and weather parameters.")

# Input features
st.header("📥 Enter Soil & Weather Details:")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=150)
P = st.number_input("Phosphorous (P)", min_value=0, max_value=150)
K = st.number_input("Potassium (K)", min_value=0, max_value=150)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0)

# Predict button
if st.button("🌱 Recommend Crop"):
    # Prepare input
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Crop: **{prediction[0].capitalize()}**")
