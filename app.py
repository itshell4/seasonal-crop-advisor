import streamlit as st
import numpy as np
import pickle
import requests
import geocoder

# ----------------------------
# 🔑 Your OpenWeatherMap API Key
API_KEY = "6fe290f8f8c522d4991273dc3c3d02a9"  
# ----------------------------

# 🌐 Get user location using IP
g = geocoder.ip('me')
lat, lon = g.latlng if g.latlng else (26.1445, 91.7362)  # Default: Guwahati

# 🛰️ Get weather data from OpenWeatherMap
def get_weather_data(lat, lon, api_key):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()

        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        rainfall = data.get('rain', {}).get('1h', 0)  # in mm
        return temperature, humidity, rainfall
    except:
        return 25.0, 60.0, 100.0  # Fallback values

# 🎯 Load model
with open('models/crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 🎨 UI setup
st.set_page_config(page_title="Crop Advisor 🌾", layout="centered")
st.title("🌿 AI-Based Seasonal Crop Advisor")
st.markdown("Get smart crop recommendations based on your soil and weather conditions.")

# 🌤️ Get real-time weather
temperature, humidity, rainfall = get_weather_data(lat, lon, API_KEY)
st.success("📡 Weather auto-fetched for your location!")

# 📥 Input form
with st.form("crop_form"):
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=90)
        P = st.number_input("Phosphorous (P)", min_value=0, max_value=150, value=42)
        K = st.number_input("Potassium (K)", min_value=0, max_value=150, value=43)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    with col2:
        st.number_input("Temperature (°C)", value=temperature, disabled=True)
        st.number_input("Humidity (%)", value=humidity, disabled=True)
        st.number_input("Rainfall (mm)", value=rainfall, disabled=True)

    submit = st.form_submit_button("🌱 Recommend Crop")

# 🧠 Predict and Show Result
if submit:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)[0]
    crop_name = prediction.capitalize()

    st.markdown("---")
    st.markdown(f"### ✅ Recommended Crop: **{crop_name}**")

    # (Optional) Description card
    crop_descriptions = {
        "rice": "🌾 Grows best in warm, humid regions with standing water.",
        "wheat": "🌿 Prefers cool climates with loamy soil.",
        "maize": "🌽 Requires moderate rainfall and warm weather.",
        "cotton": "🧵 Needs sunny conditions and well-drained soil.",
        # Add more descriptions as needed
    }
    desc = crop_descriptions.get(prediction.lower(), "No description available.")
    st.info(desc)
