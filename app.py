import streamlit as st
import numpy as np
import pickle
import requests
import geocoder

# ----------------------------
# 🔑 Your OpenWeatherMap API Key
API_KEY = "6fe290f8f8c522d4991273dc3c3d02a9"

# ----------------------------

# 🌐 Get weather using city name
def get_weather_by_city(city, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        rainfall = data.get('rain', {}).get('1h', 0)
        return temperature, humidity, rainfall
    except:
        return 25.0, 60.0, 100.0

# 🛰️ Get weather using IP-based location
def get_weather_by_ip(api_key):
    try:
        g = geocoder.ip('me')
        lat, lon = g.latlng if g.latlng else (26.1445, 91.7362)
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        rainfall = data.get('rain', {}).get('1h', 0)
        return temperature, humidity, rainfall
    except:
        return 25.0, 60.0, 100.0

# 🎯 Load ML model
with open('models/crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 🎨 UI setup
st.set_page_config(page_title="Crop Advisor 🌾", layout="centered")
st.title("🌿 AI-Based Seasonal Crop Advisor")
st.markdown("Get smart crop recommendations based on your soil and current weather.")

# 📍 City or Auto Location
st.markdown("---")
use_manual = st.checkbox("📍 Enter city manually (override GPS)", value=False)

if use_manual:
    city = st.text_input("Enter City Name", value="Guwahati")
    if city:
        temperature, humidity, rainfall = get_weather_by_city(city, API_KEY)
        st.success(f"✅ Weather fetched for **{city}**")
else:
    temperature, humidity, rainfall = get_weather_by_ip(API_KEY)
    st.success("📡 Weather auto-fetched using your IP location")

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

    # (Optional) Crop Descriptions
    crop_descriptions = {
        "rice": "🌾 Grows well in warm, humid conditions with plenty of water.",
        "wheat": "🌿 Best in cool climates with loamy soil.",
        "maize": "🌽 Requires warm weather and moderate rainfall.",
        "cotton": "🧵 Grows in black soil and sunny conditions.",
        "banana": "🍌 Loves rich soil and humid environments.",
        # Add more as needed...
    }
    desc = crop_descriptions.get(prediction.lower(), "Crop description not available.")
    st.info(desc)
