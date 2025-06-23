import streamlit as st
import pickle
import requests
import datetime
import pandas as pd
import numpy as np
import geocoder
from collections import defaultdict

# Load the trained model
with open('models/crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

API_KEY = st.secrets["API_KEY"]

# 📅 Final Month-to-Crop Mapping (22 crops)
SEASONAL_CROPS = {
    "January": ['apple', 'banana', 'chickpea', 'coconut', 'grapes', 'kidneybeans', 'lentil', 'orange', 'pomegranate'],
    "February": ['apple', 'banana', 'chickpea', 'coconut', 'grapes', 'kidneybeans', 'lentil', 'orange', 'pomegranate'],
    "March": ['apple', 'banana', 'chickpea', 'coconut', 'grapes', 'kidneybeans', 'lentil', 'orange', 'pomegranate'],
    "April": ['apple', 'banana', 'coconut', 'grapes', 'mango', 'muskmelon', 'orange', 'papaya', 'pomegranate', 'watermelon'],
    "May": ['apple', 'banana', 'coconut', 'grapes', 'mango', 'muskmelon', 'orange', 'papaya', 'pomegranate', 'watermelon'],
    "June": ['apple', 'banana', 'blackgram', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'maize', 'mango', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon'],
    "July": ['apple', 'banana', 'blackgram', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'maize', 'mungbean', 'orange', 'pigeonpeas', 'pomegranate', 'rice'],
    "August": ['apple', 'banana', 'blackgram', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'maize', 'mungbean', 'orange', 'pigeonpeas', 'pomegranate', 'rice'],
    "September": ['apple', 'banana', 'blackgram', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'maize', 'mungbean', 'orange', 'pigeonpeas', 'pomegranate', 'rice'],
    "October": ['apple', 'banana', 'blackgram', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'maize', 'mungbean', 'orange', 'pigeonpeas', 'pomegranate', 'rice'],
    "November": ['apple', 'banana', 'chickpea', 'coconut', 'grapes', 'kidneybeans', 'lentil', 'orange', 'pomegranate'],
    "December": ['apple', 'banana', 'chickpea', 'coconut', 'grapes', 'kidneybeans', 'lentil', 'orange', 'pomegranate']
}

# Weather API fetch
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200 and "main" in data:
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return temperature, humidity
    else:
        return None, None

# Streamlit UI
st.set_page_config(page_title="Seasonal Crop Advisor 🌾", layout="centered")
st.title("🌾 AI-Based Seasonal Crop Advisor")
st.markdown("Enter soil values and your city to get season-aware crop suggestions.")

# Location detection
g = geocoder.ip('me')
default_city = g.city if g.city else ""

city = st.text_input("📍 Enter your City", value=default_city)

temperature, humidity = None, None

if city:
    temperature, humidity = get_weather(city)
    if temperature is not None:
        st.success(f"✅ Weather in {city}: 🌡️ {temperature}°C, 💧 {humidity}% humidity")
    else:
        st.warning("⚠️ Could not fetch weather. Please check the city name.")

# User input
st.subheader("🧪 Enter Soil Nutrient & pH Levels")

N = st.slider("Nitrogen (N)", 0, 140, 50)
P = st.slider("Phosphorus (P)", 5, 145, 50)
K = st.slider("Potassium (K)", 5, 205, 50)
ph = st.slider("pH Level", 3.5, 9.5, 6.5)

rainfall = st.number_input("🌧️ Rainfall (in mm)", min_value=0.0, max_value=400.0, value=100.0, step=0.1)


# Predict button
if st.button("🚀 Predict Crop"):
    if temperature is not None and humidity is not None:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)[0]

        current_month = datetime.datetime.now().strftime("%B")
        valid_crops = SEASONAL_CROPS.get(current_month, [])

        st.markdown(f"📅 **Month**: `{current_month}`")
        st.markdown(f"🔍 **Predicted Crop**: `{prediction}`")

        if prediction in valid_crops:
            st.success("✅ This crop is suitable for the current season!")
        else:
            st.warning(f"⚠️ {prediction} is not ideal for {current_month}.")
            st.info("👉 Try crops like: " + ", ".join(valid_crops))
    else:
        st.error("🌐 Weather data missing. Please enter a valid city.")
