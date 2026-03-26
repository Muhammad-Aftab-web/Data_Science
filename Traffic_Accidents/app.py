import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🚗 Accident Severity Prediction App")

st.write("Enter accident details:")

# User Inputs
start_lat = st.number_input("Start Latitude", value=0.0)
start_lng = st.number_input("Start Longitude", value=0.0)
hour = st.slider("Hour of Day", 0, 23, 12)
Day_of_Week = st.selectbox("Day of Week", [0,1,2,3,4,5,6])
Month = st.selectbox("Month", [0,1,2,3,4,5,6,7,8,9,10,11])

temperature = st.number_input("Temperature (F)", value=70)
humidity = st.slider("Humidity (%)", 0, 100, 50)
visibility = st.number_input("Visibility (miles)", value=10)
wind_speed = st.number_input("Wind Speed (mph)", value=5)
Precipitation = st.number_input("Precipitation (inches)", value=0)
junction = st.selectbox("Junction", [0,1])
traffic_signal = st.selectbox("Traffic Signal", [0,1])
crossing = st.selectbox("Crossing", [0,1])
Sunrise_Sunset = st.selectbox("Sunrise_Sunset", [0,1])

# Predict button
if st.button("Predict Severity"):
    
    input_data = np.array([[
        start_lat,
        start_lng,
        hour,
        Day_of_Week,
        Month,
        temperature,
        humidity,
        visibility,
        wind_speed,
        Precipitation,
        junction,
        traffic_signal,
        crossing,
        Sunrise_Sunset
    ]])
    
    prediction = model.predict(input_data)
    if prediction == 0:
        prediction = 1
    elif prediction == 1:
        prediction = 2
    elif prediction == 2:
        prediction = 3
    elif prediction == 3:
        prediction = 4
        
    severity = prediction
       # convert back to 1–4
    
    st.success(f"Predicted Accident Severity: {severity}")