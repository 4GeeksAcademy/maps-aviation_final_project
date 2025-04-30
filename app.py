import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.title("Worried Your Plane might Crash?")
st.subheader("Submit your flight details to find out.")
st.divider()

st.image(os.path.join(os.getcwd(), "static", "photo.jpg"))

flight_details = {
        "origin": None,
        "destination": None,
        "departure_time": None,
        "tail_number": None
    }

with st.form(key="user_flight_details"):
    flight_details["origin"] = st.text_input("Enter the departure airport (abbreviated): ")
    flight_details["destination"] = st.text_input("Enter the destination airport (abbreviated): ")
    flight_details["departure_time"] = st.time_input("Enter your departure time (use military time): ")
    flight_details["tail_number"] = st.text_input("Enter the airplane tail number aka N-number: ")

    submit = st.form_submit_button("Submit")

if submit:
    if not all(flight_details.values()):
        st.warning("Please fill in all of the fields")
    else:
        # Convert departure_time (which is a time object) to string
        df = pd.DataFrame({
            "origin": [flight_details["origin"]],
            "destination": [flight_details["destination"]],
            "departure_time": [flight_details["departure_time"].strftime("%H:%M")],
            "tail_number": [flight_details["tail_number"]]
        })
        # create and encode route
        df['route'] = df['origin'] + '_' + df['destination']
        route_frequency = df['route'].value_counts()
        df['route_encoded'] = df['route'].map(route_frequency)
        df.drop(columns=['route'], inplace=True)

        # create and encode time-sin and time-cosine
        def hhmm_to_minutes(hhmm):
            hours, minutes = map(int, hhmm.split(":"))
            return hours * 60 + minutes  
        
        df['Time'] = df['departure_time'].apply(hhmm_to_minutes)
        df['time_sin'] = np.sin(2 * np.pi * df['Time'] / 1440)  # 1440 minutes in a day
        df['time_cos'] = np.cos(2 * np.pi * df['Time'] / 1440)
        
        df = df[['route_encoded', 'time_sin', 'time_cos']]
        
        # Load the trained model
        model_path = os.path.join("..", "/models/model.pkl")

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # make the predictions
    