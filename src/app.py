import configuration as config 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.title("Worried Your Plane might Crash?")
st.subheader("Submit your flight details to find out.")
st.divider()

st.image(os.path.join(os.path.dirname(__file__), "..", "static", "photo.jpg"))

# load dataset
data_path = os.path.join(os.getcwd(), "data", "processed", "combined_data.csv")
data_df = pd.read_csv(data_path)

flight_details = {
        "origin": None,
        "destination": None,
        "departure_time": None,
    }

with st.form(key="user_flight_details"):
    flight_details["origin"] = st.selectbox(label="Enter the departure airport: ", options=data_df['origin'].unique(), placeholder='LGA')
    flight_details["destination"] = st.selectbox(label="Enter the destination airport: ", options=data_df['destination'].unique(), placeholder='ORF')
    flight_details["departure_time"] = st.time_input("Enter your departure time (use military time): ")

    submit = st.form_submit_button("Submit")

if submit:
    if not all(flight_details.values()):
        st.warning("Please fill in all of the fields")
    else:
        # Convert departure_time (which is a time object) to string
        df = pd.DataFrame({
            "origin": [flight_details["origin"]],
            "destination": [flight_details["destination"]],
            "departure_time": [flight_details["departure_time"].strftime("%H:%M")]
        })
        # create and encode route
        route_frequency = data_df['origin'] + '_' + data_df['destination']
        route_frequency = route_frequency.value_counts().to_dict()
        df['route'] = df['origin'] + '_' + df['destination']
        df['route_encoded'] = df['route'].map(route_frequency)
        df['route_encoded'].fillna(0, inplace=True)
        df.drop(columns=['route'], inplace=True)
        print(df)

        # create and encode time-sin and time-cosine
        def hhmm_to_minutes(hhmm):
            hours, minutes = map(int, hhmm.split(":"))
            return hours * 60 + minutes  
        
        df['Time'] = df['departure_time'].apply(hhmm_to_minutes)
        df['time_sin'] = np.sin(2 * np.pi * df['Time'] / 1440)  # 1440 minutes in a day
        df['time_cos'] = np.cos(2 * np.pi * df['Time'] / 1440)
        
        df = df[['time_sin', 'time_cos', 'route_encoded']]
        
        # Load the trained model
        model_path = os.path.join(os.getcwd(), "models", "model.pkl")


        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # make the predictions
        probability = model.predict_proba(df)
        percent_probability = probability[:, 1] * 100
        print(percent_probability)

        # Display predictions
        st.write(f"The probability of your plane crashing is {percent_probability.item():.2f}%")