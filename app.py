import streamlit as st
import os

st.title("Worried Your Plane might Crash?")
st.subheader("Submit your flight details to find out.")
st.divider()

st.image(os.path.join(os.getcwd(), "static", "photo.jpg"))

with st.form(key="user_flight_details"):
    departure_airport = st.text_input("Enter the departure airport (abbreviated): ")
    arrival_airport = st.text_input("Enter the destination airport (abbreviated): ")
    departure_time = st.time_input("Enter your departure time (use military time): ")
    tail_number = st.text_input("Enter the airplane tail number aka N-number: ")

    st.form_submit_button()
