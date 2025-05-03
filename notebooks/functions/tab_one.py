'''Functions for streamlit web application.'''

import streamlit as st
import pandas as pd

def tab1(airports:pd.DataFrame, model:callable, frequencies:dict):
    '''Defines incident prediction tab.'''

    st.header("Predict Flight Incident Risk")
    col1, col2 = st.columns(2)
    with col1:
        origin_label = st.selectbox("Origin Airport", airports['Label'])
        dest_label = st.selectbox("Destination Airport", airports['Label'])
    with col2:
        departure_time = st.number_input("Departure Time (e.g., 1430 for 2:30 PM)", min_value=0, max_value=2359, step=5)

    if st.button("Predict Incident Probability"):
            st.warning("Prediction logic not implemented.")

    return origin_label, dest_label
