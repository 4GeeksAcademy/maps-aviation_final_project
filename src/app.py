import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import time
import configuration as config 

import streamlit as st
import os

# Load airport data
@st.cache_data
def load_airports():
    df=pd.read_csv(config.COMBINED_DATAFILE) # load data from the combined data file
    return df[['origin', 'destination', 'departure_time', 'tail_number']] # return the relevant columns 

# Arc generator
#def generate_arc(p1, p2, num_points=100, height=10):
    #lon1, lat1 = p1
    #lon2, lat2 = p2
    #lons = np.linspace(lon1, lon2, num_points)
    #lats = np.linspace(lat1, lat2, num_points)
    #curve = np.sin(np.linspace(0, np.pi, num_points)) * height
    #curved_lats = lats + curve / 100
    #return [[lon, lat] for lon, lat in zip(lons, curved_lats)]

# Icons for origin/destination
#icon_data = {
    #"url": "https://img.icons8.com/plasticine/100/000000/marker.png",
    #"width": 128,
    #"height": 128,
    #"anchorY": 128,
#}

# App layout
st.set_page_config(page_title="✈️ Worried Your Plane might Crash?", layout="wide")
st.title("✈️ Worried Your Plane might Crash?")
st.subheader("Submit your flight details to find out.")

# Sidebar
st.sidebar.title("Flight Controls")
airports = load_airports()
origin_label = st.sidebar.selectbox("Origin Airport", airports['origin'])
dest_label = st.sidebar.selectbox("Destination Airport", airports['destination'])

origin = airports[airports['origin'] == origin_label].iloc[0]
destination = airports[airports['destination'] == dest_label].iloc[0]

# Header KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Origin", f"{origin['origin']}")
col2.metric("Destination", f"{destination['destination']}")
col3.metric("Departure Time", f"{origin['departure_time']}")
col4.metric("Tail Number", f"{origin['tail_number']}")
