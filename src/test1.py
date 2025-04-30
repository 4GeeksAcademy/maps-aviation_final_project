import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import time

# Load airports
@st.cache_data
def load_airports():
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    cols = ['AirportID', 'Name', 'City', 'Country', 'IATA', 'ICAO',
            'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz', 'Type', 'Source']
    df = pd.read_csv(url, header=None, names=cols)
    us_airports = df[(df['Country'] == 'United States') & (df['IATA'].notnull()) & (df['IATA'] != '\\N')]
    us_airports["Label"] = us_airports["City"] + " - " + us_airports["Name"] + " (" + us_airports["IATA"] + ")"
    return us_airports[['Label', 'Name', 'City', 'IATA', 'Latitude', 'Longitude']]

# Create curved arc
def generate_arc(p1, p2, num_points=100, height=10):
    lon1, lat1 = p1
    lon2, lat2 = p2
    lons = np.linspace(lon1, lon2, num_points)
    lats = np.linspace(lat1, lat2, num_points)
    curve = np.sin(np.linspace(0, np.pi, num_points)) * height
    curved_lats = lats + curve / 100
    return [[lon, lat] for lon, lat in zip(lons, curved_lats)]

# App layout
st.set_page_config(page_title="✈️ Flight Tracker Dashboard", layout="wide")
st.title("✈️ Flight Dashboard")

# Sidebar
st.sidebar.title("Flight Controls")
airports = load_airports()
origin_label = st.sidebar.selectbox("Origin Airport", airports['Label'])
dest_label = st.sidebar.selectbox("Destination Airport", airports['Label'])

origin = airports[airports['Label'] == origin_label].iloc[0]
destination = airports[airports['Label'] == dest_label].iloc[0]

# Header KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Origin", f"{origin['City']} ({origin['IATA']})")
col2.metric("Destination", f"{destination['City']} ({destination['IATA']})")
col3.metric("Lat/Lon Distance", f"{np.round(np.linalg.norm(np.array([origin['Latitude'], origin['Longitude']]) - np.array([destination['Latitude'], destination['Longitude']])), 2)}°")

# Tabs
tab1, tab2 = st.tabs(["🌍 Map View", "🛫 Flight Animation"])

# Static route (straight line)
solid_route = [[origin['Longitude'], origin['Latitude']],
               [destination['Longitude'], destination['Latitude']]]

#solid_route_layer = pdk.Layer(
    #"PathLayer",
    #data=[{"path": solid_route}],
    #get_path="path",
    #get_color=[0, 0, 0],
    #width_scale=20,
   # width_min_pixels=2,
    #get_width=3,
#)

# View settings
view_state = pdk.ViewState(
    latitude=(origin['Latitude'] + destination['Latitude']) / 2,
    longitude=(origin['Longitude'] + destination['Longitude']) / 2,
    zoom=3,
    pitch=0,
)

# Curved path
curved_path = generate_arc(
    (origin['Longitude'], origin['Latitude']),
    (destination['Longitude'], destination['Latitude']),
    num_points=100,
    height=8
)

# Tab 1: Static Map View
with tab1:
    airport_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[
            {"position": [origin['Longitude'], origin['Latitude']]},
            {"position": [destination['Longitude'], destination['Latitude']]}
        ],
        get_position="position",
        get_color=[0, 0, 200],
        get_radius=6000,
    )
    path_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": curved_path}],
        get_path="path",
        get_color=[255, 100, 100],
        width_scale=20,
        width_min_pixels=3,
        get_width=5,
    )
    st.pydeck_chart(pdk.Deck(
        layers=[airport_layer, path_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9"
    ))

# Tab 2: Flight Animation
with tab2:
    chart_placeholder = st.empty()
    start = st.button("🛫 Start Flight")
    if start:
        trail_length = 5
        for i in range(len(curved_path)):
            plane_position = curved_path[i]
            trail = curved_path[max(0, i - trail_length):i + 1]

            plane_layer = pdk.Layer(
                "TextLayer",
                data=[{"position": plane_position, "text": "✈️"}],
                get_position="position",
                get_text="text",
                get_size=32,
                get_angle=0,
                get_color=[0, 0, 0],
            )
            trail_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": trail}],
                get_path="path",
                get_color=[50, 50, 255],
                width_scale=20,
                width_min_pixels=2,
                get_width=3,
            )
            airport_layer = pdk.Layer(
                "ScatterplotLayer",
                data=[
                    {"position": [origin['Longitude'], origin['Latitude']]},
                    {"position": [destination['Longitude'], destination['Latitude']]}
                ],
                get_position="position",
                get_color=[0, 0, 200],
                get_radius=6000,
            )
            base_path_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": curved_path}],
                get_path="path",
                get_color=[255, 100, 100],
                width_scale=20,
                width_min_pixels=3,
                get_width=5,
            )
            r = pdk.Deck(
                layers=[airport_layer, base_path_layer, trail_layer, plane_layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/light-v9",
            )
            chart_placeholder.pydeck_chart(r)
            time.sleep(0.05)
