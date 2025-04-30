import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import time
import requests

# ---------------------------
# WEATHER FUNCTION (Open-Meteo)
# ---------------------------
def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        response = requests.get(url)
        data = response.json()
        temp = data["current_weather"]["temperature"]
        code = data["current_weather"]["weathercode"]
        condition = weather_code_to_description(code)
        return f"{temp}°C, {condition}"
    except:
        return "Weather Unavailable"

def weather_code_to_description(code):
    weather_descriptions = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
        55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Light freezing rain", 67: "Heavy freezing rain",
        71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
        77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
    }
    return weather_descriptions.get(code, "Unknown")

# ---------------------------
# LOAD AIRPORT DATA
# ---------------------------
@st.cache_data
def load_airports():
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    cols = ['AirportID', 'Name', 'City', 'Country', 'IATA', 'ICAO',
            'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz', 'Type', 'Source']
    df = pd.read_csv(url, header=None, names=cols)
    us_airports = df[(df['Country'] == 'United States') & (df['IATA'].notnull()) & (df['IATA'] != '\\N')]
    us_airports["Label"] = us_airports["City"] + " - " + us_airports["Name"] + " (" + us_airports["IATA"] + ")"
    return us_airports[['Label', 'Name', 'City', 'IATA', 'Latitude', 'Longitude']]

# ---------------------------
# CURVED PATH GENERATOR
# ---------------------------
def generate_arc(p1, p2, num_points=100, height=10):
    lon1, lat1 = p1
    lon2, lat2 = p2
    lons = np.linspace(lon1, lon2, num_points)
    lats = np.linspace(lat1, lat2, num_points)
    curve = np.sin(np.linspace(0, np.pi, num_points)) * height
    curved_lats = lats + curve / 100
    return [[lon, lat] for lon, lat in zip(lons, curved_lats)]

# ---------------------------
# ICON DEFINITIONS
# ---------------------------
icon_marker = {
    "url": "https://img.icons8.com/plasticine/100/000000/marker.png",
    "width": 128,
    "height": 128,
    "anchorY": 128
}

icon_plane = {
    "url": "https://img.icons8.com/color/48/airplane-take-off.png",
    "width": 64,
    "height": 64,
    "anchorY": 32
}

# ---------------------------
# MAIN APP
# ---------------------------
st.set_page_config(page_title="✈️ Flight Dashboard", layout="wide")
st.title("✈️ Flight Dashboard")

# Sidebar selection
airports = load_airports()
st.sidebar.title("Flight Controls")
origin_label = st.sidebar.selectbox("Origin Airport", airports['Label'])
dest_label = st.sidebar.selectbox("Destination Airport", airports['Label'])

origin = airports[airports['Label'] == origin_label].iloc[0]
destination = airports[airports['Label'] == dest_label].iloc[0]

# Weather using Open-Meteo (no API key needed)
origin_weather = get_weather(origin['Latitude'], origin['Longitude'])
dest_weather = get_weather(destination['Latitude'], destination['Longitude'])

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Origin", f"{origin['City']} ({origin['IATA']})", origin_weather)
col2.metric("Destination", f"{destination['City']} ({destination['IATA']})", dest_weather)
distance = np.round(np.linalg.norm(np.array([origin['Latitude'], origin['Longitude']]) - np.array([destination['Latitude'], destination['Longitude']])), 2)
col3.metric("Distance (approx)", f"{distance}°")

# View
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

# Icon data for map
icon_layer_data = pd.DataFrame([
    {
        "name": f"{origin['City']} ({origin['IATA']})",
        "coordinates": [origin['Longitude'], origin['Latitude']],
        "icon_data": icon_marker
    },
    {
        "name": f"{destination['City']} ({destination['IATA']})",
        "coordinates": [destination['Longitude'], destination['Latitude']],
        "icon_data": icon_marker
    }
])

icon_layer = pdk.Layer(
    "IconLayer",
    data=icon_layer_data,
    get_icon="icon_data",
    get_size=4,
    size_scale=15,
    get_position="coordinates",
    pickable=True,
)

# Flight path (trail)
curved_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": curved_path}],
    get_path="path",
    get_color=[255, 100, 100],
    width_scale=20,
    width_min_pixels=3,
    get_width=5,
)

tooltip = {
    "html": "<b>Airport:</b> {name}",
    "style": {"backgroundColor": "black", "color": "white"}
}

# Tabs
tab1, tab2 = st.tabs(["🌍 Map View", "🛫 Flight Animation"])

with tab1:
    st.pydeck_chart(pdk.Deck(
        layers=[icon_layer, curved_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9",
        tooltip=tooltip
    ))

with tab2:
    chart_placeholder = st.empty()
    if st.button("🛫 Start Flight"):
        for i in range(len(curved_path)):
            plane_position = curved_path[i]
            trail = curved_path[max(0, i - 5):i + 1]

            plane_data = pd.DataFrame([{
                "coordinates": plane_position,
                "icon_data": icon_plane,
                "name": "Plane"
            }])

            plane_layer = pdk.Layer(
                "IconLayer",
                data=plane_data,
                get_icon="icon_data",
                get_size=4,
                size_scale=10,
                get_position="coordinates",
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

            r = pdk.Deck(
                layers=[icon_layer, trail_layer, plane_layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/light-v9",
                tooltip=tooltip
            )

            chart_placeholder.pydeck_chart(r)
            time.sleep(0.05)
