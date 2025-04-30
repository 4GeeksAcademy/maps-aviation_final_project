import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import time

# Load airport data
@st.cache_data
def load_airports():
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    cols = ['AirportID', 'Name', 'City', 'Country', 'IATA', 'ICAO',
            'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz', 'Type', 'Source']
    df = pd.read_csv(url, header=None, names=cols)
    us_airports = df[(df['Country'] == 'United States') & (df['IATA'].notnull()) & (df['IATA'] != '\\N')]
    us_airports["Label"] = us_airports["City"] + " - " + us_airports["Name"] + " (" + us_airports["IATA"] + ")"
    return us_airports[['Label', 'Name', 'City', 'IATA', 'Latitude', 'Longitude']]

# Generate a curved arc path
def generate_arc(p1, p2, num_points=100, height=10):
    lon1, lat1 = p1
    lon2, lat2 = p2
    lons = np.linspace(lon1, lon2, num_points)
    lats = np.linspace(lat1, lat2, num_points)
    curve = np.sin(np.linspace(0, np.pi, num_points)) * height
    curved_lats = lats + curve / 100
    return [[lon, lat] for lon, lat in zip(lons, curved_lats)]

# Marker icon
icon_marker = {
    "url": "https://img.icons8.com/plasticine/100/000000/marker.png",
    "width": 128,
    "height": 128,
    "anchorY": 128
}

# Plane icon
icon_plane = {
    "url": "https://img.icons8.com/color/48/airplane-take-off.png",
    "width": 64,
    "height": 64,
    "anchorY": 32
}

# App config
st.set_page_config(page_title="✈️ Flight Dashboard", layout="wide")
st.title("✈️ Flight Dashboard")

# Sidebar inputs
airports = load_airports()
st.sidebar.title("Flight Controls")
origin_label = st.sidebar.selectbox("Origin Airport", airports['Label'])
dest_label = st.sidebar.selectbox("Destination Airport", airports['Label'])

origin = airports[airports['Label'] == origin_label].iloc[0]
destination = airports[airports['Label'] == dest_label].iloc[0]

# Top KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Origin", f"{origin['City']} ({origin['IATA']})")
col2.metric("Destination", f"{destination['City']} ({destination['IATA']})")
col3.metric("Distance (approx)", f"{np.round(np.linalg.norm(np.array([origin['Latitude'], origin['Longitude']]) - np.array([destination['Latitude'], destination['Longitude']])), 2)}°")

# Tabs
tab1, tab2 = st.tabs(["🌍 Map View", "🛫 Flight Animation"])

# View state
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

# Data for origin/destination icon layer
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

# Curved route layer
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

# Tab 1: Static Map View
with tab1:
    st.pydeck_chart(pdk.Deck(
        layers=[icon_layer, curved_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9",
        tooltip=tooltip
    ))

# Tab 2: Animation
with tab2:
    chart_placeholder = st.empty()
    start = st.button("🛫 Start Flight")

    if start:
        for i in range(len(curved_path)):
            plane_position = curved_path[i]
            trail = curved_path[max(0, i - 5):i + 1]

            # Moving plane icon
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
                layers=[icon_layer, curved_layer, trail_layer, plane_layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/light-v9",
                tooltip=tooltip
            )

            chart_placeholder.pydeck_chart(r)
            time.sleep(0.05)
