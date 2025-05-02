import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import time
from PIL import Image

import functions.tab_one as tab_one_func

# ---------------------------
# PAGE CONFIG & BANNER IMAGE
# ---------------------------
st.set_page_config(page_title="✈️ Flight Incident App", layout="wide")
image = Image.open("src/static/photo.jpg")

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
# ARCS & MAP HELPERS
# ---------------------------
def generate_arc(p1, p2, num_points=100, height=10):
    lon1, lat1 = p1
    lon2, lat2 = p2
    lons = np.linspace(lon1, lon2, num_points)
    lats = np.linspace(lat1, lat2, num_points)
    curve = np.sin(np.linspace(0, np.pi, num_points)) * height
    curved_lats = lats + curve / 100
    return [[lon, lat] for lon, lat in zip(lons, curved_lats)]

icon_data = {
    "url": "https://img.icons8.com/plasticine/100/000000/marker.png",
    "width": 128,
    "height": 128,
    "anchorY": 128,
}

# ---------------------------
# MAIN APP
# ---------------------------
def main():
    st.title("✈️ Flight Incident Predictor & Dashboard")
    st.image(image, width=200)
    st.markdown("""
    This app predicts the likelihood of a flight incident and visualizes the flight path between selected airports.
    """)

    # Load airports data
    airports = load_airports()

    # UI: Select Airports & Departure Time
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Incident Predictor", "Model Performance", "Data Exploration", "🌍 Map View", "🛫 Flight Animation"
    ])

    with tab1:
        origin_label, dest_label = tab_one_func.tab1(airports)

    with tab2:
        st.header("Model Performance Metrics")
        st.info("Model evaluation metrics and visualizations will appear here.")

    with tab3:
        st.header("Data Exploration")
        st.info("Raw data views, visualizations, and correlation plots will be added here.")

    origin = airports[airports['Label'] == origin_label].iloc[0]
    destination = airports[airports['Label'] == dest_label].iloc[0]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Origin", f"{origin['City']} ({origin['IATA']})")
    col2.metric("Destination", f"{destination['City']} ({destination['IATA']})")
    distance = np.round(np.linalg.norm(np.array([origin['Latitude'], origin['Longitude']]) -
                                       np.array([destination['Latitude'], destination['Longitude']])), 2)
    col3.metric("Lat/Lon Distance", f"{distance}°")

    # Shared map data
    curved_path = generate_arc(
        (origin['Longitude'], origin['Latitude']),
        (destination['Longitude'], destination['Latitude']),
        num_points=100,
        height=8
    )
    icon_layer_data = pd.DataFrame([
        {
            "name": f"{origin['City']} ({origin['IATA']})",
            "coordinates": [origin['Longitude'], origin['Latitude']],
            "icon_data": icon_data
        },
        {
            "name": f"{destination['City']} ({destination['IATA']})",
            "coordinates": [destination['Longitude'], destination['Latitude']],
            "icon_data": icon_data
        }
    ])
    view_state = pdk.ViewState(
        latitude=(origin['Latitude'] + destination['Latitude']) / 2,
        longitude=(origin['Longitude'] + destination['Longitude']) / 2,
        zoom=3,
        pitch=0,
    )
    tooltip = {
        "html": "<b>Airport:</b> {name}",
        "style": {"backgroundColor": "black", "color": "white"}
    }
    icon_layer = pdk.Layer(
        type='IconLayer',
        data=icon_layer_data,
        get_icon='icon_data',
        get_size=4,
        size_scale=15,
        get_position='coordinates',
        pickable=True,
    )

    with tab4:
        curved_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": curved_path}],
            get_path="path",
            get_color=[255, 100, 100],
            width_scale=20,
            width_min_pixels=3,
            get_width=5,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[icon_layer, curved_layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v9",
            tooltip=tooltip
        ))

    with tab5:
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
                curved_layer = pdk.Layer(
                    "PathLayer",
                    data=[{"path": curved_path}],
                    get_path="path",
                    get_color=[255, 100, 100],
                    width_scale=20,
                    width_min_pixels=3,
                    get_width=5,
                )
                r = pdk.Deck(
                    layers=[icon_layer, curved_layer, trail_layer, plane_layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/light-v9",
                    tooltip=tooltip
                )
                chart_placeholder.pydeck_chart(r)
                time.sleep(0.05)

if __name__ == "__main__":
    main()