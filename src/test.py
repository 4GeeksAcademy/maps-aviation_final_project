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

# Create a curved arc between two points
def generate_arc(p1, p2, num_points=100, height=10):
    lon1, lat1 = p1
    lon2, lat2 = p2
    lons = np.linspace(lon1, lon2, num_points)
    lats = np.linspace(lat1, lat2, num_points)
    curve = np.sin(np.linspace(0, np.pi, num_points)) * height
    curved_lats = lats + curve / 100
    curved_path = [[lon, lat] for lon, lat in zip(lons, curved_lats)]
    return curved_path

# Calculate bearing (angle) between two points
def calculate_bearing(p1, p2):
    lon1, lat1 = np.radians(p1)
    lon2, lat2 = np.radians(p2)
    d_lon = lon2 - lon1
    x = np.sin(d_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(d_lon))
    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360
    return bearing

# Main App
def main():
    st.set_page_config(page_title="Flight Route Animation", layout="wide")
    st.title("🛩️ Animated Flight Along Curved Route Between US Airports")

    airports = load_airports()

    st.sidebar.header("Flight Settings")
    origin_label = st.sidebar.selectbox("Origin Airport", airports['Label'])
    dest_label = st.sidebar.selectbox("Destination Airport", airports['Label'])

    origin = airports[airports['Label'] == origin_label].iloc[0]
    destination = airports[airports['Label'] == dest_label].iloc[0]

    curved_path = generate_arc(
        (origin['Longitude'], origin['Latitude']),
        (destination['Longitude'], destination['Latitude']),
        num_points=100,
        height=8
    )

    view_state = pdk.ViewState(
        latitude=(origin['Latitude'] + destination['Latitude']) / 2,
        longitude=(origin['Longitude'] + destination['Longitude']) / 2,
        zoom=3,
        pitch=30,
    )

    # Static path layer
    path_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": curved_path}],
        get_path="path",
        get_color=[255, 100, 100],
        width_scale=20,
        width_min_pixels=3,
        get_width=5,
        pickable=True,
    )

    # Icon for airports
    airport_icons = [
        {
            "position": [origin['Longitude'], origin['Latitude']],
            "icon": {
                "url": "https://img.icons8.com/emoji/48/airplane-departure.png",
                "width": 48,
                "height": 48,
                "anchorY": 48
            }
        },
        {
            "position": [destination['Longitude'], destination['Latitude']],
            "icon": {
                "url": "https://img.icons8.com/emoji/48/airplane-arrival.png",
                "width": 48,
                "height": 48,
                "anchorY": 48
            }
        }
    ]

    airports_data = pd.DataFrame([
        {"position": [origin['Longitude'], origin['Latitude']], "icon": airport_icon},
        {"position": [destination['Longitude'], destination['Latitude']], "icon": airport_icon},
    ])

    airport_layer = pdk.Layer(
        "IconLayer",
        data=airports_data,
        get_icon="icon",
        get_size=4,
        size_scale=10,
        get_position="position",
        pickable=True,
    )

    # Placeholder for animation
    chart_placeholder = st.empty()

    trail_length = 5

    # Start / Restart buttons
    if st.button("✈️ Start Flight"):
        i = 0
        while i < len(curved_path):
            plane_position = curved_path[i]

            if i < len(curved_path) - 1:
                next_position = curved_path[i + 1]
                angle = calculate_bearing(plane_position, next_position)
            else:
                angle = 0

            if i >= trail_length:
                trail = curved_path[i-trail_length:i]
            else:
                trail = curved_path[:i+1]

            plane_icon = [
                {
                    "position": plane_position,
                    "angle": angle,
                    "icon": {
                        "url": "https://img.icons8.com/emoji/48/small-airplane.png",
                        "width": 48,
                        "height": 48,
                        "anchorY": 24
                    }
                }
            ]

            plane_layer = pdk.Layer(
                "IconLayer",
                data=plane_icon,
                get_icon="icon",
                get_size=5,
                size_scale=15,
                get_position="position",
                get_angle="angle",
                pickable=False,
            )

            trail_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": trail}],
                get_path="path",
                get_color=[100, 100, 255],
                width_scale=20,
                width_min_pixels=2,
                get_width=3,
            )

            r = pdk.Deck(
                layers=[path_layer, trail_layer, airport_layer, plane_layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/light-v9",
            )

            chart_placeholder.pydeck_chart(r)
            time.sleep(0.05)

            i += 1

    if st.button("🔄 Restart Flight"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
