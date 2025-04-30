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

# Main App
def main():
    st.set_page_config(page_title="Flight Route Animation", layout="wide")
    st.title("🛩️ Animated Flight Along Curved Route Between US Airports")

    airports = load_airports()  
    print(airports.head())  # Debugging line to check the loaded airports
    st.sidebar.header("Flight Settings")
    origin_label = st.sidebar.selectbox("Origin Airport", airports['Label'])
    dest_label = st.sidebar.selectbox("Destination Airport", airports['Label'])

    origin = airports[airports['Label'] == origin_label].iloc[0]
    destination = airports[airports['Label'] == dest_label].iloc[0]

    # Generate curve path
    curved_path = generate_arc(
        (origin['Longitude'], origin['Latitude']),
        (destination['Longitude'], destination['Latitude']),
        num_points=100,
        height=8
    )

    # View
    view_state = pdk.ViewState(
        latitude=(origin['Latitude'] + destination['Latitude']) / 2,
        longitude=(origin['Longitude'] + destination['Longitude']) / 2,
        zoom=3,
        pitch=0,
    )

    # Path Layer
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

    # Airport Icons (Static)
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

    airport_layer = pdk.Layer(
        "IconLayer",
        data=airport_icons,
        get_icon="icon",
        get_size=4,
        size_scale=15,
        get_position="position",
        pickable=True,
    )

    # Placeholder for animation
    chart_placeholder = st.empty()

    # Animation
    trail_length = 5

    for i in range(len(curved_path)):
        plane_position = curved_path[i]

        # Create trail behind plane
        if i >= trail_length:
            trail = curved_path[i-trail_length:i]
        else:
            trail = curved_path[:i+1]

        # Moving Plane Icon
        plane_icon = [
            {
                "position": plane_position,
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
            get_size=4,
            size_scale=15,
            get_position="position",
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

if __name__ == "__main__":
    main()

