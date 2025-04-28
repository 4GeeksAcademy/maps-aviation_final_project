import streamlit as st
import pandas as pd
import pydeck as pdk
import math
import numpy as np

# Load US airports dataset
@st.cache_data
def load_airports():
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    cols = ['AirportID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 
            'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz', 'Type', 'Source']
    df = pd.read_csv(url, header=None, names=cols)
    us_airports = df[(df['Country'] == 'United States') & (df['IATA'].notnull()) & (df['IATA'] != '\\N')]
    us_airports["Label"] = us_airports["City"] + " - " + us_airports["Name"] + f" (" + us_airports["IATA"] + ")"
    return us_airports[['Label', 'Name', 'City', 'IATA', 'Latitude', 'Longitude']]

# Generate arc between two points
def generate_arc(origin, destination, num_points=100):
    lat1, lon1 = origin
    lat2, lon2 = destination
    arc = []
    for i in range(num_points + 1):
        frac = i / num_points
        lat = lat1 + (lat2 - lat1) * frac
        lon = lon1 + (lon2 - lon1) * frac + math.sin(frac * math.pi) * 5
        arc.append([lon, lat])
    return arc

# Main Streamlit app
def main():
    st.set_page_config(page_title="Flight Route with Animation", layout="wide")
    st.title("✈️ USA Flight Route Mapper (Animated)")

    airports = load_airports()

    st.sidebar.header("Select Airports")
    origin_label = st.sidebar.selectbox("Origin Airport", airports['Label'])
    dest_label = st.sidebar.selectbox("Destination Airport", airports['Label'])

    origin = airports[airports['Label'] == origin_label].iloc[0]
    destination = airports[airports['Label'] == dest_label].iloc[0]

    st.write(f"**Origin:** {origin['Name']} ({origin['IATA']}) - {origin['City']}")
    st.write(f"**Destination:** {destination['Name']} ({destination['IATA']}) - {destination['City']}")

    # Generate arc
    arc = generate_arc((origin['Latitude'], origin['Longitude']), 
                       (destination['Latitude'], destination['Longitude']))

    # Simulate airplane position
    steps = len(arc)
    step = st.slider("Flight Progress", 0, steps - 1, 0)

    # Prepare data
    path_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": arc}],
        get_path="path",
        get_width=4,
        get_color=[0, 100, 255],
        width_min_pixels=2,
    )

    airplane_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"position": arc[step]}],
        get_position="position",
        get_color=[255, 0, 0],
        get_radius=10000,
    )

    view_state = pdk.ViewState(
        latitude=arc[step][1],
        longitude=arc[step][0],
        zoom=4,
        pitch=30,
    )

    r = pdk.Deck(
        layers=[path_layer, airplane_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9",
    )

    st.pydeck_chart(r)

    # Auto Animate
    animate = st.checkbox("Auto Animate Flight", value=False)
    if animate:
        for i in range(steps):
            st.experimental_rerun()

if __name__ == "__main__":
    main()