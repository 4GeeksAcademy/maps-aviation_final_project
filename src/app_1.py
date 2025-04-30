import streamlit as st
import pandas as pd
import pydeck as pdk

# Load or define your airport DataFrame
@st.cache_data
def load_airports():
    df = pd.read_csv("airports.csv")  # Replace with your actual file or model data
    df = df[df['IATA'].notnull() & (df['IATA'] != '\\N')]
    df['Label'] = df['City'] + " - " + df['Name'] + " (" + df['IATA'] + ")"
    return df[['Label', 'City', 'Name', 'IATA', 'Latitude', 'Longitude']]

airports = load_airports()

# Streamlit UI
st.set_page_config(page_title="Flight Mapper", layout="wide")
st.title("Flight Mapper Dashboard")

# Sidebar inputs
st.sidebar.header("Select Airports")
origin_label = st.sidebar.selectbox("Origin Airport", airports['Label'])
dest_label = st.sidebar.selectbox("Destination Airport", airports['Label'])

# Extract selected airport details
origin = airports[airports['Label'] == origin_label].iloc[0]
destination = airports[airports['Label'] == dest_label].iloc[0]

# Map rendering
view_state = pdk.ViewState(
    latitude=(origin['Latitude'] + destination['Latitude']) / 2,
    longitude=(origin['Longitude'] + destination['Longitude']) / 2,
    zoom=3,
    pitch=45,
)

# Create route and markers
route = [[origin['Longitude'], origin['Latitude']], [destination['Longitude'], destination['Latitude']]]

route_layer = pdk.Layer(
    "LineLayer",
    data=pd.DataFrame([{"coordinates": route}]),
    get_source_position="coordinates[0]",
    get_target_position="coordinates[1]",
    get_color=[200, 30, 0, 160],
    get_width=5,
)

marker_data = pd.DataFrame([
    {"name": "Origin", "coordinates": [origin['Longitude'], origin['Latitude']]},
    {"name": "Destination", "coordinates": [destination['Longitude'], destination['Latitude']]}
])

marker_layer = pdk.Layer(
    "ScatterplotLayer",
    data=marker_data,
    get_position="coordinates",
    get_color=[0, 0, 255],
    get_radius=10000,
    pickable=True
)

tooltip = {
    "html": "<b>{name}</b>",
    "style": {"backgroundColor": "black", "color": "white"}
}

st.pydeck_chart(pdk.Deck(
    layers=[route_layer, marker_layer],
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style="mapbox://styles/mapbox/light-v9"
))
