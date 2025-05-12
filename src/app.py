"""Flight Incident Predictor: Streamlit app for predicting flight incidents."""

# Import necessary libraries for configuration, data processing, visualization, and model loading
import os
import json
import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk
from PIL import Image
import configuration as config

# ---------------------------
# DIRECTORY CONFIGURATION
# ---------------------------
# Get the base directory of the application
base_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# PAGE CONFIGURATION & BANNER IMAGE
# ---------------------------
# Set Streamlit page configuration
st.set_page_config(
    page_title="Flight Incident Predictor",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------
# LOAD MODEL AND DATA
# ---------------------------
# Define the path to the trained model
model_path = os.path.join(os.path.dirname(base_dir), "models", "model.pkl")

# Check if the model file exists, otherwise stop execution
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Load the trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")

# Define paths for static data files
route_frequency_path = os.path.join(base_dir, "static", "route_frequency.json")
destination_path = os.path.join(base_dir, "static", "unique_destination.json")
origin_path = os.path.join(base_dir, "static", "unique_origin.json")

# Load static data files
with open(origin_path, 'r') as f:
    unique_origin = json.load(f)

with open(destination_path, 'r') as f:
    unique_destination = json.load(f)

with open(route_frequency_path, 'r') as f:
    route_frequency = json.load(f)

# ---------------------------
# LOAD AIRPORT DATA
# ---------------------------
@st.cache_data
def load_airports() -> pd.DataFrame:
    """
    Load airport data from an external source and filter for US airports.

    Returns:
    - pd.DataFrame: Filtered DataFrame containing airport details.
    """
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    cols = [
        'AirportID', 'Name', 'City', 'Country', 'IATA', 'ICAO',
        'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz', 'Type', 'Source'
    ]
    
    # Read airport data from the URL
    df = pd.read_csv(url, header=None, names=cols)

    # Filter for US airports with valid IATA codes
    us_airports = df[(df['Country'] == 'United States') & (df['IATA'].notnull()) & (df['IATA'] != '\\N')]

    # Create a label for display
    us_airports["Label"] = us_airports["City"] + " - " + us_airports["Name"] + " (" + us_airports["IATA"] + ")"

    return us_airports[['Label', 'Name', 'City', 'IATA', 'Latitude', 'Longitude']]

# ---------------------------
# FUNCTIONS
# ---------------------------
def check_route(df: pd.DataFrame, route_frequency: dict) -> pd.DataFrame:
    """
    Checks if a flight route exists in the frequency dataset and encodes it.

    Parameters:
    - df (pd.DataFrame): DataFrame containing flight details.
    - route_frequency (dict): Dictionary mapping routes to frequency values.

    Returns:
    - pd.DataFrame: Updated DataFrame with encoded route information.
    """
    
    # Create and encode the flight route
    df['route'] = df['origin'] + '_' + df['destination']

    if df['route'].iloc[0] not in route_frequency:
        return None

    df['route_encoded'] = df['route'].map(route_frequency)
    df['route_encoded'].fillna(0, inplace=True)
    df.drop(columns=['route'], inplace=True)

    return df


def hhmm_to_minutes(hhmm: str) -> int:
    """
    Converts HH:MM time format to total minutes.

    Parameters:
    - hhmm (str): Time in HH:MM format.

    Returns:
    - int: Total minutes.
    """
    hours, minutes = map(int, hhmm.split(":"))
    return hours * 60 + minutes  


def generate_arc(p1: tuple, p2: tuple, num_points: int = 100, height: int = 10) -> list:
    """
    Generates a curved flight path between two points.

    Parameters:
    - p1 (tuple): Starting point (longitude, latitude).
    - p2 (tuple): Ending point (longitude, latitude).
    - num_points (int, optional): Number of points in the arc (default: 100).
    - height (int, optional): Maximum curve height (default: 10).

    Returns:
    - list: List of coordinates representing the curved flight path.
    """
    lon1, lat1 = p1
    lon2, lat2 = p2

    # Generate linear interpolation for longitude and latitude
    lons = np.linspace(lon1, lon2, num_points)
    lats = np.linspace(lat1, lat2, num_points)

    # Apply sinusoidal curve effect
    curve = np.sin(np.linspace(0, np.pi, num_points)) * height
    curved_lats = lats + curve / 100

    return [[lon, lat] for lon, lat in zip(lons, curved_lats)]


# Define icon data for map visualization
icon_data = {
    "url": "https://img.icons8.com/plasticine/100/000000/marker.png",
    "width": 128,
    "height": 128,
    "anchorY": 128,
}
# ---------------------------
# MAIN APP FUNCTION
# ---------------------------
def main():
    """
    Main function to render the Streamlit app UI and handle user input for flight incident prediction.
    """
    
    # Create three columns for parallel display
    col1, col2, col3 = st.columns([1, 2, 1])

    # Display GIFs in respective columns
    with col1:
        st.image(os.path.join(os.path.dirname(__file__), "static", "take_off_1.gif"), use_container_width=True)

    # Set title and subtitle
    with col2:
        # Add custom CSS for styling
        st.markdown("""
            <style>
            .centered-title {
                text-align: center;
                font-size: 2.5rem;
                font-weight: bold;
                color: #1E88E5;
                margin-bottom: 1rem;
                padding: 1rem;
                border-bottom: 2px solid #1E88E5;
            }
            </style>
        """, unsafe_allow_html=True)

        # Use the styled title
        st.markdown('<h1 class="centered-title">Flight Incident Risk Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="centered-text">This app predicts the likelihood of a flight incident based on origin, destination, and departure time information!</p>', unsafe_allow_html=True)

    # Display GIFs in respective columns
    with col3:
        st.image(os.path.join(os.path.dirname(__file__), "static", "crash_1.gif"), use_container_width=True)

    # Add a divider for better UI separation
    st.divider()

    # ---------------------------
    # TABS
    # ---------------------------
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Incident Predictor", "📊 Model Performance", "📈 Data Exploration", "✈️ Flight Route Animation"
    ])

    # TAB 1: Incident Predictor UI Skeleton
    with tab1:
        # Custom CSS for tab styling
        st.markdown("""
            <style>
            .sub-header {
                font-size: 1.2rem;
                font-weight: bold;
                color: #1E88E5;
                margin-top: 0.5rem;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<h2 class="sub-header">Please choose from the options below for a prediction!</h2>', unsafe_allow_html=True)

        # Flight details dictionary
        flight_details = {
            "origin": None,
            "destination": None,
            "departure_time": None,
        }

        # Create a form for user input
        with st.form(key="user_flight_details"):
            # Create three columns for parallel display
            col1, col2, col3 = st.columns(3)

            with col1:
                flight_details["origin"] = st.selectbox(
                    label="Enter the departure airport:", options=unique_origin, placeholder='LGA'
                )

            with col2:
                flight_details["destination"] = st.selectbox(
                    label="Enter the destination airport:", options=unique_destination, placeholder='ORF'
                )

            with col3:
                flight_details["departure_time"] = st.time_input("Enter your departure time (use military time):")

            # Submit button
            submit = st.form_submit_button("Submit")

        # Process the form submission
        if submit:
            if not all(flight_details.values()):
                st.warning("Please fill in all of the fields")
            else:
                # Convert departure_time (which is a time object) to string
                df = pd.DataFrame({
                    "origin": [flight_details["origin"]],
                    "destination": [flight_details["destination"]],
                    "departure_time": [flight_details["departure_time"].strftime("%H:%M")]
                })

                # Check if the route is valid
                df = check_route(df, route_frequency)

                if df is None:
                    st.error("Invalid Flight Route. Please check your origin and destination.")
                else:
                    # Encode time features
                    df['Time'] = df['departure_time'].apply(hhmm_to_minutes)
                    df['time_sin'] = np.sin(2 * np.pi * df['Time'] / 1440)  # 1440 minutes in a day
                    df['time_cos'] = np.cos(2 * np.pi * df['Time'] / 1440)

                    # Select relevant features for prediction
                    df = df[['time_sin', 'time_cos', 'route_encoded']]

                    # Make the predictions
                    probability = model.predict_proba(df)
                    percent_probability = probability[:, 1] * 100

                    # Display predictions
                    st.write(f"The probability of your plane crashing is {percent_probability.item():.2f}%")

                    # Gauge chart visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=percent_probability.item(),
                        title={'text': "Flight Incident Risk (%)", 'font': {'size': 30}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen", 'name': "Low Risk"},
                                {'range': [25, 50], 'color': "yellow", 'name': "Medium Risk"},
                                {'range': [50, 75], 'color': "orange", 'name': "High Risk"},
                                {'range': [75, 100], 'color': "red", 'name': "Very High Risk"},
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 6},
                                'thickness': 1,
                                'value': percent_probability.item()
                            }
                        },
                    ))

                    # Display the gauge chart
                    st.plotly_chart(fig, use_container_width=True)
    # ---------------------------
    # TAB 2: Model Performance UI
    # ---------------------------
    with tab2:
        st.header("Model Performance Visualizations")

        # Create two columns for parallel display
        col1, col2 = st.columns(2)

        # Display ROC curve and feature importance images
        with col1:
            st.subheader("ROC Curve")
            st.image(os.path.join(os.path.dirname(__file__), "static", "ROC.png"), use_container_width=True)

        with col2:
            st.subheader("Feature Importance")
            st.image(os.path.join(os.path.dirname(__file__), "static", "feature_importance.png"), use_container_width=True)

        st.header("Probability Plots")

        # Create two columns for probability plots
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Calibrated Model")
            st.image(os.path.join(os.path.dirname(__file__), "static", "prob_plot_calib.png"), use_container_width=True)

        with col2:
            st.subheader("Optimized Model")
            st.image(os.path.join(os.path.dirname(__file__), "static", "prob_plot_optimized.png"), use_container_width=True)

        st.header("Calibration Plots")

        # Create two columns for calibration plots
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Base Model")
            st.image(os.path.join(os.path.dirname(__file__), "static", "calib_plot_base.png"), use_container_width=True)

        with col2:
            st.subheader("Optimized Model")
            st.image(os.path.join(os.path.dirname(__file__), "static", "calib_plot_calib.png"), use_container_width=True)

        st.header("Confusion Matrix")

        # Create three columns for confusion matrix plots
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Base Model")
            st.image(os.path.join(os.path.dirname(__file__), "static", "test_confusion.png"), use_container_width=True)

        with col2:
            st.subheader("Calibrated Model")
            st.image(os.path.join(os.path.dirname(__file__), "static", "test_confusion_calibrated.png"), use_container_width=True)

        with col3:
            st.subheader("Optimized Model")
            st.image(os.path.join(os.path.dirname(__file__), "static", "test_confusion_optimized.png"), use_container_width=True)

    # ---------------------------
    # TAB 3: Data Exploration UI
    # ---------------------------
    with tab3:
        st.header("Data Visualizations")

        # Create two columns for data visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Incident Feature")
            st.image(os.path.join(os.path.dirname(__file__), "static", "incidents.png"), use_container_width=True)

        with col2:
            st.subheader("Feature Importance")
            st.image(os.path.join(os.path.dirname(__file__), "static", "cross-correlation-matrix.png"), use_container_width=True)

        st.header("Airport Features")

        # Create three columns for airport-related visualizations
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Origin Airports")
            st.image(os.path.join(os.path.dirname(__file__), "static", "origin_airports.png"), use_container_width=True)

        with col2:
            st.subheader("Destination Airports")
            st.image(os.path.join(os.path.dirname(__file__), "static", "destination_airports.png"), use_container_width=True)

        with col3:
            st.subheader("Incident Routes")
            st.image(os.path.join(os.path.dirname(__file__), "static", "incident_routes.png"), use_container_width=True)

        st.header("Time Features")

        # Create two columns for time-related visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cyclical Encoding Visualization")
            st.image(os.path.join(os.path.dirname(__file__), "static", "cyclical_encoding.png"), use_container_width=True)

        with col2:
            st.subheader("Time Representation in a Circle Plot")
            st.image(os.path.join(os.path.dirname(__file__), "static", "circle_time.png"), use_container_width=True)

    # ---------------------------
    # TAB 4: Flight Route Animation UI
    # ---------------------------
    with tab4:
        # Map style selector
        map_styles = {
            "Outdoors": "mapbox://styles/mapbox/outdoors-v11",
            "Satellite": "mapbox://styles/mapbox/satellite-v9",
        }

        # Create two columns for route selection and map display
        col1, col2 = st.columns(2)

        # Route selection
        with col1:
            airports = load_airports()
            st.header("Select Route")
            origin_label = st.selectbox("Origin Airport", airports['Label'])
            dest_label = st.selectbox("Destination Airport", airports['Label'])
            selected_style = st.selectbox("Choose Map Style", list(map_styles.keys()))

        # Map display
        with col2:
            origin = airports[airports['Label'] == origin_label].iloc[0]
            destination = airports[airports['Label'] == dest_label].iloc[0]

            curved_path = generate_arc(
                (origin['Longitude'], origin['Latitude']),
                (destination['Longitude'], destination['Latitude']),
                num_points=100,
                height=8
            )

            icon_layer_data = pd.DataFrame([
                {"name": f"{origin['City']} ({origin['IATA']})", "coordinates": [origin['Longitude'], origin['Latitude']], "icon_data": icon_data},
                {"name": f"{destination['City']} ({destination['IATA']})", "coordinates": [destination['Longitude'], destination['Latitude']], "icon_data": icon_data}
            ])

            view_state = pdk.ViewState(
                latitude=(origin['Latitude'] + destination['Latitude']) / 2,
                longitude=(origin['Longitude'] + destination['Longitude']) / 2,
                zoom=2.25,
                pitch=0,
            )

            tooltip = {
                "html": "<b>Airport:</b> {name}",
                "style": {"backgroundColor": "black", "color": "white"}
            }

            icon_layer = pdk.Layer(
                "IconLayer",
                data=icon_layer_data,
                get_icon="icon_data",
                get_size=4,
                size_scale=15,
                get_position="coordinates",
                pickable=True,
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

            # Flight animation
            chart_placeholder = st.empty()

            with st.form("flight_form", clear_on_submit=True):
                clicked = st.form_submit_button(label="START")

            # Animate the flight
            if clicked:
                trail_length = 5
                for i in range(len(curved_path)):
                    plane_position = curved_path[i]
                    trail = curved_path[max(0, i - trail_length):i + 1]

                    r = pdk.Deck(
                        layers=[icon_layer, curved_layer],
                        initial_view_state=view_state,
                        map_style=map_styles[selected_style],
                        tooltip=tooltip
                    )

                    chart_placeholder.pydeck_chart(r)
                    time.sleep(0.05)

if __name__ == "__main__":
    main()