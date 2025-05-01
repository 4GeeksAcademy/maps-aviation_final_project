import streamlit as st
from PIL import Image

# ---------------------------
# PAGE CONFIG & BANNER IMAGE
# ---------------------------
st.set_page_config(page_title="Flight Incident Predictor", layout="wide")
image = Image.open("/workspaces/Madesh3-aviation_final_project/static/photo.jpg")

# ---------------------------
# MAIN APP
# ---------------------------
def main():
    st.title("✈️ Flight Incident Predictor")

    # Intro Text
    st.markdown("""
    This app predicts the likelihood of a flight incident based on origin, destination, 
    and departure time information.
    """)
    st.image(image, width=200)
    # ---------------------------
    # TABS
    # ---------------------------
    tab1, tab2, tab3 = st.tabs(["Incident Predictor", "Model Performance", "Data Exploration"])

    # TAB 1: Incident Predictor UI Skeleton
    with tab1:
        st.header("Predict Flight Incident Risk")

        col1, col2 = st.columns(2)

        with col1:
            origin = st.selectbox("Origin Airport", ["Select..."])
            destination = st.selectbox("Destination Airport", ["Select..."])

        with col2:
            departure_time = st.number_input("Departure Time (e.g., 1430 for 2:30 PM)", min_value=0, max_value=2359, step=5)

        if st.button("Predict Incident Probability"):
            st.warning("Prediction logic not implemented.")
    # TAB 2: Model Performance UI Skeleton
    with tab2:
        st.header("Model Performance Metrics")
        st.info("Model evaluation metrics and visualizations will appear here.")

    # TAB 3: Data Exploration UI Skeleton
    with tab3:
        st.header("Data Exploration")
        st.info("Raw data views, visualizations, and correlation plots will be added here.")


if __name__ == "__main__":
    main()
