import streamlit as st
import pandas as pd

def main():
    st.set_page_config(page_title="Flight Data Selector", layout="wide")
    st.title("✈️ Flight Data Selector")
    
    # Load the CSV data
    @st.cache_data
    def load_data():
        df = pd.read_csv("/workspaces/maps-aviation_final_project/data/processed/combined_data.csv")
        # Strip any whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        return df
    
    # Load data
    try:
        df = load_data()
        st.success("Data loaded successfully!")
        
        # Display the raw data in an expandable section
        with st.expander("View Raw Data"):
            st.dataframe(df)
        
        # Create columns for selection widgets
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        # Get unique values for dropdowns
        origins = sorted(df['origin'].unique().tolist())
        destinations = sorted(df['destination'].unique().tolist())
        departure_times = sorted(df['departure_time'].unique().tolist())
        tail_numbers = sorted(df['tail_number'].unique().tolist())
        
        # Create dropdowns in columns
        with col1:
            selected_origin = st.selectbox("Select Origin Airport", origins)
        
        with col2:
            selected_destination = st.selectbox("Select Destination Airport", destinations)
        
        with col3:
            selected_departure = st.selectbox("Select Departure Time", departure_times)
        
        with col4:
            selected_tail = st.selectbox("Select Tail Number", tail_numbers)
        
        # Display the selected values
        st.subheader("Selected Flight Information")
        
        selected_info = {
            "Origin": selected_origin,
            "Destination": selected_destination,
            "Departure Time": selected_departure,
            "Tail Number": selected_tail
        }
        
        # Convert to DataFrame for nicer display
        selected_df = pd.DataFrame([selected_info])
        st.table(selected_df)
        
        # Show matching flights from the dataset
        st.subheader("Matching Flights")
        
        # Filter data based on selections
        matching_flights = df[
            (df['origin'] == selected_origin) &
            (df['destination'] == selected_destination) &
            (df['departure_time'] == selected_departure) &
            (df['tail_number'] == selected_tail)
        ]
        
        if len(matching_flights) > 0:
            st.success(f"Found {len(matching_flights)} matching flight(s)!")
            st.dataframe(matching_flights)
        else:
            st.warning("No exact matches found in the dataset.")
            
            # Show partial matches
            st.subheader("Partial Matches")
            
            partial_matches = df[
                (df['origin'] == selected_origin) &
                (df['destination'] == selected_destination)
            ]
            
            if len(partial_matches) > 0:
                st.info(f"Found {len(partial_matches)} flights between {selected_origin} and {selected_destination}")
                st.dataframe(partial_matches)
            else:
                st.info("No flights found between the selected airports.")
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please make sure the 'data.csv' file is in the same directory as this app.")

if __name__ == "__main__":
    main()