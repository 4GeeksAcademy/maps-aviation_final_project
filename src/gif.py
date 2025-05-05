import streamlit as st
from pathlib import Path

# Define the local folder where GIFs are stored
gif_folder = Path("static")  # Ensure this folder exists

# Choose a specific GIF to display
gif_path = gif_folder / "/workspaces/maps-aviation_final_project/static/crash.gif"  # Replace with your actual GIF filename

# Streamlit App
st.title("GIF Display App")
st.write("Here is an animated GIF from a local folder.")

# Display the GIF
if gif_path.exists():
    st.image(str(gif_path), width= 500, caption="Example Animation")
else:
    st.write("GIF not found. Make sure the file exists in the 'gifs' folder.")

