import streamlit as st
from multiapp import MultiApp
from data_parameter_selection_app import setup_app
from kinematics_visualization_app import visualization_app

from cls_vid_fbf import vid_fbf

# Use the full width of the display for the dashboard
st.set_page_config(layout="wide")

# This is a diagnostic. Set to True for development, set to False when pushing for general users.
showWarningOnDirectExecution = False

st.title("2D Kinematics Experiment")

if "vid_fbf" not in st.session_state:
    st.session_state["vid_fbf"] = vid_fbf()
    st.session_state["img_coords_prev"] = None
    st.session_state["img_coords"] = None
    st.session_state["curr_activity"] = "Set Ball Location"

app = MultiApp()

app.add_app("Setup", setup_app)
app.add_app("Visualization", visualization_app)

app.run()
