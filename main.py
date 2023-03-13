import streamlit as st
from multiapp import MultiApp
from instruction_app import instructions_app
from data_parameter_selection_app import setup_app
from kinematics_visualisation_app import visualisation_app

from cls_vid_fbf import vid_fbf

#! Issues:
#!  - Visualization App
#!      - When theta=0, the user plot is showing theta=30*
#!  - Conversions
#!      - Could be dt
#!      - Could be pxls/m

#! Image rotation issue
#!  - correct image rotation
#!  - transform user values

# RGB Corrected
# Forced a_x=0 and removed plumb lin option

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

app.add_app("Instructions", instructions_app)
app.add_app("Data Point Selection", setup_app)
app.add_app("Visualisation", visualisation_app)

app.run()
