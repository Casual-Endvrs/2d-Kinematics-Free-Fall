import streamlit as st
import cv2
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as sic
import tempfile

from cls_vid_fbf import vid_fbf

st.set_page_config(layout="wide")

if 'vid_fbf' not in st.session_state:
    st.session_state['vid_fbf'] = vid_fbf()

st.title('2D Kinematics Experiment')

uploaded_video = st.file_uploader(
    label = "Choose video:", 
    type = ["mp4", "mov"], 
    accept_multiple_files = False, 
    on_change = None, 
    # help = "Help stirng"
    )

if uploaded_video is not None :
    if 'src_video' not in st.session_state:
        st.session_state['src_video'] = uploaded_video

    print(uploaded_video)
    print(type(uploaded_video))


frame_skip = 25 # display every 300 frames

if uploaded_video is not None: # run only when user uploads video
    with tempfile.NamedTemporaryFile() as tmp_f:
        tmp_f.write(uploaded_video.read())
        st.session_state['vid_fbf'].load_video(tmp_f.name)
    

    vid = uploaded_video.name
    st.markdown(f"""
    ### Files
    - {vid}
    """,
    unsafe_allow_html=True) # display file name

    cur_frame = 0
    success = True

    frame = st.session_state['vid_fbf'].get_frame(9, 720)
    pil_img = Image.fromarray(frame)

    # for i in range(2) :
    #     cols = st.columns(1)
    #     with cols[0] :
    value = sic(
        pil_img,
        key = "pil",
        height = 720,
        width = 1280
    )

    st.write(value)

    print()
    print()
    print()
    print()
    print(dir(st))
    print()
    print()
    print()


    # container_width = st.get_container_width
    # st.write(container_width)

    screen_width = st.width
    st.write(screen_width)




