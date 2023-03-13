import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as sic
import tempfile

#! make auto-advance on click an option


def load_video(video_file=None):
    print("1")
    if st.session_state["vid_fbf"].frames is None:
        print("2")
        if video_file is None:
            print("3")
            with tempfile.NamedTemporaryFile() as tmp_f:
                print("4")
                video_file = st.session_state["source video"]
                print("5")
                tmp_f.write(video_file.read())
                print("6")
                st.session_state["vid_fbf"].load_video(tmp_f.name)
                print("7")
        else:
            print("8")
            st.session_state["vid_fbf"].load_video(video_file)
            print("9")


def prep_frame_number():
    frm_num = st.session_state["sldr_frm_num"]  #! - 1
    st.session_state["vid_fbf"].prep_frame_num(frm_num)


def display_frame():
    frame = st.session_state["vid_fbf"].get_frame()

    if frame is None:
        return None

    pil_img = Image.fromarray(frame)

    sic(
        pil_img,
        key="img_coords",
    )


def prev_frame():
    st.session_state["vid_fbf"].prep_prev_frame()


def next_frame():
    st.session_state["vid_fbf"].prep_next_frame()


def remove_ball_marker():
    st.session_state["vid_fbf"].rm_ball_frame_loc()
    prep_frame_number()


def build_and_show_composite():
    # st.session_state["vid_fbf"].create_composite_image()
    st.session_state["vid_fbf"].prep_frame_num(-1)


def set_curr_activity(activity_num: str):
    st.session_state["img_coords"] = None
    st.session_state["curr_activity"] = activity_num


def change_resolution():
    res = st.session_state["display_res"]
    st.session_state["vid_fbf"].set_display_resolution(resolution=res)
    prep_frame_number()


def set_ref_len_meters():
    len_m = 0.01 * st.session_state["ref_length"]
    st.session_state["vid_fbf"].set_ref_len_meters(len_m)
    st.session_state["vid_fbf"].calc_m_per_pxl()


def set_video_frame_rate():
    frm_rt = st.session_state["vid_frm_rt"]

    st.session_state["vid_fbf"].set_video_frame_rate(frm_rt)


def update_image_properties():
    b_fctr = st.session_state["image_brightness"]
    st.session_state["vid_fbf"].image_brightness_factor = b_fctr

    c_fctr = st.session_state["image_contrast"]
    st.session_state["vid_fbf"].image_contrast_factor = c_fctr


def tst_img_coords_slctd():
    prev_coords = st.session_state["img_coords_prev"]
    curr_coords = st.session_state["img_coords"]

    if (curr_coords is None) or (prev_coords == curr_coords):
        return

    if st.session_state["curr_activity"] == "Set Ball Location":
        st.session_state["vid_fbf"].set_ball_frame_loc(curr_coords)
        st.session_state["img_coords_prev"] = curr_coords
        next_frame()

    elif st.session_state["curr_activity"] == "Set Length Markers":
        st.session_state["vid_fbf"].set_length_markers(curr_coords)
        st.session_state["img_coords_prev"] = curr_coords
        st.session_state["vid_fbf"].calc_m_per_pxl()
        prep_frame_number()
        st.experimental_rerun()

    elif st.session_state["curr_activity"] == "Set Plum Line":
        st.session_state["vid_fbf"].set_plum_line_marker(curr_coords)
        st.session_state["img_coords_prev"] = curr_coords
        st.session_state["vid_fbf"].calc_frame_theta()
        prep_frame_number()
        st.experimental_rerun()

    st.session_state["img_coords"] = None


def panel_mjr_actions():
    st.write("Section action to perform:")

    cols = st.columns(5)
    # 1. ball location
    with cols[0]:
        lbl = "Set Ball Location"
        st.button(
            lbl,
            key="btn_set_ball_loc",
            on_click=set_curr_activity,
            args=(lbl,),
        )
        st.button(
            "Remove Frame Marker",
            key="btn_rm_frm_ball_mrkr",
            on_click=remove_ball_marker,
        )
        #! this needs to be a local function so the display can be updated

    # 2. length markers
    with cols[1]:
        lbl = "Set Length Markers"
        st.button(
            lbl,
            key="btn_ln_mrkrs",
            on_click=set_curr_activity,
            args=(lbl,),
        )
        rel_len_m = (
            0
            if st.session_state["vid_fbf"].ref_len_m is None
            else st.session_state["vid_fbf"].ref_len_m
        )
        st.number_input(
            "Length between markers (centimeters)",
            key="ref_length",
            min_value=0.0,
            format="%.3f",
            value=100.0 * rel_len_m,
            on_change=set_ref_len_meters,
        )
        if st.session_state["vid_fbf"].m_per_pxl in [0, 0.0, None]:
            m_per_pxl = "undefined"
            pxl_per_m = "undefined"
        else:
            m_per_pxl_val = st.session_state["vid_fbf"].m_per_pxl
            m_per_pxl = f"{m_per_pxl_val:.5f} m/pxl"
            pxl_per_m = f"{1./m_per_pxl_val:.2f} pxl/m"
        st.write(f"meters/pixel --> {m_per_pxl}")
        st.write(f"pixel/meters --> {pxl_per_m}")

    # # 3. plum line
    # with cols[2]:
    #     lbl = "Set Plum Line"
    #     st.button(lbl, key="btn_plum_line", on_click=set_curr_activity, args=(lbl,))
    #     if st.session_state["vid_fbf"].image_theta is None:
    #         theta = "undefined"
    #     else:
    #         theta_val = st.session_state["vid_fbf"].image_theta * 180 / np.pi
    #         theta = f"{theta_val:.2f} degrees"
    #     st.write(f"image rotation --> {theta} degrees")
    #     #! option to zoom on 1st click

    # 4. set background frame
    with cols[2]:
        st.button(
            "Set Background Frame",
            key="btn_set_bkgrnd_frm",
            on_click=st.session_state["vid_fbf"].set_frame_as_bkgrnd,
        )
        bkrnd_frm = st.session_state["vid_fbf"].bkgrnd_frm_num
        st.write(f"Frame to use as background: {bkrnd_frm}")

    # 5. process results & obtain fit
    with cols[3]:
        st.button(
            "Show Composite Image",
            key="btn_composite_img",
            on_click=build_and_show_composite,
        )

    # 6. set image size
    with cols[4]:
        # frm_rt = st.session_state["vid_fbf"].frame_rate
        # st.write(f"Video frame rate: {frm_rt:.2f} fps")
        st.number_input(
            "Video frame rate: ",
            key="vid_frm_rt",
            min_value=0.0,
            format="%.3f",
            value=st.session_state["vid_fbf"].frame_rate,
            on_change=set_video_frame_rate,
        )

        st.selectbox(
            "Image size",
            options=("original", "720p", "1080p", "2k"),
            index=0,
            key="display_res",
            on_change=change_resolution,
        )

        b_fctr = st.session_state["vid_fbf"].image_brightness_factor
        st.slider(
            "Image Brightness",
            key="image_brightness",
            value=b_fctr,
            min_value=1.0,
            max_value=100.0,
            step=0.01,
            on_change=update_image_properties,
        )

        b_fctr = st.session_state["vid_fbf"].image_contrast_factor
        st.slider(
            "Image contrast",
            key="image_contrast",
            value=b_fctr,
            min_value=0.0,
            max_value=2.0,
            step=0.01,
            on_change=update_image_properties,
        )

    curr_action = st.session_state["curr_activity"]
    st.write(f"Current action --> {curr_action}")


def panel_frame_controls():
    # Video frame selection controls
    cols = st.columns([1, 7, 1])

    with cols[0]:
        st.button("Previous Frame:", on_click=prev_frame, key="btn_prev_frm")

    with cols[1]:
        dfn = st.session_state["vid_fbf"].display_frame_num
        sldr_val = 0 if dfn is None else dfn
        # sldr_val = 1 if dfn is None else dfn + 1
        lst_frm = st.session_state["vid_fbf"].ttl_frms - 1
        st.slider(
            "Select Frame:",
            min_value=0,
            # min_value=1,
            max_value=lst_frm,
            value=sldr_val,
            step=1,
            key="sldr_frm_num",
            on_change=prep_frame_number,
        )

    with cols[2]:
        st.button("Next Frame:", on_click=next_frame, key="btn_nxt_frm")


def setup_app():
    st.session_state["vid_fbf"].show_calibration_markers = True

    # cols = st.columns([1, 1, 10])
    # with cols[0]:
    #     st.button("Save Object", on_click=st.session_state["vid_fbf"].save_obj)
    # with cols[1]:
    #     st.button("Load Object", on_click=st.session_state["vid_fbf"].load_obj)

    if st.session_state["vid_fbf"].file_name is None:
        # print("\n" * 5)
        st.file_uploader(
            label="Choose video:",
            type=["mp4", "mov"],
            accept_multiple_files=False,
            key="source video",
            on_change=load_video,
            # label_visibility="hidden",
        )

        st.button(
            "Use Default Video",
            on_click=load_video,
            kwargs={"video_file": "./2D-Kinematics.mov"},
            key="btn_use_dflt_vid",
        )
    else:
        panel_mjr_actions()

        st.markdown("""---""")

        tst_img_coords_slctd()
        display_frame()
        panel_frame_controls()

        # if st.session_state["vid_fbf"].x_fit_result is not None:
        #     print()
        #     print()
        #     print()
        #     print(st.session_state["vid_fbf"].x_fit_result.fit_report())

        #     print()
        #     print()
        #     print()
        #     print(st.session_state["vid_fbf"].y_fit_result.fit_report())
