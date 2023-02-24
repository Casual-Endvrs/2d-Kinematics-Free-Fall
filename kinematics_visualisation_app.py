import streamlit as st
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as sic


# def prep_frame_number():
#     frm_num = st.session_state["sldr_frm_num"]  #! - 1
#     st.session_state["vid_fbf"].prep_frame_num(frm_num)


def display_frame():
    frame = st.session_state["vid_fbf"].get_frame(-2)
    sic(
        Image.fromarray(frame),
        key="composite_image",
    )


#     frame = st.session_state["vid_fbf"].get_frame(-1)

#     if frame is None:
#         return None

#     pil_img = Image.fromarray(frame)

#     sic(
#         pil_img,
#         key="composite_image",
#     )


def set_param(param):
    value = st.session_state[param]
    st.session_state["vid_fbf"].update_user_exp_params(param, value)


def update_force_a_x_zero():
    st.session_state["vid_fbf"].assume_a_x_zero = st.session_state["force_a_x_zero"]


def visualisation_app():
    #! Frame resolution

    if st.session_state["vid_fbf"].file_name is None:
        st.write(
            "Please go back to the Setup page and prepare marker points in a video."
        )
    else:
        st.button(
            "Reset Values",
            on_click=st.session_state["vid_fbf"].update_best_fit_values,
            kwargs={"force_user_update": True},
        )

        cols = st.columns([3, 7])

        m_per_pxl = (
            1
            if st.session_state["vid_fbf"].m_per_pxl is None
            else st.session_state["vid_fbf"].m_per_pxl
        )

        if st.session_state["vid_fbf"].m_per_pxl is None:
            max_v = 2000.0
        else:
            max_v = 5.0

        print(m_per_pxl)

        with cols[0]:
            # v_0
            val = float(m_per_pxl * st.session_state["vid_fbf"].usr_exp_values["v_0"])
            st.slider(
                label="$v_0$",
                key="v_0",
                min_value=0.0,
                max_value=max_v,
                value=val,
                step=None,
                format=None,
                help=None,
                on_change=set_param,
                args=("v_0",),
            )

            # v_0_x
            val = float(m_per_pxl * st.session_state["vid_fbf"].usr_exp_values["v_0_x"])
            st.slider(
                label="$v_{0,x}$",
                key="v_0_x",
                min_value=0.0,
                max_value=max_v,
                value=val,
                step=None,
                format=None,
                help=None,
                on_change=set_param,
                args=("v_0_x",),
            )

            # v_0_y
            val = float(m_per_pxl * st.session_state["vid_fbf"].usr_exp_values["v_0_y"])
            st.slider(
                label="$v_{0,y}$",
                key="v_0_y",
                min_value=0.0,
                max_value=max_v,
                value=val,
                step=None,
                format=None,
                help=None,
                on_change=set_param,
                args=("v_0_y",),
            )

            # theta_0
            val = float(
                180 / np.pi * st.session_state["vid_fbf"].usr_exp_values["theta"]
            )
            st.slider(
                label=r"$\theta_{0}$",
                key="theta",
                min_value=-45.0,
                max_value=90.0,
                value=val,
                step=1.0,
                format=None,
                help=None,
                on_change=set_param,
                args=("theta",),
            )

            # g
            val = float(
                m_per_pxl * st.session_state["vid_fbf"].usr_exp_values["gravity"]
            )
            st.slider(
                label="gravity",
                key="gravity",
                min_value=-25.0,
                max_value=5.0,
                value=val,
                step=0.1,
                format=None,
                help=None,
                on_change=set_param,
                args=("gravity",),
            )

            # assume a_x = 0
            val = st.session_state["vid_fbf"].assume_a_x_zero
            st.checkbox(
                r"Assume $a_{x}$ = 0?",
                key="force_a_x_zero",
                value=val,
                on_change=update_force_a_x_zero,
            )

        with cols[1]:
            display_frame()
            # frame = st.session_state["vid_fbf"].get_frame(-2)
            # sic(
            #     Image.fromarray(frame),
            #     key="composite_image",
            # )

        st.markdown(st.session_state["vid_fbf"].fit_reports())
