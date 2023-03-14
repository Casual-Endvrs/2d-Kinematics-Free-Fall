import numpy as np
from lmfit import Model, Parameters
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Union, List, Any, Dict

#! Add return statement type hints
#! Assumes each data point is sequential


class vid_fbf:
    # Class to store a video frame-by-frame with per-frame information.

    def __init__(self):
        self.file_name: Union[None, str] = None
        self.frames_src: Union[None, List[int]] = None
        self.frames: Union[None, List[int]] = None
        self.ttl_frms: Union[None, int] = None
        self.display_frame_num: Union[None, int] = None

        self.frame_image: Union[None, np.ndarray] = None

        #! Video editing variables
        self.composite_image = None  # : Union[None, List[int]]
        self.fit_image = None  # : Union[None, List[int]]

        self.source_resolution: Union[None, List[int]] = None
        self.display_resolution: Union[None, str, List[int]] = None
        self.image_brightness_factor: float = 1.0
        self.image_contrast_factor: float = 1.0

        self.ball_radius = 15
        self.bkgrnd_frm_num: int = 0
        self.show_calibration_markers: bool = True

        #! 2D Kinematics specific variables
        self.ball_frm_locs: Union[None, np.ndarray[(Any, 2), int]] = None
        self.length_markers_locs: Union[None, np.ndarray[(Any, 2), int]] = None
        self.plum_line_markers: Union[None, np.ndarray[(Any, 2), int]] = None
        self.image_theta: Union[None, float] = None

        self.assume_a_x_zero: bool = True

        self.x_fit_result = None
        self.y_fit_result = None

        self.best_fit_values: Union[None, Dict] = None
        self.usr_exp_values_defined = False
        self.usr_exp_values: Dict = {
            "x_0": 0.0,
            "v_0_x": 0.0,
            "y_0": 0.0,
            "v_0_y": 0.0,
            "v_0": 0.0,
            "theta": 0.0,
            "gravity": 0.0,
        }

        #! Equation variables
        self.frame_rate: float = 30.0
        self.frame_time: float = 1.0 / self.frame_rate

        self.ref_len_pxls: Union[None, int] = None
        self.ref_len_m: Union[None, float] = None
        self.m_per_pxl: Union[None, float] = None

        #! location of ball markers in pxls & m
        self.pos_pxls: Union[None, np.ndarray[(Any, 2), int]] = None
        self.pos_m: Union[None, np.ndarray[(Any, 2), float]] = None

    def set_video_frame_rate(self, frame_rate):
        self.frame_rate = frame_rate
        self.frame_time = 1.0 / self.frame_rate

    def load_video(self, file_path: Path, resolution: str = "original"):
        self.display_frame_num = 0

        vidcap = cv2.VideoCapture(file_path)
        self.frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
        if self.frame_rate == 0:
            self.frame_rate = 30
        self.frame_time = 1.0 / self.frame_rate
        src_frms = []

        while True:
            is_frame, frame_img = vidcap.read()

            if not is_frame:
                break

            src_frms.append(frame_img)

        self.file_name = file_path
        self.frames_src = np.array(src_frms)
        self.ttl_frms = np.shape(self.frames_src)[0]
        self.ball_frm_locs = np.zeros((self.ttl_frms, 2), dtype=int)

        src_res = np.shape(self.frames_src[0])
        self.source_resolution = [src_res[1], src_res[0]]

        self.set_display_resolution(resolution="original")

        self.resize_video(resolution)

        self.prep_frame_num(frame_num=0)

    def set_display_resolution(
        self,
        resolution: str = "1080p",
        height: Union[None, int] = None,
        width: Union[None, int] = None,
        # force_aspect_ratio: bool = False,
    ):
        src_sz = np.shape(self.frames_src[0])

        if height is not None:
            if height > src_sz[0]:
                self.frames = np.copy(self.frames_src)
                return
            width = src_sz[1] * height / src_sz[0]
        elif width is not None:
            if width > src_sz[1]:
                self.frames = np.copy(self.frames_src)
                return
            height = src_sz[0] * width / src_sz[1]
        elif resolution is not None:
            if resolution == "original":
                src_res = np.shape(self.frames_src[0])
                height, width = src_res[0], src_res[1]
            elif resolution == "360p":
                height, width = 360, 640
            elif resolution == "720p":
                height, width = 720, 1280
            elif resolution == "1080p":
                height, width = 1080, 1920
            elif resolution in ["1440p", "2k"]:
                height, width = 1440, 2560
            elif resolution in ["2160p", "4k"]:
                height, width = 2160, 3840
            elif resolution in ["4320p", "8k"]:
                height, width = 4320, 7680
            else:
                height, width = None, None

        if None not in [width, height]:
            self.display_resolution = [width, height]
        else:
            if self.frames_src is None:
                self.display_resolution = None
            else:
                src_res = np.shape(self.frames_src[0])
                self.display_resolution = [src_res[1], src_res[0]]

        # print(f"display res --> {self.display_resolution}")

        self.resize_video()

    def resize_display_frame(
        self,
        resolution: Union[None, str] = None,
        height: Union[None, int] = None,
        width: Union[None, int] = None,
        # force_aspect_ratio: bool = False,
    ):
        # if resolution is not None or height is not None or width is not None:
        if np.any([entry is not None for entry in [resolution, height, width]]):
            self.set_display_resolution(resolution, height, width)

        if self.display_resolution is None:
            return

        [width, height] = self.display_resolution

        self.frame_image = cv2.resize(self.frame_image, (width, height))

    def resize_video(
        self,
        resolution: Union[None, str] = None,
        height: Union[None, int] = None,
        width: Union[None, int] = None,
        # force_aspect_ratio: bool = False,
    ):
        if resolution == "original":
            self.frames = np.copy(self.frames_src)
            return

        # if resolution is not None or height is not None or width is not None:
        if np.any([entry is not None for entry in [resolution, height, width]]):
            self.set_display_resolution(resolution, height, width)

        if self.display_resolution is None:
            print("invalid: self.display_resolution is None")
            return

        [width, height] = self.display_resolution

        self.frames = []
        for frame in self.frames_src:
            frame = np.copy(frame)
            self.frames.append(cv2.resize(frame, (width, height)))

        #! I don't like this - should be np.array
        self.frames = list(self.frames)

    def prep_frame_num(self, frame_num: Union[None, int] = None):
        if frame_num is None:
            if self.display_frame_num is None:
                frame_num = 0
            else:
                frame_num = self.display_frame_num

        self.show_calibration_markers = True

        self.display_frame_num = frame_num

        if frame_num == -1:
            self.create_composite_image()
            self.frame_image = self.composite_image

            self.show_calibration_markers = False

            self.update_best_fit_values()

        elif frame_num == -2:
            self.create_fit_frame()
            self.frame_image = self.composite_image

            self.show_calibration_markers = False

        elif 0 <= frame_num < self.ttl_frms:
            self.frame_image = np.copy(self.frames_src[self.display_frame_num])
            self.plot_ball_marker()
            # self.plot_length_markers()
            # self.plot_plum_line_markers()
            # self.resize_display_frame()
        else:
            #! if an invalid frame number is requested then do???
            self.frame_image = None
            return

        if self.show_calibration_markers:
            self.plot_length_markers()
            self.plot_plum_line_markers()

        self.resize_display_frame()
        self.adjust_brightness_contrast()

    def get_frame(self, frame_num: Union[None, int] = None):
        self.prep_frame_num(frame_num)

        return self.frame_image

    def prep_prev_frame(self):
        prev_frame_idx = self.display_frame_num - 1
        self.prep_frame_num(prev_frame_idx)

    def prep_next_frame(self):
        next_frame_idx = self.display_frame_num + 1
        self.prep_frame_num(next_frame_idx)

    def adjust_brightness_contrast(
        self, brightness: Union[None, float] = None, contrast: Union[None, float] = None
    ):
        # print("adjust_brightness_contrast")

        if brightness is None:
            brightness = self.image_brightness_factor

        print(f"    contrast: {contrast}")
        if contrast is None:
            # print("contrast is None")
            contrast = self.image_contrast_factor
        # print("-----")
        # print(f"    contrast: {contrast}")
        # print(f"    image_contrast_factor: {self.image_contrast_factor}")
        # print("~~~~~")
        # print(f"    brightness: {brightness}")
        # print(f"    image_brightness_factor: {self.image_brightness_factor}")

        self.frame_image = cv2.convertScaleAbs(
            self.frame_image, alpha=contrast, beta=brightness
        )

        # print()

    #! ball markers
    def set_ball_frame_loc(self, ball_loc: Dict):
        if self.file_name is None:
            return

        x_loc, y_loc = self.scale_pixel_loc(
            ball_loc["x"], ball_loc["y"], to_original_res=True
        )

        self.ball_frm_locs[self.display_frame_num, 0] = x_loc
        self.ball_frm_locs[self.display_frame_num, 1] = y_loc

    def rm_ball_frame_loc(self):
        if self.file_name is None:
            return

        self.ball_frm_locs[self.display_frame_num, 0] = 0
        self.ball_frm_locs[self.display_frame_num, 1] = 0

    def plot_ball_marker(self):
        # print("plot_ball_marker")

        ball_loc = self.ball_frm_locs[self.display_frame_num]

        if ball_loc[0] == 0 and ball_loc[1] == 0:
            return

        x_y = ball_loc
        self.plot_marker(x_y=x_y, color=(0, 255, 0))

    #! length markers
    def set_length_markers(self, marker_loc: Dict):
        x_y = self.scale_pixel_loc(
            marker_loc["x"], marker_loc["y"], to_original_res=True
        )

        if self.length_markers_locs is None:
            self.length_markers_locs = [x_y, x_y]
        else:
            self.length_markers_locs[0] = self.length_markers_locs[1]
            self.length_markers_locs[1] = x_y

    def set_ref_len_meters(self, len_m: float):
        self.ref_len_m = len_m

    def calc_m_per_pxl(self):
        if (self.ref_len_m is None) or (self.ref_len_m < 0):
            self.m_per_pxl = None
            print("invalid: self.ref_len_m is None or 0")
            return

        if self.length_markers_locs is None:
            self.m_per_pxl = None
            print("invalid: self.length_markers_locs is None")
            return

        dx_pxl = self.length_markers_locs[0][0] - self.length_markers_locs[1][0]
        dy_pxl = self.length_markers_locs[0][1] - self.length_markers_locs[1][1]
        dist_pxl = np.sqrt(dx_pxl**2 + dy_pxl**2)

        if dist_pxl == 0:
            self.m_per_pxl = None
            print("invalid: dist_pxl == 0")
            return

        self.m_per_pxl = self.ref_len_m / dist_pxl

    def plot_length_markers(self):
        if self.length_markers_locs is None:
            return

        for x_y in self.length_markers_locs:
            self.plot_marker(x_y=x_y, color=(0, 0, 255))

    #! plum line markers
    def set_plum_line_marker(self, marker_loc: Dict):
        x_y = self.scale_pixel_loc(
            marker_loc["x"], marker_loc["y"], to_original_res=True
        )

        if self.plum_line_markers is None:
            self.plum_line_markers = [x_y, x_y]
        else:
            self.plum_line_markers[0] = self.plum_line_markers[1]
            self.plum_line_markers[1] = x_y

    def calc_frame_theta_from_plumb_line(self):
        if self.plum_line_markers is None:
            self.image_theta = None
            print("invalid: self.plum_line_markers is None")
            return

        start_pnt = self.plum_line_markers[0]
        end_pnt = self.plum_line_markers[1]
        if start_pnt[1] > end_pnt[1]:
            start_pnt, end_pnt = end_pnt, start_pnt

        dx_pxl = end_pnt[0] - start_pnt[0]
        dy_pxl = end_pnt[1] - start_pnt[1]

        if dy_pxl == 0:
            print("invalid: dy_pxl == 0")
            self.image_theta = None
            return
        else:
            self.image_theta = np.arctan(dx_pxl / dy_pxl)

    def plot_plum_line_markers(self):
        if self.plum_line_markers is None:
            return

        for x_y in self.plum_line_markers:
            self.plot_marker(x_y=x_y, color=(156, 81, 182))

    def plot_marker(
        self,
        x_y,
        color=(255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
        xy_is_src_res: bool = True,
    ):
        if not xy_is_src_res:
            x_y = self.scale_pixel_loc(*x_y, to_original_res=True)

        cv2.drawMarker(
            self.frame_image, tuple(x_y), color, markerType, markerSize, thickness
        )

    #! Video editing methods
    def create_fit_frame(self):
        self.create_composite_image()
        self.draw_parametric_fit()
        self.draw_user_line()

    def scale_pixel_loc(
        self, width: int, height: int, to_original_res: bool = True
    ) -> List[int]:
        if to_original_res:
            width = width * self.source_resolution[0] / self.display_resolution[0]
            height = height * self.source_resolution[1] / self.display_resolution[1]
        else:
            width = width * self.display_resolution[0] / self.source_resolution[0]
            height = height * self.display_resolution[1] / self.source_resolution[1]

        width = int(width)
        height = int(height)

        return [width, height]

    def set_frame_as_bkgrnd(self):
        if 0 <= self.display_frame_num < self.ttl_frms:
            self.bkgrnd_frm_num = self.display_frame_num

    def create_composite_image(self):
        background = Image.fromarray(self.frames_src[self.bkgrnd_frm_num])

        bkgrnd_shp = tuple(np.shape(background)[:2][::-1])

        for idx in np.arange(self.ttl_frms):
            ball_loc = self.ball_frm_locs[idx]
            if ball_loc[0] == 0 and ball_loc[1] == 0:
                continue

            ball_frame = Image.fromarray(self.frames_src[idx])

            xc, yc = ball_loc
            x1 = xc - self.ball_radius
            x2 = xc + self.ball_radius
            y1 = yc - self.ball_radius
            y2 = yc + self.ball_radius

            mask = Image.new("L", bkgrnd_shp, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((x1, y1, x2, y2), fill=255)

            background = Image.composite(ball_frame, background, mask)

        self.composite_image = np.array(background)

        self.fit_data()

    def draw_parametric_fit(self):
        if None in [self.x_fit_result, self.y_fit_result]:
            self.fit_data()

        if None not in [self.x_fit_result, self.y_fit_result]:
            draw_points = (
                np.asarray(
                    [self.x_fit_result.best_fit, -1 * self.y_fit_result.best_fit]
                ).T
            ).astype(np.int32)
            cv2.polylines(self.composite_image, [draw_points], False, (0, 255, 0))

        # num_data_pnts = np.shape(self.pos_m)[0]

        # steps_per_frm = 1
        # num_steps = steps_per_frm * num_data_pnts
        # t_max = self.frame_time * num_data_pnts
        # ts = np.linspace(0, t_max, num_steps)
        # dt = self.frame_time / steps_per_frm
        # ts = np.arange(0, t_max+dt, dt)

        # xs = self.k_displacement(ts, **self.x_fit_result.best_values)
        # ys = self.k_displacement(ts, **self.y_fit_result.best_values)

        # draw_points = (np.asarray([xs, ys]).T).astype(np.int32)
        # cv2.polylines(self.frames[0], [draw_points], False, (255, 0, 0))

    def draw_user_line(self):
        self.update_best_fit_values()

        if self.pos_pxls is not None:
            num_data_pnts = np.shape(self.pos_pxls)[0]

            steps_per_frm = 1
            num_steps = steps_per_frm * num_data_pnts
            t_max = self.frame_time * num_data_pnts
            ts = np.linspace(0, t_max, num_steps)
            # dt = self.frame_time / steps_per_frm
            # ts = np.arange(0, t_max + dt, dt)

            x_0 = self.usr_exp_values["x_0"]
            v_0_x = self.usr_exp_values["v_0_x"]
            a_x = 0
            y_0 = self.usr_exp_values["y_0"]
            v_0_y = self.usr_exp_values["v_0_y"]
            a_y = self.usr_exp_values["gravity"]
            # v_0 = self.usr_exp_values["v_0"]
            # theta = self.usr_exp_values["theta"]

            xs = self.k_displacement(ts, x_0=x_0, v_0=v_0_x, a=a_x)
            ys = self.k_displacement(ts, x_0=y_0, v_0=v_0_y, a=a_y)

            if self.image_theta is not None:
                # [xs, ys] = self.rotate_vector([xs, ys], self.image_theta, rotate_about_first=True)

                cos_theta = np.cos(-1 * self.image_theta)
                sin_theta = np.sin(-1 * self.image_theta)

                origin = np.array([xs[0], ys[0]])

                xs -= origin[0]
                ys -= origin[1]

                pos_vec = np.array(list(zip(*[xs, ys])))
                pos_vec = np.transpose(pos_vec)

                rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

                pos_vec = np.matmul(rot_matrix, pos_vec)
                pos_vec = np.transpose(pos_vec)

                xs = np.array(pos_vec[:, 0])
                ys = np.array(pos_vec[:, 1])

                xs += origin[0]
                ys += origin[1]

            draw_points = (np.asarray([xs, -1 * ys]).T).astype(np.int32)
            cv2.polylines(self.composite_image, [draw_points], False, (255, 0, 0))

    #! Equation variables
    def create_ball_pos_pxls(self):
        locs = np.copy(self.ball_frm_locs).transpose()

        ts = self.frame_time * np.arange(np.shape(locs)[1])

        # print()
        # print(f"self.frame_time --> {self.frame_time}")
        # print()

        locs = np.array([ts, locs[0], locs[1]])
        locs = locs.transpose()

        ball_has_pos = locs[:, 1] != 0
        if np.any(ball_has_pos):
            self.pos_pxls = locs[ball_has_pos, :]

            # t_0 = 0
            self.pos_pxls[:, 0] -= self.pos_pxls[0, 0]

            # inverting the y-axis
            self.pos_pxls[:, 2] *= -1.0

    def calc_ball_pos_meters(self):
        self.create_ball_pos_pxls()

        #! origin translation should be moved to an export function
        # # set origin to be at the first ball
        # #   x-axis shift
        # self.pos_pxls[:, 0] -= self.pos_pxls[0, 0]

        # #   y-axis shift & flip --> flip as y-axis is inverted
        # self.pos_pxls[:, 1] = -1 * self.pos_pxls[:, 1] + self.pos_pxls[0, 1]

        # convert position in pxls --> meters
        if self.m_per_pxl is None:
            self.pos_m = None
        else:
            self.pos_m = np.copy(self.pos_pxls)
            self.pos_m[1] = self.pos_m[1] * self.m_per_pxl
            self.pos_m[2] = self.pos_m[2] * self.m_per_pxl

    def fit_data(self):
        self.create_ball_pos_pxls()

        self.fit_x_data()
        self.fit_y_data()

        self.update_best_fit_values()

    def fit_x_data(self):
        self.create_ball_pos_pxls()

        if self.pos_pxls is not None:
            data = self.pos_pxls[:, 1]
            ts = self.pos_pxls[:, 0]

            model = Model(self.k_displacement)
            params = Parameters()
            params.add("x_0", value=data[0], vary=False)
            params.add("v_0", value=1)
            params.add("a", value=0)
            self.x_fit_result = model.fit(data, t=ts, params=params)

    def fit_y_data(self):
        self.create_ball_pos_pxls()

        if self.pos_pxls is not None:
            data = self.pos_pxls[:, 2]
            ts = self.pos_pxls[:, 0]

            a_0 = 1e-2
            if self.m_per_pxl is not None:
                a_0 = -10 / self.m_per_pxl

            model = Model(self.k_displacement)
            params = Parameters()
            params.add("x_0", value=data[0], vary=False)
            params.add("v_0", value=1)
            params.add("a", value=a_0)
            self.y_fit_result = model.fit(data, t=ts, params=params)

    def update_best_fit_values(self, force_user_update: bool = False):
        if self.pos_pxls is not None:
            x_0 = self.x_fit_result.best_values["x_0"]
            v_0_x = self.x_fit_result.best_values["v_0"]
            a_x = self.x_fit_result.best_values["a"]

            y_0 = self.y_fit_result.best_values["x_0"]
            v_0_y = self.y_fit_result.best_values["v_0"]
            a_y = self.y_fit_result.best_values["a"]

            v_0 = np.sqrt(v_0_x**2 + v_0_y**2)
            theta = np.arctan(a_x / a_y)

            self.best_fit_values = {
                "x_0": x_0,
                "v_0_x": v_0_x,
                "a_x": a_x,
                "y_0": y_0,
                "v_0_y": v_0_y,
                "a_y": a_y,
                "v_0": v_0,
                "theta": theta,
            }

            self.usr_exp_values["x_0"] = x_0
            self.usr_exp_values["y_0"] = y_0

            if force_user_update or not self.usr_exp_values_defined:
                self.set_user_exp_to_fit_values()

            if self.image_theta is None:
                self.image_theta = theta

    def set_user_exp_to_fit_values(self):
        if self.pos_pxls is not None:
            x_0 = self.x_fit_result.best_values["x_0"]
            v_0_x = self.x_fit_result.best_values["v_0"]
            a_x = self.x_fit_result.best_values["a"]

            y_0 = self.y_fit_result.best_values["x_0"]
            v_0_y = self.y_fit_result.best_values["v_0"]
            a_y = self.y_fit_result.best_values["a"]

            v_0 = np.sqrt(v_0_x**2 + v_0_y**2)
            theta = np.arctan(v_0_y / v_0_x)

            self.usr_exp_values["x_0"] = x_0
            self.usr_exp_values["y_0"] = y_0

            theta_rotate = np.arctan(a_x / a_y)
            [a_x, a_y] = self.rotate_vector([a_x, a_y], theta_rotate)
            [v_0_x, v_0_y] = self.rotate_vector([v_0_x, v_0_y], theta_rotate)
            self.usr_exp_values["v_0_x"] = v_0_x
            self.usr_exp_values["v_0_y"] = v_0_y
            self.usr_exp_values["gravity"] = a_y
            self.usr_exp_values["v_0"] = v_0
            self.usr_exp_values["theta"] = theta + theta_rotate

            self.usr_exp_values_defined = True

    def fit_reports(self) -> str:
        self.update_best_fit_values()

        if self.pos_pxls is not None:
            conv_fctr = 1 if self.m_per_pxl is None else self.m_per_pxl
            units = "pxls" if self.m_per_pxl is None else "m"

            a_x = self.best_fit_values["a_x"]
            a_y = self.best_fit_values["a_y"]
            v_0_x = self.best_fit_values["v_0_x"]
            v_0_y = self.best_fit_values["v_0_y"]
            theta = self.best_fit_values["theta"]

            theta_correction = None
            if self.assume_a_x_zero:
                theta_correction = np.arctan(a_x / a_y)
            elif self.image_theta is not None:
                theta_correction = -1 * self.image_theta

            if theta_correction is not None:
                [v_0_x, v_0_y] = self.rotate_vector([v_0_x, v_0_y], theta_correction)
                [a_x, a_y] = self.rotate_vector([a_x, a_y], theta_correction)
                # cos_t = np.cos(theta_correction)
                # sin_t = np.sin(theta_correction)

                # R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                # [v_0_x, v_0_y] = R @ np.array([v_0_x, v_0_y])
                # [a_x, a_y] = R @ np.array([a_x, a_y])

                theta += theta_correction

            fit_report = "  \n".join(
                [
                    "---",
                    "# Fit results",
                    "## x-axis values:",
                    # f"* x_0 = {conv_fctr * self.best_fit_values['x_0']:.2f} {units}",
                    f"* v_0_x = {conv_fctr * v_0_x:.2f} {units}/s",
                    f"* a_x = {conv_fctr * a_x:.2f} {units}/s$^2$",
                    "## y-axis values:",
                    # f"* y_0 = {conv_fctr * self.best_fit_values['y_0']:.2f} {units}",
                    f"* v_0_y = {conv_fctr * v_0_y:.2f} {units}/s",
                    f"* a_y = g = {conv_fctr * a_y:.2f} {units}/s$^2$",
                    "## General values:",
                    f"* v_0 = {conv_fctr * self.best_fit_values['v_0']:.2f} {units}/s",
                    f"* theta = {180 / np.pi * theta:.2f} degrees",
                ]
            )

            return fit_report

    def rotate_vector(self, vector, angle, rotate_about_first=False):
        """
        vector --> [xs, ys]
        """
        vector = np.array(vector)

        origin = [vector[0], vector[1]]
        if rotate_about_first:
            vector[0] -= origin[0]
            vector[1] -= origin[1]

        cos_t = np.cos(angle)
        sin_t = np.sin(angle)

        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        vector = R @ vector

        if rotate_about_first:
            vector[0] += origin[0]
            vector[1] += origin[1]

        return vector

    @staticmethod
    def k_displacement(t, x_0, v_0, a):
        return x_0 + v_0 * t + 0.5 * a * t**2

    #! User experimentation
    def update_user_exp_params(
        self,
        param: str,
        value: Union[int, float, np.float_],
        theta_is_degrees: bool = True,
    ):
        v_0 = self.usr_exp_values["v_0"]
        v_0_x = self.usr_exp_values["v_0_x"]
        v_0_y = self.usr_exp_values["v_0_y"]
        theta = self.usr_exp_values["theta"]
        gravity = self.usr_exp_values["gravity"]

        if param in ["v_0", "v_0_x", "v_0_y", "gravity"]:
            if self.m_per_pxl is not None:
                value /= self.m_per_pxl

        if param == "v_0":
            v_0 = value
            v_0_x = v_0 * np.cos(theta)
            v_0_y = v_0 * np.sin(theta)
        elif param == "v_0_x":
            v_0_x = value
            v_0 = np.sqrt(v_0_x**2 + v_0_y**2)
            theta = np.arctan(v_0_y / v_0_x)
        elif param == "v_0_y":
            v_0_y = value
            v_0 = np.sqrt(v_0_x**2 + v_0_y**2)
            theta = np.arctan(v_0_y / v_0_x)
        elif param == "theta":
            if theta_is_degrees:
                value = value * np.pi / 180
            theta = value
            v_0_x = v_0 * np.cos(theta)
            v_0_y = v_0 * np.sin(theta)
        elif param == "gravity":
            gravity = value

        self.usr_exp_values["v_0"] = float(v_0)
        self.usr_exp_values["v_0_x"] = float(v_0_x)
        self.usr_exp_values["v_0_y"] = float(v_0_y)
        self.usr_exp_values["theta"] = float(theta)
        self.usr_exp_values["gravity"] = float(gravity)


if __name__ == "__main__":
    file_path = "/home/braden/Git_Repos/test_video_processing/20230131_220018.mp4"

    x = vid_fbf()
    x.load_video(file_path)

    print(x.ttl_frms)
