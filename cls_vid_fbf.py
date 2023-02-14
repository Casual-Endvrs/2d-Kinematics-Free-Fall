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
        self.composite_image: Union[None, List[int]] = None
        self.bkgrnd_frm_num: int = 0
        self.ball_radius = 15
        self.source_resolution: Union[None, List[int]] = None
        self.display_resolution: Union[None, str, List[int]] = None

        #! 2D Kinematics specific variables
        self.ball_frm_locs: Union[None, np.ndarray[(Any, 2), int]] = None
        self.length_markers_locs: Union[None, np.ndarray[(Any, 2), int]] = None
        self.plum_line_markers: Union[None, np.ndarray[(Any, 2), int]] = None
        self.image_theta: Union[None, float] = None

        self.x_fit_result = None
        self.y_fit_result = None

        #! Equation variables
        self.frame_rate: float = 30.0
        self.frame_time: float = 1.0 / self.frame_rate

        self.ref_len_pxls: Union[None, int] = None
        self.ref_len_m: Union[None, float] = None
        self.m_per_pxl: Union[None, float] = None

        #! location of ball markers in pxls & m
        self.pos_pxls: Union[None, np.ndarray[(Any, 2), int]] = None
        self.pos_m: Union[None, np.ndarray[(Any, 2), float]] = None

    def load_video(self, file_path: Path, resolution: str = "original"):
        self.display_frame_num = 0

        vidcap = cv2.VideoCapture(file_path)
        self.frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1.0 / self.frame_rate
        src_frms = []

        while True:
            is_frame, frame_img = vidcap.read()

            if not is_frame:
                break

            src_frms.append(frame_img)

        print("Source size:")
        print(f"\t{np.shape(src_frms[0])}")

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
                print(f"height > {src_sz[0]}")
                self.frames = np.copy(self.frames_src)
                return
            width = src_sz[1] * height / src_sz[0]
        elif width is not None:
            if width > src_sz[1]:
                print(f"width > {src_sz[1]}")
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
                print("resolution was not found")
                height, width = None, None

            print(f"using resolution --> {resolution}")
            print(f"{width} x {height}")

        if None not in [width, height]:
            print("setting new resolution")
            self.display_resolution = [width, height]
        else:
            print("forcing original resolution")
            if self.frames_src is None:
                self.display_resolution = None
            else:
                src_res = np.shape(self.frames_src[0])
                self.display_resolution = [src_res[1], src_res[0]]

        print(f"display res --> {self.display_resolution}")

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

        if frame_num == -1:
            self.display_frame_num = frame_num
            self.create_composite_image()
            self.frame_image = self.composite_image
            # self.resize_display_frame()

        elif 0 <= frame_num < self.ttl_frms:
            self.display_frame_num = frame_num
            self.frame_image = np.copy(self.frames_src[self.display_frame_num])
            self.plot_ball_marker()
            # self.plot_length_markers()
            # self.plot_plum_line_markers()
            # self.resize_display_frame()
        else:
            #! if an invalid frame number is requested then do???
            self.frame_image = None
            return

        self.plot_length_markers()
        self.plot_plum_line_markers()
        self.resize_display_frame()

    def get_frame(self, frame_num: Union[None, int] = None):
        self.prep_frame_num(frame_num)

        return self.frame_image

    def prep_prev_frame(self):
        prev_frame_idx = self.display_frame_num - 1
        self.prep_frame_num(prev_frame_idx)

    def prep_next_frame(self):
        next_frame_idx = self.display_frame_num + 1
        self.prep_frame_num(next_frame_idx)

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
        print("plot_ball_marker")

        ball_loc = self.ball_frm_locs[self.display_frame_num]

        if ball_loc[0] == 0 and ball_loc[1] == 0:
            return

        print(f"\toriginal: {ball_loc}")
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

    # self.ref_len_pxls: Union[None, int] = None
    # self.ref_len_m: Union[None, float] = None
    # self.m_per_pxl: Union[None, float] = None
    # self.plum_line_markers: Union[None, np.ndarray[(Any, 2), int]] = None
    # self.image_theta: Union[None, float] = None

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

    def calc_frame_theta(self):
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

    def plot_kin_eqns(self):
        if self.composite_image is None:
            self.create_composite_image()

        if None in [self.x_fit_result, self.y_fit_result]:
            self.fit_data()

        self.draw_parametric_fit()

    def draw_parametric_fit(self):
        draw_points = (
            np.asarray([self.x_fit_result.best_fit, self.y_fit_result.best_fit]).T
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

    #! Equation variables
    def calc_ball_displacements(self):
        ball_has_pos = self.ball_frm_locs[:, 0] != 0
        self.pos_pxls = self.ball_frm_locs[ball_has_pos, :]

        #! origin translation should be moved to an export function
        # # set origin to be at the first ball
        # #   x-axis shift
        # self.pos_pxls[:, 0] -= self.pos_pxls[0, 0]

        # #   y-axis shift & flip --> flip as y-axis is inverted
        # self.pos_pxls[:, 1] = -1 * self.pos_pxls[:, 1] + self.pos_pxls[0, 1]

        # convert position in pxls --> meters
        self.pos_m = self.pos_pxls * self.m_per_pxl

    def fit_data(self):
        self.fit_x_data()
        self.fit_y_data()

    def fit_x_data(self):
        self.calc_ball_displacements()

        data = self.pos_pxls[:, 0]
        ts = self.frame_time * np.arange(len(data))

        model = Model(self.k_displacement)
        params = Parameters()
        params.add("x_0", value=data[0], vary=False)
        params.add("v_0", value=1)
        params.add("a", value=0)
        self.x_fit_result = model.fit(data, t=ts, params=params)

    def fit_y_data(self):
        self.calc_ball_displacements()

        data = self.pos_pxls[:, 1]
        ts = self.frame_time * np.arange(len(data))

        model = Model(self.k_displacement)
        params = Parameters()
        params.add("x_0", value=data[0], vary=False)
        params.add("v_0", value=1)
        params.add("a", value=-10 / self.m_per_pxl)
        self.y_fit_result = model.fit(data, t=ts, params=params)

    @staticmethod
    def k_displacement(t, x_0, v_0, a):
        return x_0 + v_0 * t + 0.5 * a * t**2


if __name__ == "__main__":
    file_path = "/home/braden/Git_Repos/test_video_processing/20230131_220018.mp4"

    x = vid_fbf()
    x.load_video(file_path)

    print(x.ttl_frms)
