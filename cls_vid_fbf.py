import numpy as np
import cv2
from pathlib import Path

class vid_fbf:
    # Class to store a video frame-by-frame with per-frame information.

    def __init__(self):
        self.file_name: str = None
        self.frames: list[int] = None
        self.ttl_frms: int = None
        self.display_frame: int = 0
    
    def load_video(self, file_path: Path):
        self.frames = []
        self.ttl_frms = 0
        self.display_frame = 0

        vidcap = cv2.VideoCapture(file_path)

        while True:
            is_frame, frame_img = vidcap.read()

            if not is_frame :
                break

            self.frames.append(frame_img)
            self.ttl_frms += 1

    def get_frame(self, frame_num:int = 0, target_res_h:int = 720):

        fsr = int(5)
        if 0 <= frame_num < self.ttl_frms :
            frame = np.copy(self.frames[frame_num])

            return frame


if __name__ == '__main__' :

    file_path = '/home/braden/Git_Repos/test_video_processing/20230131_220018.mp4'

    x = vid_fbf()
    x.load_video(file_path)

    print(x.ttl_frms)




