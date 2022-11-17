import glob
import cv2 as cv
import numpy as np
import os
from decord import VideoReader
from decord import cpu, gpu

"""
   file: video.py
   subject : class video
   Author : AVS7
   Creation : 7/11/2022
   Last Update : 7/11/2022
   Update Note:
        7/11/2022 : Creation


"""


class video:
    DISTANCE_FROM_MIDDLE = 200

    def __init__(self, video, name_video, video_number):
        """

        Args:
            video: video

        """

        # gereral attributes

        self.name = name_video  # video's name
        self.vidcap = cv.VideoCapture(video)  # vidcap is the video
        self.number_frames = int(self.vidcap.get(cv.CAP_PROP_FRAME_COUNT))  # number of frames in the video
        self.num_video = video_number  # number of the video
        self.count_fish = 0  # number of fish found
        ret, next_frame = self.vidcap.read()  # read the first frame
        self.height, self.width, self.channels = next_frame.shape  # get the height, width and channels of the video
        self.frames = np.empty((self.number_frames, self.height, self.width, self.channels), dtype=np.uint8)  # array of frames
        self.gray_frames = np.empty((self.number_frames, self.height, self.width), dtype=np.uint8)  # array of gray frames
        self.foreground_frames = np.zeros((self.height, self.width), dtype=np.uint8)  # foreground frames
        self.numbers_frames_fish_detected = None  # number of frames where fish is detected
        self.frames_direction = np.zeros((self.number_frames, 1), dtype=np.int8)  # direction of the fish
        self.exit_frames_numbers = np.array([])  # list of frames where fish left
        self.enter_frames_numbers = np.array([])  # list of frames where fish enters
        self.fish_count_frames = np.array([])  # list of frames where fish count is updated
        # baseline 0.1 attributes
        self.line1_after_background = np.array([])
        self.line_2_after_background = np.array([])
        self.line1 = np.zeros((self.height, 1), dtype=np.uint8)
        self.line2 = np.zeros((self.height, 1), dtype=np.uint8)
        self.evolution_var1 = np.zeros(self.number_frames)
        self.evolution_var2 = np.zeros(self.number_frames)
        self.sequence = None

        k = 0
        while ret:
            self.frames[k] = next_frame
            self.gray_frames[k] = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
            ret, next_frame = self.vidcap.read()
            k += 1
        self.vidcap.release()

    def set_lines(self):
        # TODO : set lines for background, or normale frames
        # [:, middle + distance: middle + distance + 1]
        self.line1 = self.gray_frames[:, :, self.width // 2 + self.DISTANCE_FROM_MIDDLE:self.width // 2 + self.DISTANCE_FROM_MIDDLE + 1].copy()
        self.line2 = self.gray_frames[:, :, self.width // 2 - self.DISTANCE_FROM_MIDDLE:self.width // 2 - self.DISTANCE_FROM_MIDDLE + 1].copy()
