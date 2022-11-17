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
        vr = VideoReader(video, ctx=cpu(0))
        frames_list = list(range(0, len(vr)))
        self.frames = vr.get_batch(frames_list).asnumpy()
        self.height, self.width, self.channels = self.frames[0].shape
        self.gray_frames = np.empty((len(vr), self.height, self.width), dtype=np.uint8)  # array of gray frames

        k = 0
        while k < len(vr):
            self.gray_frames[k] = cv.cvtColor(self.frames[k], cv.COLOR_BGR2GRAY)
            k += 1


        self.name = name_video  # video's name
        self.number_frames = len(vr)  # number of frames in the video
        self.num_video = video_number  # number of the video
        self.count_fish = 0  # number of fish found
        self.frames_direction = np.zeros((self.number_frames, 1), dtype=np.int8)  # direction of the fish
        self.exit_frames_numbers = np.array([])  # list of frames where fish left
        self.enter_frames_numbers = np.array([])  # list of frames where fish enters
        self.fish_count_frames = np.array([])  # list of frames where fish count is updated
        # baseline 0.1 attributes
        self.line1 = np.zeros((self.height, 1), dtype=np.uint8)
        self.line2 = np.zeros((self.height, 1), dtype=np.uint8)
        self.sequence = None


    def set_lines(self):
        # TODO : set lines for background, or normale frames
        # [:, middle + distance: middle + distance + 1]
        self.line1 = self.gray_frames[:, :, self.width // 2 + self.DISTANCE_FROM_MIDDLE:self.width // 2 + self.DISTANCE_FROM_MIDDLE + 1].copy()
        self.line2 = self.gray_frames[:, :, self.width // 2 - self.DISTANCE_FROM_MIDDLE:self.width // 2 - self.DISTANCE_FROM_MIDDLE + 1].copy()
