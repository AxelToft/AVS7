import glob
import cv2 as cv
import numpy as np


class video:
    def __init__(self, video,num_video):
        """
        Video object
        Args:
            video_path: path to the video
        """
        self.vidcap = cv.VideoCapture(video)
        self.background_vidcap = cv.VideoCapture(video)
        self.number_frames = int(self.vidcap.get(cv.CAP_PROP_FRAME_COUNT))
        ret, next_frame = self.vidcap.read()
        self.num_video = num_video
        self.count_fish = 0
        self.actual_count = 0
        self.k = 0
        self.gray_frames = np.array([])
        self.height, self.width, self.channels = next_frame.shape
        self.gray_background =None
        self.gray_line_1 = None
        self.gray_line_2 = None
        self.evolution_var1 = np.zeros(self.number_frames)
        self.evolution_var2 = np.zeros(self.number_frames)
        self.bg1 = np.array([])
        self.bg2 = np.array([])
    def set_background_line(self,distance,middle):
        """
        Set background of the video
        Returns:

        """
        l=0
        ret, next_frame = self.background_vidcap.read()
        gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

        frame_concat_line1 = gray[:, middle + distance:middle + distance + 1].copy()
        frame_concat_line2 = gray[:, middle - distance:middle - distance + 1].copy()
        while True :
            ret, next_frame = self.background_vidcap.read()
            if ret and l < 100:
                gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
                line1 = gray[:, middle + distance:middle + distance + 1]
                line2 = gray[:, middle - distance:middle - distance + 1]
                frame_concat_line1 = np.concatenate((frame_concat_line1, line1), axis=1)
                frame_concat_line2 = np.concatenate((frame_concat_line2, line2), axis=1)
                l+=1
            else :
                break
        self.bg1 = np.median(frame_concat_line1, axis=1)
        self.bg2 = np.median(frame_concat_line2, axis=1)



    def background_subtraction(self,gray,num_line):
        """
        Substract background from the video
        Args:
            gray: gray frame

        Returns:
            gray frame without background
        """
        self.bg1 = self.bg1.reshape((self.bg1.shape[0], 1))
        self.bg2 = self.bg2.reshape((self.bg2.shape[0], 1))

        if num_line == 1 :
            return np.where((gray - self.bg1) < 0, 0, gray - self.bg1).astype(np.uint8)
        else :
            return np.where((gray - self.bg2) < 0, 0, gray - self.bg2).astype(np.uint8)
