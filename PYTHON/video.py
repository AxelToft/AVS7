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

        """while True :
            ret, next_frame = self.vidcap.read()
            if ret:
                gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
                line1 = gray[:, middle + distance:middle + distance + 1]
                line2 = gray[:, middle - distance:middle - distance + 1]
                self.bg1 =np.append(self.bg1,line1)
                self.bg2 = np.append(self.bg2, line2)
                l+=1
            else :
                break
        self.bg1 = self.bg1[0]
        self.bg2 = self.bg2[0]"""
        ret, next_frame = self.vidcap.read()
        self.bg1 = cv.cvtColor(next_frame[:, middle + distance:middle + distance + 1], cv.COLOR_BGR2GRAY)
        self.bg2 = cv.cvtColor(next_frame[:, middle - distance-5:middle - distance -4], cv.COLOR_BGR2GRAY)


    def background_subtraction(self,gray,num_line):
        """
        Substract background from the video
        Args:
            gray: gray frame

        Returns:
            gray frame without background
        """

        if num_line == 1 :
            return np.abs(gray - self.bg1)
        else :
            return np.abs(gray - self.bg2)
