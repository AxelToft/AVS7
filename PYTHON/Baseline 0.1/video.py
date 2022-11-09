import glob
import cv2 as cv
import numpy as np

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
    def __init__(self, video, num_video, name_video):
        """

        Args:
            video: video
            num_video: adding number of the video
        """

        # gereral attributes
        self.name = name_video  # video's name
        self.vidcap = cv.VideoCapture(video)  # vidcap is the video
        self.number_frames = int(self.vidcap.get(cv.CAP_PROP_FRAME_COUNT))  # number of frames in the video

        self.num_video = num_video  # number of the video
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
        self.background1 = np.array([])
        self.background2 = np.array([])
        self.line1 = np.zeros((self.height, 1), dtype=np.uint8)
        self.line2 = np.zeros((self.height, 1), dtype=np.uint8)
        self.background_vidcap = cv.VideoCapture(video)
        self.evolution_var1 = np.zeros(self.number_frames)
        self.evolution_var2 = np.zeros(self.number_frames)
        self.sequence = None


        k = 0
        while ret :
            self.frames[k] = next_frame
            self.gray_frames[k] = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
            ret, next_frame = self.vidcap.read()
            k+= 1
        self.vidcap.release()


    def set_background_line(self, distance, middle):
        """
        Set background of the video
        Returns:

        """
        l = 0
        ret, next_frame = self.background_vidcap.read()
        gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

        frame_concat_line1 = gray[:, middle + distance:middle + distance + 1].copy()
        frame_concat_line2 = gray[:, middle - distance:middle - distance + 1].copy()
        while True:
            ret, next_frame = self.background_vidcap.read()
            if ret and l < 100:
                gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
                line1 = gray[:, middle + distance:middle + distance + 1]
                line2 = gray[:, middle - distance:middle - distance + 1]
                frame_concat_line1 = np.concatenate((frame_concat_line1, line1), axis=1)
                frame_concat_line2 = np.concatenate((frame_concat_line2, line2), axis=1)
                l += 1
            else:
                break
        self.bg1 = np.median(frame_concat_line1, axis=1)
        self.bg2 = np.median(frame_concat_line2, axis=1)

    def background_subtraction(self, gray, num_line):
        """
        Substract background from the video
        Args:
            gray: gray frame

        Returns:
            gray frame without background
        """
        self.bg1 = self.bg1.reshape((self.bg1.shape[0], 1))
        self.bg2 = self.bg2.reshape((self.bg2.shape[0], 1))

        if num_line == 1:
            return np.where((gray - self.bg1) < 0, 0, gray - self.bg1).astype(np.uint8)
        else:
            return np.where((gray - self.bg2) < 0, 0, gray - self.bg2).astype(np.uint8)

    def set_lines(self,distance_from_middle = 50):
        #TODO : set lines for background, or normale frames
        #[:, middle + distance: middle + distance + 1]
        self.line1 = self.gray_frames[:, self.width // 2 + distance_from_middle:self.width // 2 + distance_from_middle + 1].copy()
        self.line2 = self.gray_frames[:, self.width // 2 - distance_from_middle:self.width // 2 - distance_from_middle + 1].copy()

