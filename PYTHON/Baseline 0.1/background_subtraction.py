"""

    file: background_subtraction.py
    subject: Background
    subtraction of frames
    Author: AVS7
    Creation: 7 / 11 / 2022
    Last Update: 7 / 11 / 2022
    Update Note:
        7 / 11 / 2022: Creation
"""
import cv2 as cv
import numpy as np

DISTANCE_FROM_MIDDLE = 50
def background_subtraction(video) :
    """

    Args:
        list_videos : list of videos

    Returns:

    """
    median_background_subtraction(video)






def set_background_line(video):
    """
    Set background of the video
    Returns:

    """
    l = 0
    ret, next_frame = video.background_vidcap.read()
    gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

    frame_concat_line1 = gray[:, video.width // 2 + DISTANCE_FROM_MIDDLE:video.width // 2 + DISTANCE_FROM_MIDDLE + 1].copy()
    frame_concat_line2 = gray[:, video.width // 2 - DISTANCE_FROM_MIDDLE:video.width // 2 - DISTANCE_FROM_MIDDLE + 1].copy()
    while True:
        ret, next_frame = video.background_vidcap.read()
        if ret and l < 100:
            gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
            line1 = gray[:, video.width // 2 + DISTANCE_FROM_MIDDLE:video.width // 2 + DISTANCE_FROM_MIDDLE + 1]
            line2 = gray[:, video.width // 2 - DISTANCE_FROM_MIDDLE:video.width // 2 - DISTANCE_FROM_MIDDLE + 1]
            frame_concat_line1 = np.concatenate((frame_concat_line1, line1), axis=1)
            frame_concat_line2 = np.concatenate((frame_concat_line2, line2), axis=1)
            l += 1
        else:
            break
    video.bg1 = np.median(frame_concat_line1, axis=1)
    video.bg2 = np.median(frame_concat_line2, axis=1)

def median_background_subtraction(video):
    """
    Substract background from the video
    Args:
        gray: gray frame

    Returns:
        gray frame without background
    """

    set_background_line(video)
    gray = video.gray_frames
    video.bg1 = video.bg1.reshape((video.bg1.shape[0], 1))
    video.bg2 = video.bg2.reshape((video.bg2.shape[0], 1))
    video.bg1 = np.where((gray - video.bg1) < 0, 0, gray - video.bg1).astype(np.uint8)
    video.bg2 = np.where((gray - video.bg2) < 0, 0, gray - video.bg2).astype(np.uint8)

