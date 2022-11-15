"""

    file: background_subtraction.py
    subject: Background
    subtraction of frames
    Author: AVS7

"""
import numpy as np


def background_subtraction(video,method='None'):
    """

    Args:
        video : video

    Returns:

    """
    if method == 'None':
        video.set_lines()
    elif method == 'median':
        median_background_subtraction(video)


def median_background_subtraction(video):
    """
    Substract background from the video
    Args:
        gray: gray frame

    Returns:
        gray frame without background
    """
    video.set_lines() # set lines to all frames in video
    video.bg1 = np.median(video.gray_frames[:, :, video.width // 2 + video.DISTANCE_FROM_MIDDLE:video.width // 2 + video.DISTANCE_FROM_MIDDLE + 1],
                          axis=0) # get median of line 1
    video.bg2 = np.median(video.gray_frames[:, :, video.width // 2 - video.DISTANCE_FROM_MIDDLE:video.width // 2 - video.DISTANCE_FROM_MIDDLE + 1], axis=0) # get median of line 2

    video.bg1 = video.bg1.reshape((video.bg1.shape[0], 1)) # reshape bg1
    video.bg2 = video.bg2.reshape((video.bg2.shape[0], 1)) # reshape bg2

    video.line1 = np.where((video.line1 - video.bg1) < 0, 0, video.line1 - video.bg1).astype(np.uint8)  # subtract background from line 1
    video.line2 = np.where((video.line2 - video.bg2) < 0, 0, video.line1 - video.bg2).astype(np.uint8) # subtract background from line 2
