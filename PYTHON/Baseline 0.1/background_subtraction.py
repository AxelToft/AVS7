"""

    file: background_subtraction.py
    subject: Background
    subtraction of frames
    Author: AVS7

"""
import cv2 as cv
import numpy as np


def background_subtraction(video, method='None', entire_frame=False):
    """

    Args:
        entire_frame:  if True, background subtraction is applied to entire frame
        method:  method of background subtraction
        video : video

    Returns:

    """
    if method == 'None':
        video.set_lines()
    elif method == 'median':
        median_background_subtraction(video, entire_frame)
    elif method == 'mean':
        mean_background_subtraction(video, entire_frame)
    elif method == 'KNN':
        background_subtraction_KNN(video, method, entire_frame)


def median_background_subtraction(video, entire_frame):
    """
    Subtract background from the video
    Args:
        entire_frame:  if True, background subtraction is applied to entire frame
        video:  video

    Returns:
        gray frame without background
    """
    video.set_lines()  # set lines to all frames in video
    video.bg1 = np.median(video.gray_frames[:, 250:1440, video.width // 2 + video.DISTANCE_FROM_MIDDLE:video.width // 2 + video.DISTANCE_FROM_MIDDLE + 1],
                          axis=0)  # get median of line 1
    video.bg2 = np.median(video.gray_frames[:, 250:1440, video.width // 2 - video.DISTANCE_FROM_MIDDLE:video.width // 2 - video.DISTANCE_FROM_MIDDLE + 1],
                          axis=0)  # get median of line 2
    video.bg1 = video.bg1.reshape((video.bg1.shape[0], 1))  # reshape bg1
    video.bg2 = video.bg2.reshape((video.bg2.shape[0], 1))  # reshape bg2

    video.line1 = np.where((video.line1 - video.bg1) < 0, 0, video.line1 - video.bg1).astype(np.uint8)  # subtract background from line 1
    video.line2 = np.where((video.line2 - video.bg2) < 0, 0, video.line2 - video.bg2).astype(np.uint8)  # subtract background from line 2
    if entire_frame:
        background = np.median(video.gray_frames[:, :, :], axis=0)
        video.frames_subtract = np.where((video.gray_frames - background) < 0, 0, video.gray_frames - background).astype(np.uint8)


def mean_background_subtraction(video, entire_frame):
    """
    Substract background from the video
    Args:
        entire_frame:  if True, background subtraction is applied to entire frame
        video:  video

    """
    video.set_lines()  # set lines to all frames in video
    video.bg1 = np.mean(video.gray_frames[:, 250:1440, video.width // 2 + video.DISTANCE_FROM_MIDDLE:video.width // 2 + video.DISTANCE_FROM_MIDDLE + 1],
                        axis=0)  # get median of line 1
    video.bg2 = np.mean(video.gray_frames[:, 250:1440, video.width // 2 - video.DISTANCE_FROM_MIDDLE:video.width // 2 - video.DISTANCE_FROM_MIDDLE + 1],
                        axis=0)  # get median of line 2
    video.bg1 = video.bg1.reshape((video.bg1.shape[0], 1))  # reshape bg1
    video.bg2 = video.bg2.reshape((video.bg2.shape[0], 1))  # reshape bg2

    video.line1 = np.where((video.line1 - video.bg1) < 0, 0, video.line1 - video.bg1).astype(np.uint8)  # subtract background from line 1
    video.line2 = np.where((video.line2 - video.bg2) < 0, 0, video.line2 - video.bg2).astype(np.uint8)  # subtract background from line 2
    if entire_frame:
        print('entire frame')
        background = np.mean(video.gray_frames[:, :, :], axis=0)
        video.frames_subtract = np.where((video.gray_frames - background) < 0, 0, video.gray_frames - background).astype(np.uint8)
        print('entire frame done')


def background_subtraction_KNN(video, method='None', entire_frame=False):
    """

    Args:
        video : video

    Returns:

    """
    backSub = cv.createBackgroundSubtractorKNN()
    video.set_lines()  # set lines to all frames in video
    for k, frame in enumerate(video.gray_frames):
        video.frames_subtract[k] = backSub.apply(frame)
    video.line1 = video.frames_subtract[:, 250:1440, video.width // 2 + video.DISTANCE_FROM_MIDDLE:video.width // 2 + video.DISTANCE_FROM_MIDDLE + 1]
    video.line2 = video.frames_subtract[:, 250:1440, video.width // 2 - video.DISTANCE_FROM_MIDDLE:video.width // 2 - video.DISTANCE_FROM_MIDDLE + 1]
