"""

    file: background_subtraction.py
    subject: Background
    subtraction of frames
    Author: AVS7

"""
import glob

import cv2 as cv
import numpy as np
import gc
from video import video as vd


def background_subtraction(video, method='None', entire_frame=False, save=True):
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
        median_background_subtraction(video, entire_frame, save)
    elif method == 'mean':
        mean_background_subtraction(video, entire_frame, save)
    elif method == 'KNN':
        background_subtraction_KNN(video, method, entire_frame)


def median_background_subtraction(video, entire_frame, save):
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
    return video.line1, video.line2


def mean_background_subtraction(video, entire_frame, save):
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
    # video.bg1 = cv.imread("../Background_frames/median_background_line1.png", cv.IMREAD_GRAYSCALE)
    # video.bg2 = cv.imread("../Background_frames/median_background_line2.png", cv.IMREAD_GRAYSCALE)
    video.bg1 = video.bg1.reshape((video.bg1.shape[0], 1))  # reshape bg1
    video.bg2 = video.bg2.reshape((video.bg2.shape[0], 1))  # reshape bg2


    video.line1 = np.where((video.line1 - video.bg1) < 0, 0, video.line1 - video.bg1).astype(np.uint8)  # subtract background from line 1
    video.line2 = np.where((video.line2 - video.bg2) < 0, 0, video.line2 - video.bg2).astype(np.uint8)  # subtract background from line 2
    if entire_frame:
        print('entire frame')
        background = np.mean(video.gray_frames[:, :, :], axis=0)
        video.frames_subtract = np.where((video.gray_frames - background) < 0, 0, video.gray_frames - background).astype(np.uint8)
        print('entire frame done')


def background_subtraction_KNN(video, entire_frame, save):
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


def export_background_lines(path):
    """
    Export median background
    """
    line1_background = np.array([])
    line2_background = np.array([])
    k = 0
    for vid in glob.glob(path):


        video = vd(vid, k)
        video.set_lines()
        if k == 0:
            line1_background = video.line1
            line2_background = video.line2
        else :
            line1_background = np.concatenate((line1_background, video.line1), axis=0)
            line2_background = np.concatenate((line2_background, video.line2), axis=0)
        k += 1
        print(k)
        del video

    bg1_median = np.median(line1_background, axis=0)
    bg2_median = np.median(line2_background, axis=0)
    bg1_mean = np.mean(line1_background, axis=0)
    bg2_mean = np.mean(line2_background, axis=0)
    cv.imwrite("../Background_frames/median_background_line1.png", bg1_median)
    cv.imwrite("../Background_frames/median_background_line2.png", bg2_median)
    cv.imwrite("../Background_frames/mean_background_line1.png", bg1_mean)
    cv.imwrite("../Background_frames/mean_background_line2.png", bg2_mean)

if __name__ == '__main__':
    path = 'C:/Users/\julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents (1)/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/new_split/train/*.mp4'
    export_background_lines(path)
