"""
   file: fish_detection.py
   subject : detect fish
   Author : AVS7
   Creation : 7/11/2022
   Last Update : 7/11/2022
   Update Note:
        7/11/2022 : Creation


"""
import glob
import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from video import video as vd
from background_subtraction import *
THRESHOLD = 1000


def fish_detection(video):
    """
    detect fish

    Args:
        video : video

    Returns:

    """
    define_sequence(video)  # specific for baseline v0.1


def define_sequence(video):
    """
    define sequence for lines 1 and 2 in accordance with baseline v0.1
    Args:
        video: video object

    Returns:
    """
    sequence = np.array([[0, 0]]) # TODO : change initialisation
    list_frame_changing = []
    k = 0
    for line1, line2,gray in zip(video.line1, video.line2,video.gray_frames):
    #for frame in video.gray_frames:  # get all frames in video


        hist1 = cv.calcHist([line1], [0], None, [256], [0, 256])  # get histogram of line 1
        hist2 = cv.calcHist([line2], [0], None, [256], [0, 256])  # get histogram of line 2

        var1 = np.var(hist1)  # get variance of line 1
        var2 = np.var(hist2)  # get variance of line 2
        '''print("------",var1)
        print(var2)
        cv.line(gray, (video.width//2 + video.DISTANCE_FROM_MIDDLE +1, 0), (video.width//2 + video.DISTANCE_FROM_MIDDLE +1, video.height), (0, 0, 255), 1)  # draw line 1
        cv.line(gray, (video.width // 2 - video.DISTANCE_FROM_MIDDLE + 1, 0), (video.width // 2 - video.DISTANCE_FROM_MIDDLE + 1, video.height), (0, 0, 255), 1)  # draw line 1
        cv.resize(gray, (0, 0), fx=0.5, fy=0.1)
        cv.imshow("frame", gray)
        cv.resize(gray, (0, 0), fx=0.5, fy=0.1)
        cv.waitKey(30)'''

        if var1 < THRESHOLD:  # if variance of line 1 is lower than threshold
            if var2 < THRESHOLD:  # if variance of line 2 is lower than threshold
                if (sequence[-1] != [1, 1]).any():  # if last sequence is not [1,1]
                    sequence = np.concatenate((sequence, np.array([[1, 1]])), axis=0)  # add [1,1] to sequence
                    list_frame_changing = np.append(list_frame_changing, k)
            else:
                if (sequence[-1] != [0, 1]).any():  # if last sequence is not [0,1]
                    sequence = np.append(sequence, np.array([[0, 1]]), axis=0)  # add [0,1] to sequence
                    list_frame_changing = np.append(list_frame_changing, k)
        else:  # if variance of line 1 is higher than threshold
            if var2 < THRESHOLD:  # if variance of line 2 is lower than threshold
                if (sequence[-1] != [1, 0]).any():  # if last sequence is not [1,0]
                    sequence = np.append(sequence, np.array([[1, 0]]), axis=0)
                    list_frame_changing = np.append(list_frame_changing, k)

            else:
                if (sequence[-1] != [0, 0]).any():  # if last sequence is not [0,0]
                    sequence = np.append(sequence, np.array([[0, 0]]), axis=0)
                    list_frame_changing = np.append(list_frame_changing, k)

        k += 1

    video.sequence = np.array([sequence, list_frame_changing])

    print("sequence : ", video.sequence)



path = 'C:/Users/julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/training/*.mp4'

number_video = 0
list_videos = np.array([], dtype=object)
for vid in glob.glob(path):  # get all videos in folder
    if number_video < 2:
        video = vd(vid, number_video, os.path.basename(vid))  # create video object
        background_subtraction(video)  # set background
        fish_detection(video)  # detect fish
