"""
   file: fish_detection.py
   subject : detect fish
   Author : AVS7
   Creation : 7/11/2022
   Last Update : 7/11/2022
   Update Note:
        7/11/2022 : Creation


"""
import os

import cv2 as cv
import numpy as np
import glob
from video import video as vd

THRESHOLD = 340


def fish_detection(video):
    """
    detect fish

    Args:
        frames: list of frames

    Returns:

    """
    define_sequence(video)  # specific for baseline v0.1




def define_sequence(video):
    """
    define sequence for lines 1 and 2
    Args:
        video: video object

    Returns:
    """
    video.set_lines()  # set lines 1 and 2
    sequence = np.array([[0,0]])
    k=0
    for frame in video.gray_frames:  # get all frames in video

        hist1 = cv.calcHist([video.line1[k]], [0], None, [256], [0, 256])  # get histogram of line 1
        hist2 = cv.calcHist([video.line2[k]], [0], None, [256], [0, 256])  # get histogram of line 2

        var1 = np.var(hist1)  # get variance of line 1
        var2 = np.var(hist2)  # get variance of line 2
        if var1 < THRESHOLD:  # if variance of line 1 is lower than threshold
            if var2 < THRESHOLD:  # if variance of line 2 is lower than threshold
                if (sequence[-1] != [1, 1]).any():  # if last sequence is not [1,1]
                    sequence = np.concatenate((sequence,np.array([[1,1]])),axis=0)  # add [1,1] to sequence
            else:
                if (sequence[-1] != [0, 1]).any():  # if last sequence is not [0,1]
                    sequence = np.append(sequence, np.array([[0, 1]]),axis=0)  # add [0,1] to sequence
        else:  # if variance of line 1 is higher than threshold
            if var2 < THRESHOLD:  # if variance of line 2 is lower than threshold
                if (sequence[-1] != [1, 0]).any(): # if last sequence is not [1,0]
                    sequence = np.append(sequence, np.array([[1, 0]]),axis=0)

            else:
                if (sequence[-1] != [0, 0]).any(): # if last sequence is not [0,0]
                    sequence = np.append(sequence, np.array([[0, 0]]),axis=0)

        k+=1

    video.sequence = sequence

'''
path = 'C:/Users/julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/training/*.mp4'

number_video = 0
list_videos = np.array([], dtype=object)
for vid in glob.glob(path):  # get all videos in folder
    if number_video < 2:
        video = vd(vid, number_video, os.path.basename(vid))  # create video object
        list_videos = np.append(list_videos, video)  # add video to list
        number_video += 1  # increment video number
#fish_detection(list_videos)  # detect fish'''