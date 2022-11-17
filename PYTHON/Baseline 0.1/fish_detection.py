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

from video import video as vd
from background_subtraction import *




def fish_detection(video,threshold):
    """
    detect fish

    Args:
        video : video

    Returns:

    """

    define_sequence(video,threshold)  # specific for baseline v0.1


def define_sequence(video,threshold):
    """
    define sequence for lines 1 and 2 in accordance with baseline v0.1
    Args:
        threshold: threshold for histogram variance
        video: video object

    Returns:
    """
    sequence = np.array([[None, None]])  # TODO : change initialisation
    list_frame_changing = []  # list of frames where the sequence is changing
    for k, (line1, line2) in enumerate(zip(video.line1, video.line2)):  # for each line

        hist1 = cv.calcHist([line1], [0], None, [256], [0, 256])  # get histogram of line 1
        hist2 = cv.calcHist([line2], [0], None, [256], [0, 256])  # get histogram of line 2

        var1 = np.var(hist1)  # get variance of line 1
        var2 = np.var(hist2)  # get variance of line 2

        if var1 > threshold:  # if variance of line 1 is greater than threshold
            if var2 > threshold:  # if variance of line 2 is greater than threshold
                if (sequence[-1] != [1, 1]).any():  # if last sequence is not [1,1]
                    sequence = np.concatenate((sequence, np.array([[1, 1]])), axis=0)  # add [1,1] to sequence
                    list_frame_changing = np.append(list_frame_changing, k)
            else:
                if (sequence[-1] != [0, 1]).any():  # if last sequence is not [0,1]
                    sequence = np.append(sequence, np.array([[0, 1]]), axis=0)  # add [0,1] to sequence
                    list_frame_changing = np.append(list_frame_changing, k)
        else:  # if variance of line 1 is higher than threshold
            if var2 > threshold:  # if variance of line 2 is greater than threshold
                if (sequence[-1] != [1, 0]).any():  # if last sequence is not [1,0]
                    sequence = np.append(sequence, np.array([[1, 0]]), axis=0)
                    list_frame_changing = np.append(list_frame_changing, k)

            else:
                if (sequence[-1] != [0, 0]).any():  # if last sequence is not [0,0]
                    sequence = np.append(sequence, np.array([[0, 0]]), axis=0)
                    list_frame_changing = np.append(list_frame_changing, k)

    video.sequence = np.array([sequence, list_frame_changing],dtype=object)
