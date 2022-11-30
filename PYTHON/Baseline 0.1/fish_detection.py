"""
   file: fish_detection.py
   subject : detect fish
   Author : AVS7
   Creation : 7/11/2022
   Last Update : 7/11/2022
   Update Note:
        7/11/2022 : Creation


"""
import numpy as np

from background_subtraction import *


def fish_detection(video, threshold):
    """
    detect fish

    Args:
        threshold:  threshold for histogram variance
        video : video

    Returns:

    """

    define_sequence(video, threshold)  # specific for baseline v0.1


def define_sequence(video, threshold):
    """
    define sequence for lines 1 and 2 in accordance with baseline v0.1
    Args:
        threshold: threshold for histogram variance
        video: video object

    Returns:
    """
    # TODO remove test
    sum = 0

    # print("TEMP :",np.var(cv.calcHist([video.line1[0]], [0], None, [256], [0, 256])))
    sequence = np.array([[None, None]])
    list_frame_changing = []  # list of frames where the sequence is changing
    for k, (line1, line2) in enumerate(zip(video.line1, video.line2)):  # for each line

        hist1 = cv.calcHist([line1], [0], None, [256], [0, 256])  # get histogram of line 1
        hist2 = cv.calcHist([line2], [0], None, [256], [0, 256])  # get histogram of line 2
        var1, var2 = np.var(hist1), np.var(hist2)  # get variance of line 1 and line 2

        video.evolution_var1[k], video.evolution_var2[k] = var1, var2  # save variance of line 1 and line 2

        if var1 > threshold:  # if variance of line 1 is greater than threshold
            if var2 > threshold:  # if variance of line 2 is greater than threshold
                if (sequence[-1] != [1, 1]).any():  # if last sequence is not [1,1]
                    sequence = np.concatenate((sequence, np.array([[1, 1]])), axis=0)  # add [1,1] to sequence
                    list_frame_changing = np.append(list_frame_changing, k)
            else:
                if (sequence[-1] != [0, 1]).any():
                    sequence = np.append(sequence, np.array([[0, 1]]), axis=0)  # add [0,1] to sequence
                    list_frame_changing = np.append(list_frame_changing, k)
        else:
            if var2 > threshold:  # if variance of line 1 is higher than threshold
                if (sequence[-1] != [1, 0]).any():  # if last sequence is not [1,0]
                    sequence = np.append(sequence, np.array([[1, 0]]), axis=0)
                    list_frame_changing = np.append(list_frame_changing, k)
            else:
                if (sequence[-1] != [0, 0]).any():  # if last sequence is not [0,0]
                    sequence = np.append(sequence, np.array([[0, 0]]), axis=0)
                    list_frame_changing = np.append(list_frame_changing, k)

    video.sequence = np.array([sequence, list_frame_changing], dtype=object)


def calcul_treshold():
    path = 'C:/Users/\julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents (1)/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/new_split/train/*.mp4'
    list_vid = []  # list of videos
    for vid in glob.glob(path):  # for each video in the path
        list_vid = np.append(list_vid, vid)  # add video to list
    sum = 0
    number_frames = 0
    max = 0
    min = 10000
    list_max = np.array([])
    list_min = np.array([])
    for k, vid in enumerate(list_vid):
        if vid is not None:
            video = vd(vid, k)
            background_subtraction(video, method='mean', entire_frame=False)
            for k, (line1, line2) in enumerate(zip(video.line1, video.line2)):
                variance =  np.var(cv.calcHist([line1], [0], None, [256], [0, 256]))
                if variance > max :
                    max = variance
                elif variance < min :
                    min = variance
            number_frames += video.number_frames
            print("TEMP2", max, min)
            list_max = np.append(list_max,max)
            list_min = np.append(list_min,min)
            min = 1000
            max = 0

    print("TEMP2",(np.mean(list_max)-np.min(list_min))/4)


if __name__ == '__main__':
    calcul_treshold()
