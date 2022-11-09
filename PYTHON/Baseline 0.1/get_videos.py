"""
   file: background_subtraction.py
   subject : get frames from folder
   Author : AVS7
   Creation : 7/11/2022
   Last Update : 7/11/2022
   Update Note:
        7/11/2022 : Creation


"""
import glob
import os.path

import numpy as np
from numpy import ndarray

from video import video as vd


def get_videos(path, number_of_videos):
    """
    get frames from folder
    Args:
        path: path to folder containing videos
        number_of_videos: number of videos to get
    Returns:

    """
    number_video = 0
    list_videos = np.array([], dtype=object)
    for vid in glob.glob(path):  # get all videos in folder

        if number_video < number_of_videos:
            video = vd(vid, number_video, os.path.basename(vid))  # create video object
            list_videos = np.append(list_videos, video)  # add video to list
            number_video += 1  # increment video number

    return list_videos
