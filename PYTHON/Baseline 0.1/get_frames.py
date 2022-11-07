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

import numpy as np
from numpy import ndarray

from video import video as vd


def get_frames(path) -> ndarray:
    """
    get frames from folder
    Args:
        path: path to folder containing videos

    Returns:
        list_frames : list of videos
    """
    number_video = 0
    list_videos = np.array([], dtype=object)
    for vid in glob.glob(path):
        video = vd.video(vid, number_video)
        list_videos = np.append(list_videos, video)
        number_video += 1
    return list_videos
