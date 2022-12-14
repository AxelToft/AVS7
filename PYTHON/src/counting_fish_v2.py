"""
   file: counting_fish.py
   subject : count fish
   Author : AVS7
   Creation : 7/11/2022
   Last Update : 7/11/2022
   Update Note:
        7/11/2022 : Creation


"""
import numpy as np

from video import video as vd


def counting_fish(video):
    """
    count fish
    Args:
        video: video object


    Returns:

    """
    counting_fish_v01(video)  # baseline 0.1


def counting_fish_v01(video):
    stage = 0
    while True :
        if stage == 0:
            if np.array_equal(video.sequence[0][i], [0, 0]) and not mid_spawn: