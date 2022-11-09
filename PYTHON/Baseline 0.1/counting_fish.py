"""
   file: counting_fish.py
   subject : count fish
   Author : AVS7
   Creation : 7/11/2022
   Last Update : 7/11/2022
   Update Note:
        7/11/2022 : Creation


"""
import glob
import os

import numpy as np

from video import video as vd


def counting_fish(video):
    """
    count fish
    Args:
       list_videos: list of videos

    Returns:

    """
    #TODO : Define entering and exit frames
    counting_fish_v01() # baseline 0.1


def counting_fish_v01():
    i = 0
    for lines_values in video.sequence:
        if (lines_values == [1, 1]).all():

            if (video.sequence[i + 1] == [1, 0]).all():
                if (video.sequence[i + 2] == [0, 0]).all():
                    video.count_fish += 1

            elif (video.sequence[i + 1] == [0, 1]).all():

                if (video.sequence[i + 2] == [0, 0]).all():
                    video.count_fish -= 1

        i += 1
# test function :


path = 'C:/Users/julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/training/*.mp4'

number_video = 0
list_videos = np.array([], dtype=object)

for vid in glob.glob(path):
    if number_video < 2:
        print(vid)
        video = vd(vid, number_video, os.path.basename(vid))
        list_videos = np.append(list_videos,video)
    number_video   += 1
list_videos[0].sequence = np.array([[1, 1], [1, 0], [0, 0]])
list_videos[1].sequence = np.array([[1, 1], [0, 1], [0, 0]])
#counting_fish(list_videos)