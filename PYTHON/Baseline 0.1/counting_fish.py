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
    counting_fish_v01(video) # baseline 0.1


def counting_fish_v01(video):
    #TODO : study larger sequence
    i = 0
    for lines_values in video.sequence[0]:
        if i<len(video.sequence[0])-4:
            if (lines_values == [0,1]).all():
                video.enter_frames_numbers = np.append(video.enter_frames_numbers, video.sequence[1][i])
                if (video.sequence[0][i + 1] == [1, 1]).all():
                    if (video.sequence[0][i + 2] == [1, 0]).all():
                        if (video.sequence[0][i + 3] == [0, 0]).all():
                            video.count_fish += 1
                            video.exit_frames_numbers = np.append(video.exit_frames_numbers, video.sequence[1][i])
                            video.fish_count_frames = np.append(video.fish_count_frames,video.sequence[1][i])
            if (lines_values == [0,0]).all():
                if (video.sequence[0][i + 1] == [1, 0]).all():

                    if (video.sequence[0][i + 2] == [1, 1]).all():
                        if (video.sequence[0][i + 3] == [0, 1]).all():
                            if (video.sequence[0][i + 4] == [0, 0]).all():
                                video.fish_count_frames = np.append(video.fish_count_frames,video.sequence[1][i])
                                video.count_fish -= 1

        i += 1


# test function :

'''
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
list_videos[1].sequence = np.array([[1, 1], [0, 1], [0, 0]])'''
#counting_fish(list_videos)