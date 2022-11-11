"""
   file: export_json.py
   subject : count fish
   Author : AVS7
   Creation : 7/11/2022
   Last Update : 7/11/2022
   Update Note:
        7/11/2022 : Creation


"""
import glob
import json
import os

import numpy as np

from video import video as vd


def export_json(video, path, file_name):
    """
    export json
    Args:

        list_video: list of videos
        path: path to folder containing videos
        file_name: name of the file

    Returns:

    """
    #TODO : Clear file at the beginning
    with open(file_name, 'a') as outfile:
        if video.enter_frames_numbers.size != 0:
            enter_frames_numbers = video.enter_frames_numbers.tolist()
        else:
            enter_frames_numbers = []
        if video.exit_frames_numbers.size != 0:
            exit_frames_numbers = video.exit_frames_numbers.tolist()
        else:
            exit_frames_numbers = []
        if  video.fish_count_frames.size != 0:
            fish_count_frames =  video.fish_count_frames.tolist()
        else:
            fish_count_frames = []
        data = {
            video.name: {'enter_frame': enter_frames_numbers,
                         'exit_frame': exit_frames_numbers, 'fish_count': video.count_fish}}

        json.dump(data, outfile,indent=3)
        outfile.write(',\n')






# test function

'''
path = 'C:/Donnees/IMT Atlantique/TC/AAU/Semestre/Project Local/2 - Technical study/Datas/Baseline_videos_mp4/Training_data/*.mp4'
number_video = 0
list_videos = np.array([], dtype=object)
for vid in glob.glob(path):
    while number_video < 2:
        video = vd(vid, number_video, os.path.basename(vid))
        list_videos = np.append(list_videos, video)
        number_video += 1
list_videos[0].count_fish = 2
list_videos[1].count_fish = 3
list_videos[0].exit_frames_numbers = np.array([2, 3])
list_videos[1].exit_frames_number = np.array([0, 1, 2])
list_videos[0].enter_frames_numbers = np.array([5, 6])
list_videos[1].enter_frames_number = np.array([8, 9])
list_videos[0].fish_count_frames = np.array([None, None])
list_videos[1].fish_count_frames = np.array([None, None])
'''
#export_json(list_videos, ' ', 'results')
