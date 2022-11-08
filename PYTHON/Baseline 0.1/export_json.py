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


def export_json(list_video, path, file_name):
    """
    export json
    Args:
        list_video: list of videos
        file_name: name of the file

    Returns:
        file_name : name of the file
    """
    # TODO : doesn't work if file doesn't exist or if file is empty

    for video in list_video:
        # export to json file
        with open(file_name, 'a') as outfile:
            data = {
                video.name: {'fish_count_frames': video.fish_count_frames.tolist(), 'enter_frame': video.enter_frames_numbers.tolist(),
                             'exit_frame': video.exit_frames_numbers.tolist(), 'fish_count': video.count_fish}}

            json.dump(data, outfile, default=lambda o: o.__dict__, indent=4)
    return file_name





# test function


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

export_json(list_videos, ' ', 'results')
