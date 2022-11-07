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

import numpy as np

from video import video as vd


def export_json(list_video, path, file_name):
    """
    export json
    Args:
        list_video: list of videoz
        file_name: name of the file

    Returns:
        file_name : name of the file
    """

    return None


# test function

path = 'C:/Donnees/IMT Atlantique/TC/AAU/Semestre/Project Local/2 - Technical study/Datas/Baseline_videos_mp4/Training_data/*.mp4'
number_video = 0
for vid in glob.glob(path):
    while number_video < 2:
        video = vd.video(vid, number_video)
        list_videos = np.append(list_videos, video)
        number_video += 1
list_videos[0].count_fish = 2
list_videos[1].count_fish = 3
list_videos[0].frames_count = np.array([2, 3])
list_videos[1].frames_count = np.array([0, 1, 2])

export_json(list_videos, ' ', 'results')
