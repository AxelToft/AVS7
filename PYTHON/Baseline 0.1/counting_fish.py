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

import numpy as np

from video import video as vd
def counting_fish(list_vn) -> int:
    """
    count fish
    Args:
        direction: direction of fish

    Returns:
        direction : 1 if fish is going right, -1 if fish is going left
    """
    return None

# test function :

path = 'C:/Donnees/IMT Atlantique/TC/AAU/Semestre/Project Local/2 - Technical study/Datas/Baseline_videos_mp4/Training_data/*.mp4'
number_video = 0
for vid in glob.glob(path):
    while number_video < 2:
        video = vd.video(vid, number_video)
        list_videos = np.append(list_videos, video)
        number_video += 1

