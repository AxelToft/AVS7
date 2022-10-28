import glob

import numpy as np


class videos:
    def __init__(self,origin):
        self.count_fish = 0
        self.list_videos = np.array([])
        self.add_videos(origin)
        self.actual_count= 0
        self.k =0
    def add_videos(self,origin):
        for video in glob.glob("../Datas/Baseline_videos_mp4/Training_data/*.mp4"):
            self.list_videos = np.append(self.list_videos,video)
