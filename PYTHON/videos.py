import glob
import cv2 as cv
import numpy as np

import video as vd
'''
   Object : List of videos
   Autor : Julien Roussel Galle
   Creation : 1/11/2022
   Last Update : 1/11/2022
   Update Note: 
        1/11/2022 : Dosctrings
'''


class videos:
    def __init__(self, origin):
        """
        List of videos
        Args:
            origin: Full path to get videos
        """
        self.count_fish = 0
        self.list_videos = np.array([], dtype=object)
        #self.actual_count = 0
        self.k = 0
        num_video = 0
        for vid in glob.glob(origin):
            print("oui")
            while num_video < 5:
                video = vd.video(vid, num_video)
                self.list_videos = np.append(self.list_videos, video)

                num_video += 1

