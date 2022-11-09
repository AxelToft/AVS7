import argparse
from export_json import export_json
from background_subtraction import background_subtraction
from counting_fish import counting_fish
from fish_direction import fish_direction
from fish_detection import fish_detection
import glob
import numpy as np
import os
from video import video as vd
"""
    file:   main.py
    subject:    call of functions and setting parameters
    Author:  AVS7
    Creation:    7/11/2022
    Last Update: 7/11/2022
    Update Note:
        7/11/2022 : Creation

"""

if __name__ == '__main__':
    # set parameters
    path = 'C:/Users/julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/training/*.mp4'
    file_result ='results.json'
    number_of_videos=5
    '''path = argparse.ArgumentParser()
    file_result = argparse.ArgumentParser()'''

    # set frames for each videos-> video.frames
    list_vid = None
    list_name_videos = None
    for vid in glob.glob(path):

        list_vid = np.append(list_vid, vid)
        list_name_videos = np.append(list_name_videos, os.path.basename(vid))
    k=0
    print("--------------------------------------------------\nStarting "
          "...--------------------------------------------------")

    for path in vid:
        video = vd(vid, list_name_videos[k],video_number=k)
        k+=1
        print("---------------------------------------------\n Evaluating video number : " + str(video.num_video) + "---------------------------------------------")
        # subtract background of frames -> video.foreground_frames
        background_subtraction(video)

        # we look for frames where fish are detected -> video.numbers_frames_fish_detected
        fish_detection(video)

        # for these frames, we look for  the direction of the fish -> video.frames_direction
        fish_direction(video)

        # count fish and set values for each video -> video.count_fish
        counting_fish(video)

        # export results to json file from list videos results

        export_json(video, path, file_result)
    print("--------------------------------------------------\nEnd --------------------------------------------------")