import glob
import os
import time

import numpy as np

from background_subtraction import background_subtraction
from counting_fish import counting_fish
from export_json import export_json, initialize_json
from fish_detection import fish_detection
from fish_direction import fish_direction
from video import video as vd
from evaluation import evaluate_frame_time, evaluate_frame_count

"""
    file:   main.py
    subject:    call of functions and setting parameters
    Author:  AVS7
    
"""

if __name__ == '__main__':
    # set parameters

    path = 'C:/Users/\julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/training/*.mp4'
    file_result = 'results.json'  # file to save results
    if os.path.exists(file_result):  # if file exists, delete it
        os.remove(file_result)

    list_vid = None  # list of videos
    list_name_videos = np.empty(0)  # list of names of videos
    start = time.time()  # start time

    for vid in glob.glob(path):  # for each video in the path
        list_vid = np.append(list_vid, vid)  # add video to list
        list_name_videos = np.append(list_name_videos, os.path.basename(vid))  # add name of video to list
    k = 0
    print("--------------------------------------------------\nStarting --------------------------------------------------")

    initialize_json(file_result)  # initialize json file

    for vid in list_vid:  # for each video in the list
        if vid is not None:  # if video is not empty
            video = vd(vid, list_name_videos[k], video_number=k)  # create video object
            k += 1
            print("---------------------------------------------\n Evaluating video number : " + str(
                video.num_video) + "---Video : " + video.name + "---------------------------------------------")

            print("Subtracting background ...")  # subtract background
            background_subtraction(video,method='median') # median background substraction

            print("Detecting fish ...")
            fish_detection(video) # detect fish

            print("Detecting fish direction ...")
            fish_direction(video)   # detect fish direction
            print('Counting fish ...')
            counting_fish(video)   # count fish
            print('Exporting results to json file ...')
            export_json(video, path, file_result) # export results to json file

            video.vidcap.release() # release memory
    print("--------------------------------------------------\nEnd  --------------------------------------------------Time :" + str(time.time() - start))
    # Compute counting accuracy
    print("Computing counting accuracy ...")
    evaluate_frame_count()
    # Compute entering and exit frame precision
    evaluate_frame_time()
    print("Computing entering and exit frame precision ...")
