import glob
import os
import time

import numpy as np

from background_subtraction import background_subtraction
from counting_fish import counting_fish
from evaluation import evaluate_frame_count
from export_json import export_json, initialize_json
from fish_detection import fish_detection
from video import video as vd

"""
    file:   main.py
    subject:    call of functions and setting parameters
    Author:  AVS7
    
"""


def baseline(distance=300, threshold=1000, path=None, file_results='results.json'):
    start = time.perf_counter()
    if path is None:
        path = 'C:/Users/\julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/new_split/train/*.mp4'

    list_vid = []  # list of videos
    list_name_videos = np.empty(0)  # list of names of videos
    start = time.perf_counter()  # start time
    for vid in glob.glob(path):  # for each video in the path
        list_vid = np.append(list_vid, vid)  # add video to list
        list_name_videos = np.append(list_name_videos, os.path.basename(vid))  # add name of video to list
    print(f"Time for computing initalisation: {time.perf_counter() - start}")
    current_time = time.perf_counter()
    print("--------------------------------------------------\nStarting --------------------------------------------------")

    initialize_json(file_results)  # initialize json file

    for k, (vid, name_video) in enumerate(zip(list_vid, list_name_videos)):  # for each video in the list
        if vid is not None:  # if video is not empty
            video = vd(vid, name_video, k)  # create video object
            video.DISTANCE_FROM_MIDDLE = distance
            print("---------------------------------------------\n Evaluating video number : " + str(
                video.num_video) + "---Video : " + video.name + "---------------------------------------------")
            print(f"Time for initialize videos: {time.perf_counter() - current_time}")

            print("Subtracting background ...")  # subtract background
            current_time = time.perf_counter()

            background_subtraction(video, method='median')  # median background subtraction
            print(f"Time for computing bgs: {time.perf_counter() - current_time}")
            current_time = time.perf_counter()

            print("Detecting fish ...")
            fish_detection(video, threshold)  # detect fish
            print(f"Time for computing detection: {time.perf_counter() - current_time}")
            current_time = time.perf_counter()

            # print("Detecting fish direction ...")
            # fish_direction(video)  # detect fish direction

            print('Counting fish ...')

            counting_fish(video)  # count fish
            print(f"Time for computing counting: {time.perf_counter() -current_time}")
            current_time = time.perf_counter()

            print('Exporting results to json file ...')
            export_json(video, file_results)  # export results to json file
            print(f"Time for computing export: {time.perf_counter() - current_time}")
            current_time = time.perf_counter()

            video.vidcap.release()  # release memory
    # Compute counting accuracy
    print(f"Computing counting accuracy ... Time for computing accuracy: {time.perf_counter() -current_time}")
    accuracy = evaluate_frame_count(file_results)
    # Compute entering and exit frame precision
    # print("Computing entering and exit frame precision ...")

    print("--------------------------------------------------\nEnd  --------------------------------------------------Time :" + str(time.perf_counter() - current_time))
    return accuracy
