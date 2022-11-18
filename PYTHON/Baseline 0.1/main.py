import glob
import os
import time

import cv2
import numpy as np

from background_subtraction import background_subtraction
from counting_fish import counting_fish
from evaluation import evaluate_frame_count
from export_json import export_json, initialize_json
from fish_detection import fish_detection
from video import video as vd
from show_video import show_video
"""
    file:   main.py
    subject:    call of functions and setting parameters
    Author:  AVS7
    
"""
np.linspace(1000,0, 11)

def baseline(distance=280, threshold=1000, path=None, file_results='results.json'):
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
    print("--------------------------------------------------\nStarting list of videos --------------------------------------------------")

    initialize_json(file_results)  # initialize json file
    print(f"Time for initialize files: {time.perf_counter() - current_time}")

    for k, (vid, name_video) in enumerate(zip(list_vid, list_name_videos)):  # for each video in the list
        if vid is not None:  # if video is not empty
            video = vd(vid, name_video, k)  # create video object
            video.DISTANCE_FROM_MIDDLE = distance
            print("---------------------------------------------\n Evaluating video number : " + str(
                video.num_video) + "---Video : " + video.name + "---------------------------------------------")

            background_subtraction(video, method='median')  # median background subtraction
            fish_detection(video, threshold)  # detect fish

            # print("Detecting fish direction ...")
            # fish_direction(video)  # detect fish direction

            counting_fish(video)  # count fish

            export_json(video, file_results)  # export results to json file
            # show_video(video,threshold)  # show video

    cv2.destroyAllWindows()
    # Compute counting accuracy
    accuracy = evaluate_frame_count(file_results)

    print("--------------------------------------------------\nEnd  --------------------------------------------------Time :" + str(time.perf_counter() - start))
    return accuracy

if __name__ == '__main__':
    baseline()