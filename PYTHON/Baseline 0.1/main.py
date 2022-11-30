import glob
import time
import cv2
import numpy as np

from background_subtraction import *
from counting_fish import counting_fish
from evaluation import evaluate_frame_count
from export_json import export_json, initialize_json
from fish_detection import *
from video import video as vd
from show_video import show_video, show_line

"""
    file:   main.py
    subject:    call of functions and setting parameters
    Author:  AVS7
    
"""


def baseline(distance=200, threshold=900, path=None, file_results='results.json'):
    # Path to define
    if path is None:
        path = 'C:/Users/\julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents (1)/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/new_split/test/*.mp4'
    list_vid = []  # list of videos
    start = time.perf_counter()  # start time
    for vid in glob.glob(path):  # for each video in the path
        list_vid = np.append(list_vid, vid)  # add video to list
    print(f"--------------------------------------------------\nStarting list of videos -------parameters of computing : Distance {distance}   Threshold: {threshold}")

    initialize_json(file_results)  # initialize json file

    for k, vid in enumerate(list_vid):  # for each video in the list
        if vid is not None and vid:  # if video is not empty
            video = vd(vid, k)  # create video object
            video.DISTANCE_FROM_MIDDLE = distance  # set distance
            #print("\n Evaluating video number : " + str(video.num_video) + "---Video : " + video.name)
            # TODO save subtract to other files in order to not compute every time

            background_subtraction(video, method='mean', entire_frame=False, save=True)  # median background subtraction
            fish_detection(video, threshold)  # detect fish
            counting_fish(video)  # count fish
            export_json(video, file_results)  # export results to json file
            #show_video(video, threshold)  # show video
            # show_line(video, threshold) # show line
    cv2.destroyAllWindows()
    # Compute counting accuracy
    accuracy, false_videos = evaluate_frame_count(file_results)

    print("--------------------------------------------------\nEnd  --------------------------------------------------Time :" + str(time.perf_counter() - start))
    return accuracy, false_videos


if __name__ == '__main__':
    baseline()
