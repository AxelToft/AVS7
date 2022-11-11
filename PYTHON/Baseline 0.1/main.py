import glob
import os
import time

import numpy as np

from background_subtraction import background_subtraction
from counting_fish import counting_fish
from export_json import export_json
from fish_detection import fish_detection
from fish_direction import fish_direction
from video import video as vd

"""
    file:   main.py
    subject:    call of functions and setting parameters
    Author:  AVS7
    
"""

if __name__ == '__main__':
    # set parameters
    path = 'C:/Users/julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/training/*.mp4'
    file_result = 'results.json'
    if os.path.exists(file_result):
        os.remove(file_result)
    else:
        # create file
        with open(file_result, "w") as f:
            f.close()
    '''path = argparse.ArgumentParser()
    file_result = argparse.ArgumentParser()'''

    list_vid = None
    list_name_videos = np.empty(0)
    start = time.time()

    for vid in glob.glob(path):
        list_vid = np.append(list_vid, vid)
        list_name_videos = np.append(list_name_videos, os.path.basename(vid))
    k = 0
    print("--------------------------------------------------\nStarting --------------------------------------------------")

    for vid in list_vid:
        if vid is not None:
            video = vd(vid, list_name_videos[k], video_number=k)
            k += 1
            print("---------------------------------------------\n Evaluating video number : " + str(
                video.num_video) + "---Video : " + video.name + "---------------------------------------------")

            print("Subtracting background ...")
            before_time = time.time()
            background_subtraction(video)

            print("_________Computing time : " + str(time.time() - before_time) + " seconds_________")

            print("Detecting fish ...")
            before_time = time.time()
            fish_detection(video)
            print("_________Computing time : " + str(time.time() - before_time) + " seconds_________")

            print("Detecting fish direction ...")
            before_time = time.time()
            fish_direction(video)
            print("_________Computing time : " + str(time.time() - before_time) + " seconds_________")
            # count fish and set values for each video -> video.count_fish
            print('Counting fish ...')
            before_time = time.time()
            counting_fish(video)
            print("_________Computing time : " + str(time.time() - before_time) + " seconds_________")
            # export results to json file from list videos results
            print('Exporting results to json file ...')
            before_time = time.time()
            export_json(video, path, file_result)
            print("_________Computing time : " + str(time.time() - before_time) + " seconds_________")
            # release memory
            video.vidcap.release()
            video.background_vidcap.release()
    print("--------------------------------------------------\nEnd  --------------------------------------------------Time :" + str(time.time() - start))
