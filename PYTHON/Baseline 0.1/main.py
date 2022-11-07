import argparse
from export_json import export_json
from background_subtraction import background_subtraction
from counting_fish import counting_fish
from fish_direction import fish_direction
from fish_detection import fish_detection
from get_frames import get_frames
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
    path = argparse.ArgumentParser()
    file_result = argparse.ArgumentParser()
    # set frames for each videos-> video.frames
    list_videos = get_frames(path)

    # subtract background of frames -> video.foreground_frames
    background_subtraction(list_videos)

    # we look for frames where fish are detected -> video.numbers_frames_fish_detected
    fish_detection(list_videos)

    # for these frames, we look for  the direction of the fish -> video.frames_direction
    fish_direction(list_videos)

    # count fish and set values for each video -> video.count_fish
    counting_fish(list_videos)

    # export results to json file from list videos results

    export_json(list_videos, path, file_result)
