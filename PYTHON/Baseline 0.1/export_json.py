"""
   file: export_json.py
   subject : count fish
   Author : AVS7
   Creation : 7/11/2022
   Last Update : 7/11/2022
   Update Note:
        7/11/2022 : Creation


"""
import json
import os

import numpy as np

from video import video as vd


def initialize_json(file_name):
    if os.path.exists(file_name):  # if file exists, delete it
        os.remove(file_name)

    with open(file_name, 'w+') as file:  # create file
        data = "{\"results\": []}"  # initialize file
        file.write(data)  # write data
        file.close()  # close file


def export_json(video, file_name):
    """
    export json
    Args:

        video : video
        file_name: name of the file

    Returns:

    """
    with open(file_name, 'r+') as file:
        file_data = json.load(file)
        if video.enter_frames_numbers.size != 0:
            enter_frames_numbers = video.enter_frames_numbers.tolist()
        else:
            enter_frames_numbers = []
        if video.exit_frames_numbers.size != 0:
            exit_frames_numbers = video.exit_frames_numbers.tolist()
        else:
            exit_frames_numbers = []
        if video.fish_count_frames.size != 0:
            fish_count_frames = video.fish_count_frames.tolist()
        else:
            fish_count_frames = []
        file_data["results"].append({
            "video": video.name,
            "enter_frames": enter_frames_numbers,
            "exit_frames": exit_frames_numbers,
            "fish_count": video.count_fish,
            "sequence": video.sequence[0].tolist()
        })
        file.seek(0)
        json.dump(file_data, file, indent=4)
