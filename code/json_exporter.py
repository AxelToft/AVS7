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

def initialize_json(file_name):
    if os.path.exists(file_name):  # if file exists, delete it
        os.remove(file_name)

    with open(file_name, 'w+') as file:  # create file
        data = "{\"results\": []}"  # initialize file
        file.write(data)  # write data
        file.close()  # close file


def export_json(video_name, fish_count, fish_sequence, file_name):
    """
    export json
    Args:
        video : video
        file_name: name of the file
    Returns:
    """
    with open(file_name, 'r+') as file:
        file_data = json.load(file)
        #if fish_sequence.size != 0:
            #fish_count_frames = fish_sequence.tolist()
        #else:
            #fish_count_frames = []
        file_data["results"].append({
            "video": video_name,
            "fish_count_frames": fish_sequence,
            "enter_frames": None,
            "exit_frames": None,
            "fish_count": fish_count,
            "sequence": []
        })
        file.seek(0)
        json.dump(file_data, file, indent=4)