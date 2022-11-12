"""
   file: counting_fish.py
   subject : count fish
   Author : AVS7
   Creation : 7/11/2022
   Last Update : 7/11/2022
   Update Note:
        7/11/2022 : Creation


"""
import glob
import os

import numpy as np

from video import video as vd


def counting_fish(video):
    """
    count fish
    Args:
       list_videos: list of videos

    Returns:

    """
    #TODO : Define entering and exit frames
    counting_fish_v01(video) # baseline 0.1


def counting_fish_v01(video):
    #TODO : study larger sequence
    i = 0
    print(video.sequence[0][0])
    print(len(video.sequence[0]))
    # find the first frame that contains [0,0]
    reached_end = False
    state_1 = None
    state_2 = None
    state = 0
    mid_spawn = False
    while not reached_end:
        if state == 0:
            print("state 0")
            if np.array_equal(video.sequence[0][i], [0,0]) and not mid_spawn: # Look for the first frame that contains [0,0]
                state = 1
            elif np.array_equal(video.sequence[0][i], [1,1]) or mid_spawn: # if we instead find [1,1], then fish spawned in the middle
                mid_spawn = True
                state = 2
                video.enter_frames_numbers = np.append(video.enter_frames_numbers, video.sequence[1][i-1])
                video.fish_count_frames = np.append(video.fish_count_frames, video.sequence[1][i-1])
            i += 1
            if i == len(video.sequence[0]):
                break
        elif state == 1: # Then we wait for the first frame that contains [1,0] or [0,1]
            print("state 1")
            if np.array_equal(video.sequence[0][i], [1,0]):
                state = 2
                state_1 = 1
                video.enter_frames_numbers = np.append(video.enter_frames_numbers, video.sequence[1][i-1])
                video.fish_count_frames = np.append(video.fish_count_frames, video.sequence[1][i-1])
            elif np.array_equal(video.sequence[0][i], [0,1]):
                state = 2
                state_1 = 0
                video.enter_frames_numbers = np.append(video.enter_frames_numbers, video.sequence[1][i-1])
                video.fish_count_frames = np.append(video.fish_count_frames, video.sequence[1][i-1])
            i += 1
            if i == len(video.sequence[0]):
                break
        elif state == 2:
            print("state 2")
            if np.array_equal(video.sequence[0][i], [0,0]):
                state = 3

            i += 1
            if i == len(video.sequence[0]):
                break
        elif state == 3:
            # TODO: after finding a [0,0] we should actually look back in the sequence to find [1,0] or [0,1],
            #  but so far this results in infinite loop and need to be fixed. Possible solutions:
            #  - Only look back a certain number of frames before we stop looking
            #  - Switch state 2 and 3 and instead look forward once for [0,0]. If not found,
            #    then we go back to state 2 and search for new [1,0] or [0,1]
            print("state 3")
            if np.array_equal(video.sequence[0][i], [1,0]):
                if mid_spawn:
                    state_1 = 0
                state = 4
                state_2 = 1
                video.exit_frames_numbers = np.append(video.exit_frames_numbers, video.sequence[1][i-1])
                video.fish_count_frames = np.append(video.fish_count_frames, video.sequence[1][i-1])
            elif np.array_equal(video.sequence[0][i], [0,1]):
                if mid_spawn:
                    state_1 = 1
                state = 4
                state_2 = 0
                video.exit_frames_numbers = np.append(video.exit_frames_numbers, video.sequence[1][i-1])
                video.fish_count_frames = np.append(video.fish_count_frames, video.sequence[1][i-1])
            i += 1
            if i == len(video.sequence[0]) or i == 0:
                break
        elif state == 4:
            #print("state 4")
            # Now we can count the fish
            if state_1 == 1 and state_2 == 0: # [1,0] -> [0,1] = fished entered right and leaved to the left.
                video.count_fish += 1
                video.fish_count_frames = np.append(video.fish_count_frames, video.sequence[1][i-1])
            elif state_1 == 0 and state_2 == 1: # [0,1] -> [1,0] = fished entered left and leaved to the right.
                video.count_fish -= 1
                video.fish_count_frames = np.append(video.fish_count_frames, video.sequence[1][i-1])
            else:
                pass
                #print("Error : state_1 and state_2 are not compatible")
            #print("count_fish = ", video.count_fish)
            state = 0
            state_1 = None
            state_2 = None
            mid_spawn = False




    """for lines_values in video.sequence[0]:
        if i<len(video.sequence[0])-4:
            if (lines_values == [0,1]).all():
                video.enter_frames_numbers = np.append(video.enter_frames_numbers, video.sequence[1][i])
                if (video.sequence[0][i + 1] == [1, 1]).all():
                    if (video.sequence[0][i + 2] == [1, 0]).all():
                        if (video.sequence[0][i + 3] == [0, 0]).all():
                            video.count_fish += 1
                            video.exit_frames_numbers = np.append(video.exit_frames_numbers, video.sequence[1][i])
                            video.fish_count_frames = np.append(video.fish_count_frames,video.sequence[1][i])
            if (lines_values == [0,0]).all():
                if (video.sequence[0][i + 1] == [1, 0]).all():

                    if (video.sequence[0][i + 2] == [1, 1]).all():
                        if (video.sequence[0][i + 3] == [0, 1]).all():
                            if (video.sequence[0][i + 4] == [0, 0]).all():
                                video.fish_count_frames = np.append(video.fish_count_frames,video.sequence[1][i])
                                video.count_fish -= 1

        i += 1
"""

# test function :

'''
path = 'C:/Users/julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/training/*.mp4'

number_video = 0
list_videos = np.array([], dtype=object)

for vid in glob.glob(path):
    if number_video < 2:
        print(vid)
        video = vd(vid, number_video, os.path.basename(vid))
        list_videos = np.append(list_videos,video)
    number_video   += 1
list_videos[0].sequence = np.array([[1, 1], [1, 0], [0, 0]])
list_videos[1].sequence = np.array([[1, 1], [0, 1], [0, 0]])'''
#counting_fish(list_videos)