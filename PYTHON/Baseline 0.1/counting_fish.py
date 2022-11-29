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
    """
    Uses the [0,0] [0,1], and [0,1] logic for counting fish in a statemachine apporach.
    Stage0: Look for first appearance of [0,0] -> (go to stage 1) or [1,1] -> (go to stage 2). End by increment i.
    Stage1: Look for first appearance of [0,1] or [1,0], save line_state_1 and go to stage 2. End by increment i.
    Stage2: Look for first appearance of [0,0] and go to stage 3. End by decrement i.
    Stage3: go back in sequence until [0,1] or [1,0] is found. Save line_state_2
            (if we skipped stage 1, then set line_state_1 to the opposite of line_state_2) and go to stage 4
    Stage4: Depending on line_state_1 and line_state_2, increment or decrement the fish count.

    :param video:
    :return:
    """
    #TODO : study larger sequence
    i = 0
    reached_end = False
    line_state_1 = None
    line_state_2 = None
    stage = 0
    mid_spawn = False
    i_at_enterframe = None
    i_at_exitframe = None
    # We use a stage machine approach to solve the logic problem
    while not reached_end:
        if stage == 0:
            print("stage 0")
            if np.array_equal(video.sequence[0][i], [0,0]) and not mid_spawn: # Look for the first frame that contains [0,0]
                stage = 1
            elif np.array_equal(video.sequence[0][i], [1,1]) or mid_spawn: # if we instead find [1,1], then fish spawned in the middle
                mid_spawn = True
                stage = 2 # We skip stage 1 and go directly to stage 2
                # We index by -1 because the first index in video.sequence[0] contains [None, None], which does not contian a frame entry
                video.enter_frames_numbers = np.append(video.enter_frames_numbers, video.sequence[1][i - 1])
            i += 1
            if i >= len(video.sequence[0]): # Stop if we reach the end of the sequence
                break
        elif stage == 1: # Then we wait for the first frame that contains [1,0] or [0,1]
            print("stage 1")
            if np.array_equal(video.sequence[0][i], [1,0]): # Save the sequence and go to next stage
                stage = 2
                line_state_1 = [1, 0]
                video.enter_frames_numbers = np.append(video.enter_frames_numbers, video.sequence[1][i - 1])
            elif np.array_equal(video.sequence[0][i], [0, 1]):
                stage = 2
                line_state_1 = [0, 1]
                video.enter_frames_numbers = np.append(video.enter_frames_numbers, video.sequence[1][i - 1])
            i_at_enterframe = i
            i += 1
            if i >= len(video.sequence[0]):
                break
        elif stage == 2: # Now we wait for the fish to exit again
            print("stage 2")
            if np.array_equal(video.sequence[0][i], [0,0]):
                stage = 3
                i -= 1 # When we find [0,0], we go back in sequence to find the first [0,1] or [1,0]
            else:
                i += 1
            if i >= len(video.sequence[0]):
                break
        elif stage == 3:
            print("stage 3") # Now we go back in sequence to find the first [0,1] or [1,0]
            if i_at_exitframe is None:
                i_at_exitframe = i+2 # Save the index of the frame after where the fish exited
            if i_at_enterframe != i: # We continue to go back until we reach the frame where the fish entered or find [0,1] or [1,0]
                if np.array_equal(video.sequence[0][i], [1,0]):
                    if mid_spawn:
                        line_state_1 = [0, 1]
                    stage = 4
                    line_state_2 = [1, 0]
                    video.exit_frames_numbers = np.append(video.exit_frames_numbers, video.sequence[1][i - 1])
                    i = i_at_exitframe  # Return to saved i-th frame
                elif np.array_equal(video.sequence[0][i], [0, 1]):
                    if mid_spawn:
                        line_state_1 = [1, 0]
                    stage = 4
                    line_state_2 = [0, 1]
                    video.exit_frames_numbers = np.append(video.exit_frames_numbers, video.sequence[1][i - 1])
                    i = i_at_exitframe
                i -= 1
            else: # If we reach the frame where the fish entered, then we go back to stage 2 and look for the next exit
                stage = 2
                i = i_at_exitframe
            if i >= len(video.sequence[0]):
                break
        elif stage == 4:
            print("stage 4")
            # Now we can count the fish
            if line_state_1 == [0, 1] and line_state_2 == [1, 0]: # [0,1] -> [1,0] = fished entered right and leaved to the left.
                video.count_fish += 1
                video.fish_count_frames = np.append(video.fish_count_frames, 1)
            elif line_state_1 == [1, 0] and line_state_2 == [0, 1]:  # [1,0] -> [0,1] = fished entered left and leaved to the right.
                video.count_fish -= 1
                video.fish_count_frames = np.append(video.fish_count_frames, -1)
            elif line_state_1 == [0, 1] and line_state_2 == [0, 1] or line_state_1 == [1, 0] and line_state_2 == [1, 0]:
                video.fish_count_frames = np.append(video.fish_count_frames, 0)
            stage = 0
            line_state_1 = None
            line_state_2 = None
            i_at_exitframe = None
            i_at_enterframe = None
            mid_spawn = False


