import cv2 as cv
import numpy as np
from matplotlib.pyplot import subplots, xlim, ylim, tight_layout, axhline
from moviepy.video.io.bindings import mplfig_to_npimage
from numpy import argmax


def show_video(video,threshold):
    """
    Show video
    Args:
        video: video

    Returns:

    """

    # prepare a small figure to embed into frame
    fig, ax = subplots(figsize=(5, 4), facecolor='w')
    line, = ax.plot(video.evolution_var1)
    line2, = ax.plot(video.evolution_var2)
    xlim([0, video.number_frames])
    #ylim([0, np.max(video.evolution_var1)])  # setup wide enough range here

    ylim([0, 7000])  # setup wide enough range here
    axhline(y=threshold, color='r', linestyle='-')
    #('off')
    tight_layout()
    graphRGB = mplfig_to_npimage(fig)
    gh, gw, _ = graphRGB.shape
    text = ''
    for k,frame in enumerate(video.frames_subtract):
        #drw videos vertical line
        cv.line(frame, (video.width // 2-video.DISTANCE_FROM_MIDDLE, 250), (video.width // 2-video.DISTANCE_FROM_MIDDLE, 1440), (0, 0, 255), 2)
        cv.line(frame, (video.width // 2+video.DISTANCE_FROM_MIDDLE, 250), (video.width // 2+video.DISTANCE_FROM_MIDDLE, 1440), (0, 0, 255), 2)
        # resize windows to fit the screen
        cv.namedWindow('video', cv.WINDOW_NORMAL)
        cv.resizeWindow('video', 1280, 800)
        line.set_ydata(video.evolution_var1)
        line2.set_ydata(video.evolution_var2)
        frame[:gh+0, video.width - gw:] = cv.cvtColor(mplfig_to_npimage(fig), cv.COLOR_BGR2GRAY) # convert to grayscale
        text = f'Video :{video.name}frame :{k}'
        if k in video.exit_frames_numbers:
            text = 'numbers fish :' + str(video.count_fish)
        cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('video', frame)
        key = cv.waitKey(1)

def show_line(video):

    fig, ax = subplots(figsize=(5, 4), facecolor='w')
    line1plt, = ax.plot(video.line1[0])
    line2plt, = ax.plot(video.line2[0])

    xlim([0,1440])

    ylim([0, 256])  # setup wide enough range here
    #('off')
    tight_layout()
    graphRGB = mplfig_to_npimage(fig)
    gh, gw, _ = graphRGB.shape
    text = 'numbers fish :'
    for k, (line1, line2,frame) in enumerate(zip(video.line1, video.line2,video.frames)):
        #drw videos vertical line
        cv.line(frame, (video.width // 2-video.DISTANCE_FROM_MIDDLE, 0), (video.width // 2-video.DISTANCE_FROM_MIDDLE, video.height), (0, 0, 255), 2)
        cv.line(frame, (video.width // 2+video.DISTANCE_FROM_MIDDLE, 0), (video.width // 2+video.DISTANCE_FROM_MIDDLE, video.height), (0, 0, 255), 2)
        # resize windows to fit the screen
        cv.namedWindow('video', cv.WINDOW_NORMAL)
        cv.resizeWindow('video', 1280, 800)
        line1plt.set_ydata(line1)
        line2plt.set_ydata(line2)
        frame[:gh+0, video.width - gw:, :] = mplfig_to_npimage(fig)
        text = f'frame :{k} numbers fish :' + str(video.count_fish)
        if k in video.exit_frames_numbers:
            text = 'numbers fish :' + str(video.count_fish)
        cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('video', frame)
        key = cv.waitKey(1)
