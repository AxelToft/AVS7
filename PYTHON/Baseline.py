import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from videos import videos
'''
   Baseline 2 lines study
   Autor : Julien Roussel Galle
   Creation : 1/11/2022
   Last Update : 1/11/2022
   Update Note: 
        31/10/2022 : Plot histogram for every videos and every frame in same time as showing the video
        1/11/2022 : Dosctrings
'''


def is_line_activated(line, threshold) -> int:
    """
    Check if line is activated
    Args:
        line: line of pixels to analyse
        threshold: threshold to activate the line

    Returns:
        1 if line is activated, 0 otherwise
    """
    if np.median(line) > threshold:
        return 1
    if np.median(line) < threshold:
        return 0


def plot_image(image, text, middle, distance):
    """
    Plot image with 2 lines
    Args:
        image: image to plot
        text: text to display
        middle: middle of the image
        distance: distance between the 2 lines

    Returns:

    """
    cv.putText(image, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv.line(image, (middle + distance, 0), (middle + distance, image.shape[0]), (255, 255, 255), 2)
    cv.line(image, (middle - distance, 0), (middle - distance, image.shape[0]), (255, 255, 255), 2)
    cv.imshow("img", image)
    cv.waitKey(10)


def Baseline(videos, distance, threshold):
    """
    Baseline 2 lines study for each video
    Args:
        videos: list of videos
        distance: distance between middle and lines
        threshold: threshold to activate the lines (0-255)

    Returns:

    """
    plot = True
    plot_hist = True
    for video in videos.list_videos:
        vidcap = cv.VideoCapture(video)
        length = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
        ret, next_frame = vidcap.read()
        gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

        sequence = np.zeros(3)
        height = next_frame.shape[0]
        width = next_frame.shape[1]
        middle = int(width / 2)
        bg1 = gray[:, middle + distance:middle + distance + 1]
        bg2 = gray[:, middle - distance:middle - distance + 1]
        i = 0
        if plot_hist:
            plt.ion()
            fig, ax = plt.subplots(2)
            line1plot, = ax[0].plot(
                cv.calcHist([np.abs(gray[:, middle + distance:middle + distance + 1]-bg1)], [0], None, [256], [0, 256]))
            line2plot, = ax[1].plot(
                cv.calcHist([np.abs(gray[:, middle - distance:middle - distance + 1]-bg2)], [0], None, [256], [0, 256]))

        while ret:

            gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
            if plot_hist:

                plot_histogram_for_each_frame(gray, line1plot, line2plot, ax, fig, i, middle, distance,bg1,bg2)

            line1 = gray[:, middle + distance:middle + distance + 1]
            line2 = gray[:, middle - distance:middle - distance + 1]
            line1 = is_line_activated(line1, threshold)
            line2 = is_line_activated(line2, threshold)
            sequence = np.append(sequence, np.array([line1, line2]))
            if np.array_equal(sequence[len(sequence) - 3], [1, 0]) and np.array_equal(
                    sequence[len(sequence) - 2] == [0, 1]) and np.array_equal(sequence[
                                                                                  len(sequence) - 1] == [0, 0]):
                videos.count_fish += 1
            if plot:
                plot_image(gray, "Count" + str(videos.count_fish), middle, distance)

            ret, next_frame = vidcap.read()

            if not ret:
                break
        print('count' + str(videos.count_fish))


def plot_histogram_for_each_frame(gray,line1plot,line2plot,ax,fig,i,middle,distance,bg1,bg2):
    """
    Plot histogram for each frame in same time as showing the video
    Args:
        gray: Gray image
        line1plot: line 1 plot
        line2plot: line 2 plot
        ax: axis
        fig: figure
        i: Number of frames
        middle: middle of the image
        distance: distance between the middle and the lines
        bg1: background of line 1
        bg2: background of line 2

    Returns:

    """
    line1 = np.abs(gray[:, middle + distance:middle + distance + 1]-bg1)
    line2 = np.abs(gray[:, middle - distance:middle - distance + 1]-bg2)
    line1plot.set_ydata(cv.calcHist([line1], [0], None, [256], [0, 256]))
    ax[0].set_title('Histogram of line 1, Number of frame :' + str(i))
    line2plot.set_ydata(cv.calcHist([line2], [0], None, [256], [0, 256]))
    ax[1].set_title('Histogram of line 2, Number of frame :' + str(i))
    i += 1
    ax[0].autoscale()
    ax[1].autoscale()
    ax[0].axhline(y=threshold, color='r', linestyle='-')
    ax[1].axhline(y=threshold, color='r', linestyle='-')

    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_histogram(videos, distance, threshold) -> object:
    """
    Plot histogram for each video only
    Args:
        videos: list of videos
        distance: distance between middle and lines
        threshold: threshold to activate the lines (0-255)

    Returns:

    """
    number_video =0
    for video in videos.list_videos:
        vidcap = cv.VideoCapture(video)
        length = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))

        ret, next_frame = vidcap.read()
        gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

        width = next_frame.shape[1]
        middle = int(width / 2)

        i = 0
        plt.clf()
        plt.ion()
        fig, ax = plt.subplots(2)
        number_video +=1
        line1plot, = ax[0].plot(cv.calcHist([gray[:, middle + distance:middle + distance + 1]], [0], None, [256], [0, 256]))
        line2plot, = ax[1].plot(
            cv.calcHist([gray[:, middle - distance:middle - distance + 1]], [0], None, [256], [0, 256]))

        while ret:
            gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
            line1 = gray[:, middle + distance:middle + distance + 1]
            line2 = gray[:, middle - distance:middle - distance + 1]
            line1plot.set_ydata(cv.calcHist([line1], [0], None, [256], [0, 256]))
            ax[0].set_title('Histogram of line 1, Number of frame :' + str(i))
            line2plot.set_ydata(cv.calcHist([line2], [0], None, [256], [0, 256]))
            ax[1].set_title('Histogram of line 2, Number of frame :' + str(i))
            i+=1


            ax[0].axhline(y=threshold, color='r', linestyle='-')
            ax[1].axhline(y=threshold, color='r', linestyle='-')
            fig.canvas.draw()
            fig.canvas.flush_events()
            ret, next_frame = vidcap.read()
            if not ret:
                break

if __name__ == '__main__':
    """
    Main function
    """
    videos = videos("../Datas/Baseline_videos_mp4/Training_data/*.mp4")

    distance = 400
    threshold = 20
    Baseline(videos, distance, threshold)
    #plot_histogram(videos, distance, threshold)
