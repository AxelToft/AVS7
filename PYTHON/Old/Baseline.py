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
        1/11/2022 : Dosctrings, option for background substraction and plots
'''


def is_line_activated(line, threshold, i, video) -> int:
    """
    Check if line is activated
    Args:
        line: line of pixels to analyse
        threshold: threshold to activate the line

    Returns:
        1 if line is activated, 0 otherwise
    """
    hist = cv.calcHist([line], [0], None, [256], [0, 256])
    if np.var(hist) < threshold:
        # video.count_fish += 1
        # video.frames_count = np.append(video.frames_count, i)
        return 1
    if np.var(hist) > threshold:
        return 0


def plot_frame(image, text, middle, distance):
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


def check_count(video, sequence, i):
    """
    Check if count is correct
    Args:
        count: count to check
        count_fish: count of fish

    Returns:
        count_fish: count of fish
    """
    print("last sequence : ", str(sequence[-5]), str(sequence[-4]), str(
        sequence[-3]), str(sequence[-2]), str(
        sequence[-1]))
    # if str(sequence[-5]) == str(0.0) and str(sequence[-4])== str(0.1) and str(sequence[-3]== str(1.1)) and str(sequence[-2])==str(1.0) and str(sequence[-1])==str(0.0) :
    if str(sequence[-3] == str(0.1)) and str(
            sequence[-2]) == str(1.1) and str(sequence[-1]) == str(1.0):
        video.count_fish += 1
        video.frames = np.append(video.frames_count, i)
        print("fish compté +1")
    return video.count_fish


def plot_histogram_for_each_frame(line1plot, line2plot, ax, fig, i, var1plot,
                                  var2plot, video, line1, line2, plot_vertical_line1, plot_vertical_line2):
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

    hist1 = cv.calcHist([line1], [0], None, [256], [0, 256])
    hist2 = cv.calcHist([line2], [0], None, [256], [0, 256])
    video.evolution_var1[i] = np.var(hist1)
    video.evolution_var2[i] = np.var(hist2)
    line1plot.set_ydata(hist1)
    line2plot.set_ydata(hist2)
    plot_vertical_line1.set_ydata(line1)
    plot_vertical_line2.set_ydata(line2)
    var1plot.set_ydata(video.evolution_var1)
    var2plot.set_ydata(video.evolution_var2)
    ax[0, 0].set_title('Histogram of line 1, Number of frame :' + str(i))

    ax[0, 1].set_title('Histogram of line 2, Number of frame :' + str(i))

    ax[0, 0].relim()
    ax[0, 1].relim()
    ax[1, 0].relim()
    ax[1, 1].relim()
    ax[2, 0].relim()
    ax[2, 1].relim()
    ax[0, 0].autoscale_view()
    ax[0, 1].autoscale_view()
    ax[1, 1].autoscale_view()
    ax[1, 0].autoscale_view()
    ax[2, 0].autoscale_view()
    ax[1, 0].axhline(y=threshold, color='r', linestyle='-')
    ax[1, 1].axhline(y=threshold, color='r', linestyle='-')

    fig.suptitle('Video number :' + str(video.num_video), fontsize=30)
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()


def Baseline(videos, distance, threshold, background_subtraction, plot_graph=False, plot_image=False):
    """
    Baseline 2 lines study for each video
    Args:
        videos: list of videos
        distance: distance between middle and lines
        threshold: threshold to activate the lines (0-255)
        background_subtraction: True if background subtraction is activated, False otherwise
    Returns:

    """

    for video in videos.list_videos:

        i = 0
        vidcap = video.vidcap

        ret, next_frame = vidcap.read()
        gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

        sequence = np.array([])

        middle = int(video.width / 2)
        threshold = 2500
        if plot_graph:
            plt.ion()
            fig, ax = plt.subplots(3, 2)
            line1plot, = ax[0, 0].plot(
                cv.calcHist([gray[:, middle + distance:middle + distance + 1]], [0], None, [256],
                            [0, 256]))
            line2plot, = ax[0, 1].plot(
                cv.calcHist([gray[:, middle - distance:middle - distance + 1]], [0], None, [256],
                            [0, 256]))
            plot_var1, = ax[1, 0].plot(np.zeros(video.number_frames))
            plot_var2, = ax[1, 1].plot(np.zeros(video.number_frames))
            plot_vertical_line1, = ax[2, 0].plot(np.zeros(video.height))
            plot_vertical_line2, = ax[2, 1].plot(np.zeros(video.height))

        if background_subtraction:
            video.set_background_line(middle, distance)

        while ret:
            if background_subtraction:
                line1 = video.background_subtraction(gray[:, middle + distance:middle + distance + 1], num_line=1)
                line2 = video.background_subtraction(gray[:, middle - distance:middle - distance + 1], num_line=2)
            else:

                line1 = (gray[:, middle + distance:middle + distance + 1])
                line2 = (gray[:, middle - distance:middle - distance + 1])
            if plot_graph:
                plot_histogram_for_each_frame(line1plot, line2plot, ax, fig, i,
                                              plot_var1, plot_var2, video, line1, line2, plot_vertical_line1,
                                              plot_vertical_line2)
            line1 = is_line_activated(line1, threshold, i, video)
            line2 = is_line_activated(line2, threshold, i, video)
            '''if line1 == 1 or line2 == 1:
                threshold = 0'''
            if np.size(sequence) < 5:
                sequence = np.append(sequence, str(line2) + '.' + str(line1))
            elif str(sequence[-1]) != str(line2) + '.' + str(line1):
                sequence = np.append(sequence, str(line2) + '.' + str(line1))
                check_count(video, sequence, i)

            if plot_image:
                plot_frame(next_frame, "Count" + str(videos.count_fish), middle, distance)
            i += 1
            ret, next_frame = vidcap.read()
            if not ret:
                break
            gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
        print('count' + str(video.count_fish))
        # export_json(video)


def plot_histogram(videos, distance, threshold) -> object:
    """
    Plot histogram for each video only
    Args:
        videos: list of videos
        distance: distance between middle and lines
        threshold: threshold to activate the lines (0-255)

    Returns:

    """
    number_video = 0
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
        number_video += 1
        line1plot, = ax[0].plot(
            cv.calcHist([gray[:, middle + distance:middle + distance + 1]], [0], None, [256], [0, 256]))
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
            i += 1

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
    path_video = 'C:/Users/julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/training/*.mp4'
    videos = videos(path_video)

    distance = 100
    threshold = 2500
    Baseline(videos, distance, threshold, background_subtraction=True, plot_image=False, plot_graph=False)
    # plot_histogram(videos, distance, threshold)
