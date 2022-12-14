import glob
import os
import time

import numpy as np
import cv2 as cv
# import video.py from another folder

from video import video as vd
from background_subtraction import *


def get_videos(path=None):
    if path is None:
        path = 'C:/Users/\julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents (1)/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/new_split/train/*.mp4'

    list_vid = np.array([])  # list of videos
    for vid in glob.glob(path):  # for each video in the path
        list_vid = np.append(list_vid, vid)  # add video to list
    return list_vid


def plot_images(images_toPlot, k):
    cv.namedWindow('image' + k, cv.WINDOW_NORMAL)
    cv.resizeWindow('image' + k, 1200, 800)
    for image in images_toPlot:
        cv.imshow('image' + k, image)
        cv.waitKey(0)


def save_images(images_toPlot,k,i):
    cv.imwrite('../Image processing/figures/frame.png', images_toPlot[0])
    cv.imwrite('../Image processing/figures/background.png', images_toPlot[1])
    cv.imwrite(f'../Image processing/figures/foreground_video{k}frame{i}.png', images_toPlot[2])
    cv.imwrite('../Image processing/figures/binary.png', images_toPlot[3])
    cv.imwrite(''
               '/figures/binary_morph_open.png', images_toPlot[4])
    cv.imwrite(''
               '/figures/binary_morph_close.png', images_toPlot[5])
    cv.imwrite(''
               '../Image processing/figures/rgb_frame.png', images_toPlot[6])
    # cv.imwrite('../Image processing/figures/edges.png', images_toPlot[6])


def select_ROI(video, frame):
    # Select ROI
    y1, y2, x1, x2 = video.origin[0], video.origin[0] + video.roi_height, video.origin[1], video.origin[1] + video.roi_width

    frame[:y1, :] = frame[y2:, :] = frame[:, :x1] = frame[:, x2:] = 255
    return frame


def image_processing_video(video):
    background = mean_background_subtraction(video, entire_frame=True)  # get the background
    #background = cv.imread("../Image processing/backgrounds/mean_background.png", cv.IMREAD_GRAYSCALE)
    roi_background = select_ROI(video, background)
    fgbg = cv.createBackgroundSubtractorKNN()
    for i, frame in enumerate(video.gray_frames):
        roi = frame
        # roi = select_ROI(video, frame)
        foreground = fgbg.apply(roi)
        #foreground = np.where((roi - roi_background) < 0, 0, roi - roi_background).astype(np.uint8)
        #foreground = cv.subtract(roi, roi_background)

        video.frames_subtracted[i] = foreground
        #binary = cv.threshold(foreground, 25, 255, cv.THRESH_BINARY)[1]

        #binary = cv.adaptiveThreshold(foreground, 255, cv.ADAPTIVE_THRESH_GAU SSIAN_C, cv.THRESH_BINARY_INV, 11, 2)


        binary_morph_close = cv.morphologyEx(foreground, cv.MORPH_CLOSE, np.ones((10, 10), np.uint8))
        binary_morph_open = cv.morphologyEx(binary_morph_close, cv.MORPH_OPEN, np.ones((20, 20), np.uint8))
        ret, binary = cv.threshold(binary_morph_open, 100, 255, cv.THRESH_BINARY)
        gray_mask = cv.bitwise_and(roi, roi, mask=binary)


        video.line1[i] = gray_mask [video.origin[0]:video.origin[0]+video.roi_height, video.width // 2 + video.DISTANCE_FROM_MIDDLE:video.width // 2 + video.DISTANCE_FROM_MIDDLE + 1]
        video.line2[i] = gray_mask [video.origin[0]:video.origin[0]+video.roi_height, video.width // 2 - video.DISTANCE_FROM_MIDDLE:video.width // 2 - video.DISTANCE_FROM_MIDDLE + 1]
        video.frames_after_processing[i] = gray_mask
        images_toPlot = []
        '''if i == 10:
            images_toSave = [frame, background, foreground, binary, binary_morph_open, binary_morph_close]
            save_images(images_toSave)'''
            # plot_images(images_toPlot, str(k))



def image_processing():
    # initiate video object
    list_videos = get_videos()
    # get image
    for k, vid in enumerate(list_videos):  # for each video in the list
        if vid is not None and k ==1:  # if video is not empty
            video = vd(vid, k)  # create video object
            start = time.perf_counter()  # start time
            background = mean_background_subtraction(video, entire_frame=True)
            #background = cv.imread("../Image processing/backgrounds/1framebg.png", cv.IMREAD_GRAYSCALE)
            print(f"temps de background calculé : {time.perf_counter() - start}")
            for i, frame in enumerate(video.gray_frames):
                rgb_frame = video.frames[i]
                foreground = cv.absdiff(frame, background)

                binary = cv.threshold(foreground, 40, 255, cv.THRESH_BINARY)[1]
                binary_morph_open = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones((10, 10), np.uint8))
                binary_morph_close = cv.morphologyEx(binary_morph_open, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

                edges = cv.Canny(binary_morph_close, 50, 255)
                images_toPlot = [binary_morph_close]
                images_toSave = [frame, background, foreground, binary, binary_morph_open, binary_morph_close,rgb_frame]
                if i == 74 :
                    save_images(images_toSave, str(k), str(i))
                #plot_images(images_toPlot, str(k))
            cv.destroyAllWindows()


def get_mean_allbackgrounds():
    list_videos = get_videos()
    # get image
    mean_background = None
    for k, vid in enumerate(list_videos):  # for each video in the list
        if vid is not None:  # if video is not empty
            video = vd(vid, k)  # create video object

            background = median_background_subtraction(video, entire_frame=True)
            if k == 0:
                mean_background = background
            else:

                mean_background = np.mean(np.array([background, mean_background]), axis=0).astype(np.uint8)
            cv.imwrite('../Image processing/backgrounds/mean_of_median_background.png', mean_background)
            if k == 2:
                cv.imwrite('../Image processing/backgrounds/1framebg.png', video.gray_frames[0])
            '''for i,frame in enumerate(video.gray_frames):
                if not os.path.exists(f"frames/video{k}"):
                    os.makedirs(f"frames/video{k}")

                cv.imwrite(f"frames/video{k}/frames{i}.png", frame)
                print(f"frames/video{k}/frames{i}.png")'''


if __name__ == '__main__':
    image_processing()
    # get_mean_allbackgrounds()
