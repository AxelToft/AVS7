import cv2 as cv
import numpy as np
from videos import videos


def is_line_activated(line, threshold):
    if np.median(line) > threshold:
        return 1
    if np.median(line) < threshold:
        return 0


def plot_image(image,text):

    cv.putText(image, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv.imshow("img",image)
    cv.waitKey(10)


def Baseline(videos, distance, threshold):
    plot = True
    for video in videos.list_videos:
        vidcap = cv.VideoCapture(video)
        length = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
        ret, next_frame = vidcap.read()
        gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
        sequence = np.zeros(3)
        height = gray.shape[0]
        width = gray.shape[1]
        middle = int(width / 2)
        line1 = gray[:, middle + distance:middle + distance + 1]
        line2 = gray[:, middle - distance:middle - distance + 1]
        while ret:
            line1 = is_line_activated(line1, threshold)
            line2 = is_line_activated(line2, threshold)
            sequence = np.append(sequence, np.array([line1, line2]))
            if np.array_equal(sequence[len(sequence) - 3], [1, 0]) and np.array_equal(sequence[len(sequence) - 2] == [0, 1]) and np.array_equal(sequence[
                len(sequence) - 1] == [0, 0]):
                videos.count_fish += 1
            if plot:
                plot_image(gray,"Count"+str(videos.count_fish))

            ret, next_frame = vidcap.read()
            gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
            line1 = gray[:, middle + distance:middle + distance + 1]
            line2 = gray[:, middle - distance:middle - distance + 1]
            if not ret:
                break
        print('count' + videos.count_fish)


if __name__ == '__main__':
    videos = videos("../Datas/Baseline_videos_mp4/Training_data/*.mp4")

    distance = 400
    threshold = 20
    Baseline(videos, distance, threshold)
