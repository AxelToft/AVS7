import cv2
import os
import json
import numpy as np
import json_exporter as je

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 20,
                       blockSize = 5)


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (100, 100),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class point:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.OldX = 0
        self.OldY = 0
        self.direction = 0
        self.crossedEnter = False
        self.crossedExit = False
        self.points = []
        self.oldPoints = []

    def update(self, meanX, meanY, meanXOld, meanYOld, meanDirection):
        self.x = meanX
        self.y = meanY
        self.OldX = meanXOld
        self.OldY = meanYOld
        self.direction = meanDirection


def nothing(x):
    pass


def trackbar():

    # Create a window
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin','image',0,255,nothing)
    cv2.createTrackbar('VMin','image',0,255,nothing)
    cv2.createTrackbar('HMax','image',0,179,nothing)
    cv2.createTrackbar('SMax','image',0,255,nothing)
    cv2.createTrackbar('VMax','image',0,255,nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    img = cv2.imread('test.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    output = img
    waitTime = 33

    while(1):

        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')

        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(img,img, mask= mask)

        # Print if there is a change in HSV value
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('image',output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break


def load_json():
    with open('C:/Users/Benja/Aalborg Universitet/CE7-AVS 7th Semester - General/Project\Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/Annotations_full.json') as f:
        data = json.load(f)
        return data


def check_annotations(annotations_json, video_name):
    fish_count_json = 0
    fish_count_sequence_c = []
    for annotation in annotations_json:
        if annotation == video_name:
            print("Annotation found")
            fish_count_json = annotations_json[annotation]['fish_count']
            fish_count_sequence_c = annotations_json[annotation]['fish_count_frames']

    return fish_count_json, fish_count_sequence_c


def resize_image(image, scale_percent):
    # Calculate the 50 percent of original dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(image, dsize)

    return output


def background_subtraction(cap, display=True):

    # https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/video/optical_flow/optical_flow.py

    # Create the background subtractor
    fgbg = cv2.createBackgroundSubtractorKNN()
    #fgbg = cv2.createBackgroundSubtractorMOG2()

    # Take first frame and find corners in it
    p0 = None
    prvs = None
    st = None
    fish1 = point(0, 0, 1)
    fish2 = point(0, 0, 2)
    fish3= point(0, 0, 3)
    fishs = [fish1, fish2, fish3]
    fish_count_sequence = []

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    frame_count = 0
    fish_count = 0
    meanX, meanY, meanOldX, meanOldY, meanDirection = 0, 0, 0, 0, 0

    threshold_exit = [(1280, 0), (1280, 1440)]

    print("Counting fish...")

    while True:

        ret, frame = cap.read()

        if ret == True:

            # Get gray image
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply the background subtractor
            fgmask = fgbg.apply(frame)

            # Show the background images
            background_img = fgbg.getBackgroundImage()

            # use morphology to remove noise
            kernel_close = np.ones((10, 10), np.uint8)
            kernel_open = np.ones((20, 20), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)

            # Make the mask binary
            ret, fgmask = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)

            # Gray mask
            gray_mask = cv2.bitwise_and(next, next, mask=fgmask)

            # Remove noise at the top, bottom and right of the image
            gray_mask[:300, :] = 0
            gray_mask[1100:1440, :] = 0
            gray_mask[:, 2000:2560] = 0

            # Find contours in each frame and assign them as a new fish
            contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get the two biggest contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Only keep the biggest 1 contours
            contours = contours[:1]

            # draw the contours
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

            if frame_count > 3:

                for i in range(len(contours)):

                    # If the contour is too small, ignore it
                    if cv2.contourArea(contours[i]) < 5000:
                        continue

                    # Create a new mask for each image
                    gray_mask_copy = np.zeros(gray_mask.shape[:2], dtype=gray_mask.dtype)

                    cv2.drawContours(gray_mask_copy, [contours[i]], 0, 255, -1)
                    result = cv2.bitwise_and(gray_mask, gray_mask, mask=gray_mask_copy)
                    
                    # Write the size of the area of the contour
                    #cv2.putText(frame, str(cv2.contourArea(contours[i])), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, str(frame_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    # If we dont have any good features to track, find them
                    if p0 is None:
                        print("Looking for features...")
                        p0 = cv2.goodFeaturesToTrack(result, mask = None, **feature_params)
                        prvs = result

                    mask = np.zeros_like(result)

                    # If we loose all the features, find new ones
                    if st is None:
                        print("Looking for new features...")
                        p0 = cv2.goodFeaturesToTrack(result, mask = None, **feature_params)
                        prvs = result

                        # calculate optical flow with the new points
                        p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, result, p0, None, **lk_params)
                    else:
                        p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, result, p0, None, **lk_params)

                    # Select good points
                    if p1 is not None:
                        good_new = p1[st==1]
                        good_old = p0[st==1]

                        fishs[i].points = good_new
                        fishs[i].oldPoints = good_old

                    # draw the tracks
                    try:
                        for j, (new, old) in enumerate(zip(fishs[i].points, fishs[i].oldPoints)):

                            # Get the tracked points coordinates
                            a, b = new.ravel()
                            c, d = old.ravel()

                            # Draw the tracked points and their last position
                            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[j].tolist(), 2)
                            mask = cv2.circle(mask, (int(a), int(b)), 5, color[j].tolist(), -1)

                            # Draw the threshold lines
                            mask = cv2.line(mask, threshold_exit[0], threshold_exit[1], color[j].tolist(), 2)

                            # Find mean between each point
                            meanX += a
                            meanY += b

                            meanOldX += c
                            meanOldY += d
                            
                            meanDirection += int(a - c)

                        # Find the mean of all the points
                        if len(fishs[i].points) > 0:
                            meanX = meanX / len(fishs[i].points)
                            meanY = meanY / len(fishs[i].points)
                            meanOldX = meanOldX / len(fishs[i].points)
                            meanOldY = meanOldY / len(fishs[i].points)

                        # Get the mean direction
                        meanDirection = meanDirection / len(fishs[i].points)

                        # Update the fish tracker
                        fishs[i].update(meanX, meanY, meanOldX, meanOldY, meanDirection)

                        # Draw the mean point position
                        mask = cv2.circle(mask, (int(fishs[i].x), int(fishs[i].y)), 5, color[i].tolist(), -1)
                        mask = cv2.circle(mask, (int(fishs[i].OldX), int(fishs[i].OldY)), 5, color[i].tolist(), -1)
                        mask = cv2.line(mask, (int(fishs[i].x), int(fishs[i].y)), (int(fishs[i].OldX), int(fishs[i].OldY)), color[i].tolist(), 2)

                        # Check if the line has crossed the exit threshold and the direction of the line
                        if fishs[i].x < threshold_exit[0][0] and fishs[i].OldX > threshold_exit[0][0] and fishs[i].direction < 0 and fishs[i].crossedExit == False:
                            fishs[i].crossedExit = True
                            fish_count += 1
                            print("Crossed exit threshold with count + 1")
                            fish_count_sequence.append(1)
                        elif fishs[i].x > threshold_exit[0][0] and fishs[i].OldX < threshold_exit[0][0] and fishs[i].direction > 0:
                            fishs[i].crossedExit = False
                            fish_count -= 1
                            print("Crossed exit threshold with count - 1")
                            fish_count_sequence.append(-1)
                        mask2 = cv2.add(result, mask)
                        
                        if display:
                            img_re_mask = resize_image(mask2, 50)
                            cv2.imshow("mask2", img_re_mask)
                    except:
                        pass

                    #cv2.imshow("result2", result)
                    if display:
                        img_re_frame = resize_image(frame, 50)
                        cv2.imshow("frame", img_re_frame)
                        #cv2.imshow("prvs", prvs)
                        if cv2.waitKey(1) & 0xFF == ord('s'):
                            print("Saving images...")
                            cv2.imwrite("backgroundMOG.png", background_img)
                            cv2.imwrite("Frame.png", frame)
                            cv2.imwrite("Mask.png", mask2)

                    # Now update the previous frame and previous points
                    prvs = result.copy()
                    p0 = good_new.reshape(-1, 1, 2)

            # Close the program when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                print("Saving images...")
                cv2.imwrite("backgroundMOG.png", background_img)
                cv2.imwrite("Mask.png", mask2)

            # Update frame count
            frame_count += 1
        elif ret == False:
            break

    print("Done for this video! Counted " + str(fish_count) + " fish.")

    cap.release()
    cv2.destroyAllWindows()
    
    return  fish_count, fish_count_sequence


def evaluation_fish_count(fish_counted, fish_count_json):
    # Calculate the error
    error = abs(fish_counted - fish_count_json)

    # Calculate the percentage error
    percentage_error = (error / fish_count_json) * 100

    print("Actual fish count: " + str(fish_count_json))
    print("Calculated fish count: " + str(fish_counted))
    print("Error: " + str(error))
    print("Percentage error: " + str(percentage_error) + "%")


def main():
    annotations_json = load_json()
    je.initialize_json("Results_test.json")
    fish_count = 0
    fish_count_json = 0
    fish_correct_index = 0
    fish_false_index = 0
    video_dir = "C:/Users/Benja/Aalborg Universitet/CE7-AVS 7th Semester - General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/new_split/test/"
    for entry in os.listdir(video_dir):
        if os.path.isfile(os.path.join(video_dir, entry)):
            print(video_dir + entry)
            cap = cv2.VideoCapture(video_dir + entry)
            fish_count_c, fish_count_sequence = background_subtraction(cap)
            fish_count += fish_count_c
            fish_count_json_c, __ = check_annotations(annotations_json, entry)
            fish_count_json += fish_count_json_c
            je.export_json(entry, fish_count_c, fish_count_sequence, "Results_test.json")  
            
    evaluation_fish_count(fish_count, fish_count_json)
    print("Total fish count: " + str(fish_count))
    print("Total fish index correct: " + str(fish_correct_index))
    print("Total fish index false: " + str(fish_false_index))
    print("Total fish index error: " + str(fish_false_index / (fish_correct_index + fish_false_index)))


if __name__ == "__main__":
    main()