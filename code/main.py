import cv2
import os
import json
import numpy as np
import json_exporter as je


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

    def update_new(self, meanX, meanY):
        self.x = meanX
        self.y = meanY
        
    def update_old(self):
        self.OldX = self.x
        self.OldY = self.y
        
    def update_direction(self, direction):
        self.direction = direction
        
    


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
    with open('Annotations_full.json') as f:
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


def templete_matching_2(cap):
    # Load the template image
    template = cv2.imread("Images/template.png", 0)

    # Set the threshold for the template matching
    threshold = 0.7

    # Set the minimum and maximum scales for the template
    min_scale = 0.5
    max_scale = 2.0

    # Set the scale step for the template
    scale_step = 0.1

    # Set the optical flow parameters
    lk_params = dict(winSize = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create the previous frame and previous points arrays
    prev_frame = None
    prev_points = []

    while True:
        # Read the current frame
        ret, frame = cap.read()
        frame = resize_image(frame, 50)

        # Check if we have reached the end of the video
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use optical flow to find the points that have moved between the previous and current frames
        if prev_frame is not None:
            try:
                curr_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_points, None, **lk_params)
                prev_frame = gray.copy()
                prev_points = curr_points
            except:
                prev_frame = gray.copy()
                prev_points = []
        else:
            prev_frame = gray.copy()

        # Use template matching to find the location of the template in the current frame
        for scale in np.arange(min_scale, max_scale, scale_step):
            resized_template = cv2.resize(template, None, fx=scale, fy=scale)
            res = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            # Loop over all the detected locations
            for pt in zip(*loc[::-1]):
                # Draw a rectangle around the detected object
                cv2.rectangle(frame, pt, (pt[0] + resized_template.shape[1], pt[1] + resized_template.shape[0]), (0, 255, 0), 1)

        # Display the frame
        cv2.imshow("Frame", frame)

        # Check if the user pressed the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def templete_matching(cap):
    # Load the template image
    template = cv2.imread("code/template.png", 0)

    # Set the threshold for the template matching
    threshold = 0.5

    # Create the background subtractor
    fgbg = cv2.createBackgroundSubtractorKNN()

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply the background subtractor
        fgmask = fgbg.apply(frame)

        # use morphology to remove noise
        kernel_close = np.ones((10, 10), np.uint8)
        kernel_open = np.ones((20, 20), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)

        # Make the mask binary
        ret, fgmask = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)

        # Gray mask
        gray_mask = cv2.bitwise_and(gray, gray, mask=fgmask)

        # Check if we have reached the end of the video
        if not ret:
            break

        # Use template matching to find the location of the template in the frame
        res = cv2.matchTemplate(gray_mask, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        # Loop over all the detected locations
        for pt in zip(*loc[::-1]):
            # Draw a rectangle around the detected object
            cv2.rectangle(gray_mask, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 1)

        # Display the frame
        display_img = resize_image(gray_mask, 50)
        cv2.imshow("Frame", display_img)

        # Check if the user pressed the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def track_fish(cap, display=True):

    # Create the background subtractor
    fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True, dist2Threshold=400) # Defualt distThreshold = 400
    print(fgbg.getDist2Threshold())
    #fgbg = cv2.createBackgroundSubtractorMOG2()

    # Take first frame and find corners in it
    fish1 = point(0, 0, 1)
    fish2 = point(0, 0, 2)
    fishs = [fish1, fish2]
    fish_count_sequence = []

    frame_count = 0
    fish_count = 0

    threshold_exit = [(1280, 0), (1280, 1440)]

    print("Counting fish...")

    while True:

        ret, frame = cap.read()

        if ret == True:

            # Get gray image
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply the background subtractor
            fgmask = fgbg.apply(frame)

            # Gray mask for the copy image
            gray_mask_copy_n = cv2.bitwise_and(frame, frame, mask=fgmask)
            frame_copy = frame.copy()

            # Show the background images
            background_img = fgbg.getBackgroundImage()

            # use morphology to remove noise
            kernel_close = np.ones((10, 10), np.uint8)
            kernel_open = np.ones((20, 20), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)

            # Make the mask binary
            ret, fgmask = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)
            
            fgmask_copy = fgmask.copy()

            # Gray mask
            gray_mask = cv2.bitwise_and(next, next, mask=fgmask)

            # Gray mask copy
            gray_mask_copy_morph = cv2.bitwise_and(next, next, mask=fgmask)

            # Remove noise at the top, bottom and right of the image
            gray_mask[:300, :] = 0
            gray_mask[1100:1440, :] = 0
            gray_mask[:, 2000:2560] = 0
            
            #next[:300, :] = 0
            #next[1100:1440, :] = 0
            #next[:, 2000:2560] = 0
            
            # Count all non-zero pixels
            #print("Non-zero pixels: " + str(cv2.countNonZero(next)))
            
            cv2.imshow("frame_copy", next)

            # Find contours in each frame and assign them as a new fish
            contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get the two biggest contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Only keep the biggest 2 contours
            contours = contours[:2]

            if frame_count > 3:

                for i in range(len(contours)):
                    
                    #print("Contour area: " + str(cv2.contourArea(contours[i])))
                    
                    # If the contour is too small, ignore it
                    if cv2.contourArea(contours[i]) < 80000:
                        print("Skipping contour...")
                        continue
                    
                    # draw the contours
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

                    # Create a new mask for each image
                    gray_mask_copy = np.zeros(gray_mask.shape[:2], dtype=gray_mask.dtype)

                    cv2.drawContours(gray_mask_copy, [contours[i]], 0, 255, -1)
                    result = cv2.bitwise_and(gray_mask, gray_mask, mask=gray_mask_copy)
                    
                    # Write the frame number
                    cv2.putText(frame, str(frame_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    mask = np.zeros_like(result)
                    
                    # Get the center point of each contour of the fish
                    M = cv2.moments(contours[i])
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    tracked_point = (cx, cy)
                    fishs[i].update_new(tracked_point[0], tracked_point[1])
                    
                    # Make sure the fish first position is not 0, 0
                    if fishs[i].OldX == 0 and fishs[i].OldY == 0:
                        fishs[i].OldX = fishs[i].x
                        fishs[i].OldY = fishs[i].y

                    # Get the direction of the fish
                    direction_fish = (fishs[i].x - fishs[i].OldX) + (fishs[i].y - fishs[i].OldY)
                    fishs[i].direction = direction_fish

                    # Draw the mean point position
                    cv2.circle(frame, (int(fishs[i].x), int(fishs[i].y)), 5, (0, 0,255), -1)
                    cv2.circle(frame, (int(fishs[i].OldX), int(fishs[i].OldY)), 5, (255, 0, 0), -1)
                    cv2.line(frame, (int(fishs[i].x), int(fishs[i].y)), (int(fishs[i].OldX), int(fishs[i].OldY)), (0, 255, 0), 2)
                    
                    # Draw the threshold lines
                    cv2.line(frame, threshold_exit[0], threshold_exit[1], (255,255,0), 2)

                    # Check if the line has crossed the exit threshold and the direction of the line
                    if fishs[i].x < threshold_exit[0][0] and fishs[i].OldX > threshold_exit[0][0] and fishs[i].direction < 0:
                        fish_count += 1
                        print("Crossed exit threshold with count + 1")
                        fish_count_sequence.append(1)
                    elif fishs[i].x > threshold_exit[0][0] and fishs[i].OldX < threshold_exit[0][0] and fishs[i].direction > 0:
                        fish_count -= 1
                        print("Crossed exit threshold with count - 1")
                        fish_count_sequence.append(-1)
                        
                    # Update the fish tracker old points
                    fishs[i].update_old()

                    #cv2.imshow("result2", result)
                    if display:
                        img_re_frame = resize_image(frame, 50)
                        cv2.imshow("frame", img_re_frame)
                        #cv2.imshow("prvs", prvs)
                        if cv2.waitKey(1) & 0xFF == ord('s'):
                            print("Saving images...")
                            # cv2.imwrite("backgroundKNN.png", background_img)
                            cv2.imwrite("Frame.png", frame_copy)
                            cv2.imwrite("No_morph.png", gray_mask_copy_n)
                            cv2.imwrite("with_morph.png", gray_mask_copy_morph)
                            cv2.imwrite("result.png", frame)
                            cv2.imwrite("contours.png", fgmask_copy)

            # Close the program when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                print("Saving images...")
                cv2.imwrite("backgroundMOG.png", background_img)

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
    je.initialize_json("Results_val_centroid.json")
    fish_count = 0
    fish_count_json = 0
    fish_correct_index = 0
    fish_false_index = 0
    video_dir = "val/"
    for entry in os.listdir(video_dir):
        if os.path.isfile(os.path.join(video_dir, entry)):
            print(video_dir + entry)
            cap = cv2.VideoCapture(video_dir + entry)
            fish_count_c, fish_count_sequence = track_fish(cap)
            fish_count += fish_count_c
            fish_count_json_c, __ = check_annotations(annotations_json, entry)
            fish_count_json += fish_count_json_c
            je.export_json(entry, fish_count_c, fish_count_sequence, "Results_val_centroid.json")  
            
    evaluation_fish_count(fish_count, fish_count_json)
    print("Total fish count: " + str(fish_count))
    print("Total fish index correct: " + str(fish_correct_index))
    print("Total fish index false: " + str(fish_false_index))

    if fish_correct_index + fish_false_index > 0:
        print("Total fish index error: " + str(fish_false_index / (fish_correct_index + fish_false_index)))
    else:
        print("Total fish index error: 0")


if __name__ == "__main__":
    main()