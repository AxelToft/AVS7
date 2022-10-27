#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>


using namespace cv;
using namespace std;



Mat FishBox() {
	Mat	FishBox = imread("C:\\Users\\axelt\\Aalborg Universitet\\CE7-AVS 7th Semester - General\\Project\\Baselines\\Ultraline\\fishbox.jpg");
	//imshow("FishBox", FishBox);
	cvtColor(FishBox, FishBox, COLOR_BGR2HSV);
	return FishBox;
}


// 18 = going backwards

Mat FirstFrame() {
	Mat FirstFrame;
	VideoCapture capture("C:\\Users\\axelt\\Aalborg Universitet\\CE7-AVS 7th Semester - General\\Project\\Vattenfall-fish-open-data\\fishai_training_datasets_v4\\video\\Baseline_videos_avi\\Training_data\\fish_video32.avi");
	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to open file!" << endl;
	}
	capture >> FirstFrame;
	cvtColor(FirstFrame, FirstFrame, COLOR_BGR2HSV);
	//namedWindow("FirstFrame", WINDOW_NORMAL);
	//imshow("FirstFrame", FirstFrame);

	return FirstFrame;
}

tuple<Mat, Point> crop() {
	Mat ImgCrop, mask, roi_hist, templatematch;
	Mat frame = FirstFrame();
	Mat roi = FishBox();
	Point minLoc(0, 0), maxLoc(0, 0);
	double maxVal;

	matchTemplate(frame, roi, templatematch, TM_CCOEFF_NORMED);
	minMaxLoc(templatematch, NULL, &maxVal, NULL, &maxLoc);

	Rect myROI(maxLoc.x, maxLoc.y, roi.cols, roi.rows);
	ImgCrop = frame(myROI);

	return { ImgCrop, maxLoc };
}



int main(int argc, char** argv) {

	Mat Ultraline, roi1, roi2, frame, hist1, hist2;
	Mat ImgCrop = get<0>(crop());
	Point maxLoc = get<1>(crop());
	Mat first = FirstFrame();

	Point p1((ImgCrop.cols / 4) + maxLoc.x, maxLoc.y);
	Point p2((ImgCrop.cols / 4) * 3 + maxLoc.x, maxLoc.y);
	Point p1frame((ImgCrop.cols / 4) + maxLoc.x - 3, maxLoc.y - 3);
	Point p2frame((ImgCrop.cols / 4) * 3 + maxLoc.x - 3, maxLoc.y - 3);
	Size s(1, ImgCrop.rows);
	Size sframe(7, ImgCrop.rows + 6);
	Rect ultraline1(p1, s);
	Rect ultraline2(p2, s);
	Rect ultraframe1(p1frame, sframe);
	Rect ultraframe2(p2frame, sframe);

	Point boxposition(maxLoc.x - 2, maxLoc.y - 2);
	Size boxsize(ImgCrop.cols + 4, ImgCrop.rows + 4);
	Rect box(boxposition, boxsize);

	int framenr = 0;

	float meanhistvarr1[3] = { 0 };
	float meanhistv1 = 0;

	float meanhistvarr2[3] = { 0 };
	float meanhistv2 = 0;

	int leftcount = 0;
	bool left = false;

	int rightcount = 0;
	bool right = false;

	bool middle = false;

	VideoCapture capture("C:\\Users\\axelt\\Aalborg Universitet\\CE7-AVS 7th Semester - General\\Project\\Vattenfall-fish-open-data\\fishai_training_datasets_v4\\video\\Baseline_videos_avi\\Training_data\\fish_video32.avi");
	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to open file!" << endl;
	}

	cv::CommandLineParser parser(argc, argv, "{help h||}");

	while (true) {
		int histSize = 256;
		float range[] = { 0, 256 }; //the upper boundary is exclusive
		const float* histRange[] = { range };
		bool uniform = true, accumulate = false;
		int hist_w = 512, hist_h = 400;
		int bin_w = cvRound((double)hist_w / histSize);
		Mat histImage1(hist_h, hist_w, CV_8UC1);
		Mat histImage2(hist_h, hist_w, CV_8UC1);

		float histm1 = 0;
		float histv1 = 0;
		int histnr1 = 0;

		float histm2 = 0;
		float histv2 = 0;
		int histnr2 = 0;


		capture >> frame;
		if (frame.empty()) {
			break;
		}
		cvtColor(frame, frame, COLOR_BGR2GRAY, 1);

		// Get the designated line, and draw rectangle around it
		roi1 = frame(ultraline1);
		roi2 = frame(ultraline2);
		rectangle(frame, ultraframe1, 255, 2);
		rectangle(frame, ultraframe2, 255, 2);
		rectangle(frame, box, 255, 2);

		// Calculate histogram
		calcHist(&roi1, 1, 0, Mat(), hist1, 1, &histSize, histRange, uniform, accumulate);
		normalize(hist1, hist1, 0, histImage1.rows, NORM_MINMAX, -1, Mat());

		calcHist(&roi2, 1, 0, Mat(), hist2, 1, &histSize, histRange, uniform, accumulate);
		normalize(hist2, hist2, 0, histImage2.rows, NORM_MINMAX, -1, Mat());

		//Plot Histogram
		for (int i = 1; i < histSize; i++)
		{
			line(histImage1, Point(bin_w * (i - 1), hist_h - cvRound(hist1.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(hist1.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);

			line(histImage2, Point(bin_w * (i - 1), hist_h - cvRound(hist2.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(hist2.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);

			//Get Mean of Histogram

			histnr1 = cvRound(hist1.at<float>(i)) + histnr1;
			histm1 = cvRound(hist1.at<float>(i)) * i + histm1;

			histnr2 = cvRound(hist2.at<float>(i)) + histnr2;
			histm2 = cvRound(hist2.at<float>(i)) * i + histm2;

		}


		histm1 = (histm1 / histnr1);
		//cout << "Mean1: " << histm1 << endl;

		histm2 = (histm2 / histnr2);
		//cout << "Mean2: " << histm2 << endl;

		//Get Variance of Histogram
		for (int i = 1; i < histSize; i++) {
			histv1 = ((i - histm1) * (i - histm1) * cvRound(hist1.at<float>(i))) + histv1;
			histv2 = ((i - histm2) * (i - histm2) * cvRound(hist2.at<float>(i))) + histv2;
		}

		histv1 = sqrt(histv1 / histnr1);
		//cout << "Variance1: " << histv1 << endl;

		histv2 = sqrt(histv2 / histnr2);
		//cout << "Variance2: " << histv2 << endl;

		//Get mean of variance
		if (framenr < 3) {
			meanhistvarr1[framenr] = histv1;
			meanhistvarr2[framenr] = histv2;
		}
		else if (framenr == 3 || framenr > 3) {
			meanhistv1 = (meanhistvarr1[0] + meanhistvarr1[1] + meanhistvarr1[2]) / 3;
			//cout << "Threshold Variance1: " << meanhistv1+2 << endl;

			meanhistv2 = (meanhistvarr2[0] + meanhistvarr2[1] + meanhistvarr2[2]) / 3;
			//cout << "Threshold Variance2: " << meanhistv2 + 2 << endl;

			// Use mean of variance to determine if fish is present
			if (histv1> meanhistv1 + 3 && histv2 > meanhistv2 + 3){
				cout << "Fish detected1" << endl;
				cout << "Fish detected2" << endl;
				if (left == false && right == true) {
					left = true;
					middle = true;
					right = false;
				}
				else if (left == true && right == false) {
					left = false;
					middle = true;
					right = true;

				}
			}
			if (histv1 > meanhistv1 + 3 && histv2 < meanhistv2 + 3) {
				cout << "Fish detected1" << endl;
				if (left == false) {
					if (middle == false) {
						middle = true;
						left = true;
					}
				}
				else if (left == true) {
					left = true;
				}
			}
			else {
				cout << "No fish detected1" << endl;
				if (left == true) {
					if (middle == true) {
						middle = false;
						left = false;
						leftcount++;
					}
				}
				else if (left == false) {
					left = false;
				}
			}
			if (histv2 > meanhistv2 + 3 && histv1 < meanhistv1 + 3) {
				cout << "Fish detected2" << endl;
				if (middle == false) {
					middle = true;
					right = true;
				}
				else if (right == true) {
					right = true;
				}
			}
			else {
				cout << "No fish detected2" << endl;
				if (right == true) {
					if (middle == true) {
						middle = false;
						right = false;
						rightcount++;
					}
				}
				else {
					right = false;
				}
			}
		}

		framenr++;

		cout << "Left count: " << leftcount << endl;
		cout << "Right count: " << rightcount << endl;
		cout << "Frame: " << framenr << endl;
		cout << "------------------------" << endl;

		namedWindow("Source image", WINDOW_NORMAL);
		imshow("Source image", frame);
		//imshow("calcHist1", histImage1);
		//imshow("calcHist2", histImage2);


		waitKey(0);

		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;
	}


	waitKey(0);
	return 0;

}