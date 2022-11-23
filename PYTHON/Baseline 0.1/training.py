import argparse
import sys

from numpy import argmax

from main import baseline
import numpy as np


class train_learn:
    def __init__(self, list_trn_parameters):
        self.threshold = 800  # threshold
        self.distance = 300  # distance
        with open('distance_evaluation', 'w') as f:  # open file
            f.write(f"Parameters : th : {self.threshold} \n")  # write accuracy
            f.write(f"distance;accuracy_training;accuracy_validation;wrong_videos\n")  # write accuracy
        self.training(list_trn_parameters)  # train parameter

    def training(self, list_trn_parameters):  # training function*
        # Training video path :
        trn_path = 'C:/Users/\julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/new_split/train/*.mp4'
        # Validation video path :
        val_path = 'C:/Users/\julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/new_split/val/*.mp4'

        dict = {'distance': [self.distance], 'threshold': [self.threshold]}  # dictionary of parameters

        for parameter in list_trn_parameters:  # for each parameter
            min_value = list_trn_parameters[parameter][0]  # min value for the parameter
            max_value = list_trn_parameters[parameter][1]  # max value for the parameter
            step = list_trn_parameters[parameter][2]  # step for the parameter
            dict[parameter] = np.arange(min_value, max_value, step).astype(int)  # create a list of values for the parameter
        self.computing_accuracy(dict['distance'], dict['threshold'], trn_path, val_path)  # compute accuracy for each parameter

    def computing_accuracy(self, distances, thresholds, trn_path, val_path):  # compute accuracy for each parameter
        list_accuracy_trn = []
        for distance in distances:  # for each distance

            for threshold in thresholds:  # for each threshold
                print(f"===========================================Distance = {distance} Threshold = {threshold}=============================")
                accuracy_trn, wrong_videos_trn = baseline(distance=distance, threshold=threshold, path=trn_path)  # compute accuracy for training videos
                accuracy_val, wrong_videos_val = baseline(distance=distance, threshold=threshold, path=val_path)  # compute accuracy for validation videos
                list_accuracy_trn.append(accuracy_trn)
                # self.training_export(distance, threshold, accuracy_trn, accuracy_val)  # export accuracy for each parameter
                self.distance_export(distance, accuracy_trn, accuracy_val, wrong_videos_trn)
        # self.learning(distances, thresholds, list_accuracy_trn)  # learning function

    def training_export(self, distance, threshold, accuracy_trn, accuracy_val):  # export accuracy
        with open('accuracies_training', 'a') as f:  # open file
            f.write(f"Distance = {distance} Threshold = {threshold} Accuracy Training= {accuracy_trn} Accuracy Validation {accuracy_val}\n")  # write accuracy

    def learning(self, list_distance, list_threshold, list_accuracy):  # learning function
        i = argmax(list_accuracy)  # index of the best accuracy
        d = list_distance[i]  # distance of the best accuracy
        t = list_threshold[i]  # threshold of the best accuracy

        print(f"LEARNINIG NEW PARAMETERS Distance1 = {d} Threshold1 = {t}")
        self.training({'distance': [d - 50, d + 50, 3], 'threshold': [t - 200, t + 200, 3]})

    def distance_export(self, distance, accuracy_trn, accuracy_val, wrong_videos):
        with open('distance_evaluation', 'a') as f:  # open file
            f.write(f"{distance};{accuracy_trn};{accuracy_val};{wrong_videos}\n")  # write accuracy


def get_arguments(args):  # get arguments

    parser = argparse.ArgumentParser()  # initialize parser
    # Adding optional argument
    parser.add_argument("-train", "--training")  # True or False
    parser.add_argument("-d", "--distance", nargs="+", type=int)  # distance values : [min, max, step]
    parser.add_argument("-th", "--threshold", nargs="+", type=int)  # Treshold values : [min, max, step]
    # Read arguments from command line
    args = parser.parse_args()  # read arguments from command line
    list_trn_parameters = {}
    train = False
    if args.training:
        train = True
    if args.distance:
        distance = args.distance
        list_trn_parameters['distance'] = distance
    if args.threshold:
        threshold = args.threshold
        list_trn_parameters['threshold'] = threshold

    return train, list_trn_parameters  # return arguments


if __name__ == '__main__':
    # -train True - d 250 500 5 - th 1000 1800 5
    Training, list_trn_parameters = get_arguments(sys.argv)  # get arguments
    if not Training:  # if no argument is given for training
        baseline()  # run baseline
    else:
        train = train_learn(list_trn_parameters)
