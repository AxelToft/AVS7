'''

'''
import argparse
import os
import sys
from main import baseline
import numpy as np


def training(list_trn_parameters):  # training function*
    #TODO : choose range of values which give best parameters then start again.
    # Training video path :
    trn_path = 'C:/Users/\julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/new_split/train/*.mp4'
    # Validation video path :
    val_path = 'C:/Users/\julie/Aalborg Universitet/CE7-AVS 7th Semester - Documents/General/Project/Vattenfall-fish-open-data/fishai_training_datasets_v4/video/Baseline_videos_mp4_full/new_split/val/*.mp4'

    if os.path.exists('accuracies_training'):  # if file exists, delete  it
        os.remove('accuracies_training')  # delete file

    dict = {'distance': [300], 'threshold': [1000]}  # dictionary of parameters

    for parameter in list_trn_parameters :  # for each parameter
        min_value = list_trn_parameters[parameter][0]  # min value for the parameter
        max_value = list_trn_parameters[parameter][1]  # max value for the parameter
        step = list_trn_parameters[parameter][2]  # step for the parameter
        dict[parameter] = np.linspace(min_value, max_value, step).astype(int)  # create a list of values for the parameter

    computing_accuracy(dict['distance'], dict['threshold'], trn_path, val_path)  # compute accuracy for each parameter


def computing_accuracy(distances, thresholds, trn_path, val_path):  # compute accuracy for each parameter
    for distance in distances:  # for each distance

        for threshold in thresholds:  # for each threshold

            accuracy_trn = baseline(distance=distance, threshold=threshold, path=trn_path)  # compute accuracy for training videos
            accuracy_val = baseline(distance=distance, threshold=threshold, path=val_path)  # compute accuracy for validation videos
            training_export(distance, threshold, accuracy_trn, accuracy_val)  # export accuracy for each parameter


def training_export(distance, threshold, accuracy_trn, accuracy_val):  # export accuracy
    with open('accuracies_training', 'a') as f:  # open file
        f.write(f"Distance = {distance} Threshold = {threshold} Accuracy Training= {accuracy_trn} Accuracy Validation {accuracy_val}\n")  # write accuracy


def get_arguments(args):  # get arguments

    # options to return
    Training = None  # what to train
    min_value = 0  # min value for the parameter
    max_value = 1  # max value for the parameter
    step = 0.1  # step for the parameter
    # Options
    # Initialize parser
    parser = argparse.ArgumentParser()  # initialize parser
    # Adding optional argument
    parser.add_argument("-train","--training")  # True or False
    parser.add_argument("-d", "--distance", nargs="+",type=int)  # distance values : [min, max, step]
    parser.add_argument("-th", "--threshold", nargs="+",type=int)  # Treashold values : [min, max, step]
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
    # -train True -d 250 400 2 -th 1200 1800 2
    Training, list_trn_parameters = get_arguments(sys.argv)  # get arguments
    if not Training:  # if no argument is given for training
        baseline()  # run baseline
    else:

        training(list_trn_parameters)  # train parameter
