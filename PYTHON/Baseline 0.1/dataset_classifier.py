import export_json
import json
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
import cv2
import os
import pandas as pd
import shutil
import numpy as np
import matplotlib.gridspec as gridspec


# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def find_folder():
    "C:/Users/"
    "D:/"
    for root, subdirs, files in os.walk("D:/"):
        for d in subdirs:
            if d == "new_split":
                return os.path.join(root, d)


def get_fish_time():
    '''
    Gets the first enter frame and the last exit frame for each video
    Returns: a dictionary with the video name as key and a list with the first enter frame and the last exit frame

    '''

    anno_file = open('Annotations_full.json', 'r')
    anno_data = json.load(anno_file)
    video_range = range(len(anno_data)+4)
    fish_time_info = []
    root = find_folder()
    print(root)
    for i in video_range:
        total_line_time = 0
        try:
            index_txt = "fish_video"+str(i)+".mp4"
            video = anno_data[index_txt]
            enter_frame = video['enter_frame']
            exit_frame = video['exit_frame']

            for enter, exit in zip(enter_frame, exit_frame):
                if exit is not None:
                    total_line_time += (exit - enter)
                else:
                    path = find(index_txt, root)
                    video = cv2.VideoCapture(path)
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    total_line_time += (total_frames - enter)
        except:
            print("Video: fish_video"+ str(i)+".mp4" +" contains no Salmon")
        print("Total time salmon spends on line for " + index_txt + " = " + str(total_line_time) + " frames")
        fish_time_dict = {
            "name" : index_txt,
            "time" : total_line_time
        }
        fish_time_info.append(fish_time_dict)
    return fish_time_info

def save_time_to_json(fish_time):
    '''
    Update existing annotations json file to include the total time salmon spends on the lines for each video
    '''
    anno_file = open('Annotations_full.json', 'r+')
    anno_data = json.load(anno_file)
    for time in fish_time:
        index_txt = time['name']
        try:
            anno_data[index_txt]['total_time'] = time['time']
        except:
            print("Video: " + index_txt + " contains no Salmon")
    return anno_data

def plot_datasets_distribution(train_fast, test_fast, val_fast, train_slow, test_slow, val_slow, train_medium, test_medium, val_medium, font):

    print(train_fast, test_fast, val_fast, train_slow, test_slow, val_slow, train_medium, test_medium, val_medium)
    multiple_occlusions_video_count = 6
    multiple_occlusions_video_count_train = 2
    multiple_occlusions_video_count_test = 2
    multiple_occlusions_video_count_val = 2

    other_fish_video_count = 4
    other_fish_video_count_train = 2
    other_fish_video_count_test = 1
    other_fish_video_count_val = 1

    #labels = 'Multiple Fish/ \n Occlusions', 'Other Fish', 'Fast Fish', 'Medium Fish', 'Slow Fish'
    labels = 'Multiple Salmons/ \n Occlusions', 'Other species', 'Brief appearance', 'Average appearance', 'Extended appearance'
    sizes_train = [multiple_occlusions_video_count_train, other_fish_video_count_train, train_fast, train_medium, train_slow]
    sizes_test = [multiple_occlusions_video_count_test, other_fish_video_count_test, test_fast, test_medium, test_slow]
    sizes_val = [multiple_occlusions_video_count_val, other_fish_video_count_val, val_fast, val_medium, val_slow]
    print("sizes of train data:", sizes_train)
    print("sizes of test data:", sizes_test)
    print("sizes of val data:", sizes_val)
    explode = (0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    colors = ['yellowgreen', 'gold', 'red', 'green', 'blue']


    #fig, ax = plt.subplots(2,2)
    gs = gridspec.GridSpec(2, 2)

    #fig1, ax1, fig2, ax2, fig3, ax3 = plt.subplots(3, 1)
    pl.figure()
    ax1 = plt.subplot(gs[0, 0])
    ax1.pie(sizes_train, explode=explode, autopct=lambda pct: func(pct, sizes_train),
            shadow=True, startangle=0, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title('Training Set', fontdict=font)

    ax2 = plt.subplot(gs[1, 0])
    ax2.pie(sizes_test, explode=explode, autopct=lambda pct: func(pct, sizes_test),
            shadow=True, startangle=0, colors=colors)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_title('Test Set', fontdict=font)
    #Move plot to the right
    box = ax2.get_position()
    box.x0 = box.x0 + 0.2
    box.x1 = box.x1 + 0.2
    ax2.set_position(box)
#y=1.08
    ax3 = plt.subplot(gs[0, 1])
    ax3.pie(sizes_val, explode=explode, autopct=lambda pct: func(pct, sizes_val),
            shadow=True, startangle=0, colors=colors)
    ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax3.set_title('Validation Set', fontdict=font)
    # Place legend in the right bottom corner
    plt.legend(labels, loc='lower right', bbox_to_anchor=(1.2, -1.5), fontsize=12)


    #plt.title('Distribution of video categories in the entire dataset', y=1.08, font=font)

def add_to_dict(dict, key, class_to_add):
    #for i in dict:
    #    if i == key:
    dict[key]["classification"] = class_to_add
    return dict

def analyze_fish_time(fish_time, updated_dict):
    '''
    Analyze the total time salmon spends on the lines for each video
    '''
    # First, filter out videos that contain occlusions or multiple salmon
    #fish_time = [x for x in fish_time if updated_dict[x['name']]['classification'] != 'Occlusion/multiple fish']
    fish_time_no_occlusions = []
    for x in fish_time:
        try:
            if updated_dict[x['name']]['classification'] != 'Occlusion/multiple fish':
                fish_time_no_occlusions.append(x)
        except:
            print("Video: " + x['name'] + " contains no Salmon")
    # Order the fish_time list by amount of time salmon spends on the line
    fish_time_no_occlusions.sort(key=lambda x: x['time'], reverse=True,)
    x_labels = [x['name'] for x in fish_time_no_occlusions]
    y_values = pd.Series([x['time'] for x in fish_time_no_occlusions])
    print("mean frames on line: " + str((np.mean(y_values))))
    median_frames_on_line = np.median(y_values)
    print("median frames on line: " + str(median_frames_on_line))
    # Using median is better, since it is less affected by outliers

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    plt.rc('font', **font)
    # Then, plot the time salmon spends on the line for each video
    #plt.figure(figsize=(12, 8))
    #fig = plt.bar(range(len(fish_time_no_occlusions)), [x['time'] for x in fish_time_no_occlusions])
    color_list = []
    fast_threshold = median_frames_on_line * 0.75
    slow_threshold = median_frames_on_line * 1.25
    #fast_threshold = 34
    #slow_threshold = 71.25
    print("fast threshold: " + str(fast_threshold))
    print("slow threshold: " + str(slow_threshold))
    # box plot
    #bp = plt.boxplot(y_values)
    #print(bp)
    #plt.title('Distribution of time salmon spends on the line')
    #plt.ylabel('Frames on line')
    #plt.xlabel('Video')
    #plt.xticks([1], ['Salmon'])
    #plt.show()


    x_index = 0
    for y in y_values:
        # Fast fish are salmon that spend less than 25% of the median time on the line
        if y < fast_threshold:
            add_to_dict(updated_dict, x_labels[x_index], "Fast fish")
            color_list.append('red')
        # Slow fish are salmon that spend more than 25% of the median time on the line
        elif y > slow_threshold:
            add_to_dict(updated_dict, x_labels[x_index], "Slow fish")
            color_list.append('blue')
        # Medium fish are salmon that spend between 25% and 125% of the median time on the line
        else:
            add_to_dict(updated_dict, x_labels[x_index], "Normal fish")
            color_list.append('green')
        x_index += 1
    fig = y_values.plot(kind='bar', color=color_list)
    fig.set_xticklabels(x_labels, rotation=45, ha='right')
    fig.bar_label(fig.containers[0], label_type='edge')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.935, bottom=0.17)

    plt.xlabel('Video Number', font=font)
    plt.ylabel('Time (frames)', font=font)
    plt.title('Time Salmon Spends on intersecting with LOI', font=font)
    # Set legend of each color
    labels = [r'Brief appearance', 'Average appearance', 'Extended appearance']

    plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor='none'),
                        plt.Rectangle((0, 0), 1, 1, fc="green", edgecolor='none'),
                        plt.Rectangle((0, 0), 1, 1, fc="blue", edgecolor='none')],
               labels=labels)
    fig.set_xticklabels(x_labels)

    # Find how many salmon are fast, medium, and slow in the train, test, and validation sets
    split_file = json.load(open('test_train_val_split.json', 'r'))
    train = split_file['train']
    test = split_file['test']
    val = split_file['val']
    for i in range(len(train['file_name'])):
        train['file_name'][i] = train['file_name'][i]+'.mp4'
    for i in range(len(test['file_name'])):
        test['file_name'][i] = test['file_name'][i]+'.mp4'
    for i in range(len(val['file_name'])):
        val['file_name'][i] = val['file_name'][i]+'.mp4'
    train_fast = 0
    train_medium = 0
    train_slow = 0
    test_fast = 0
    test_medium = 0
    test_slow = 0
    val_fast = 0
    val_medium = 0
    val_slow = 0
    for x in fish_time_no_occlusions:
        if x['time'] < fast_threshold:
            if x['name'] in train['file_name']:
                print(x['name'] + " is a fast fish in the train set")
                train_fast += 1
            elif x['name'] in test['file_name']:
                #print(x['name'] + " is a fast fish in the test set")
                test_fast += 1
            elif x['name'] in val['file_name']:
                #print(x['name'] + " is a fast fish in the validation set")
                val_fast += 1
        elif x['time'] > slow_threshold:
            #print(x['name'], train['file_name'])
            if x['name'] in train['file_name']:
                print(x['name'] + " is a slow fish in the train set")
                train_slow += 1
            elif x['name'] in test['file_name']:
                #print(x['name'] + " is a slow fish in the test set")
                test_slow += 1
            elif x['name'] in val['file_name']:
                #print(x['name'] + " is a slow fish in the validation set")
                val_slow += 1
        else:
            if x['name'] in train['file_name']:
                print(x['name'] + " is a medium fish in the train set")
                train_medium += 1
            elif x['name'] in test['file_name']:
                #print(x['name'] + " is a medium fish in the test set")
                test_medium += 1
            elif x['name'] in val['file_name']:
                #print(x['name'] + " is a medium fish in the validation set")
                val_medium += 1
    print("Train Fast: " + str(train_fast))
    print("Test Fast: " + str(test_fast))
    print("Val Fast: " + str(val_fast))
    print(10*'-')
    print("Train Medium: " + str(train_medium))
    print("Test Medium: " + str(test_medium))
    print("Val Medium: " + str(val_medium))
    print(10*'-')
    print("Train Slow: " + str(train_slow))
    print("Test Slow: " + str(test_slow))
    print("Val Slow: " + str(val_slow))

    # Plot the number of fast, medium, and slow fish in the train, test, and validation sets
    index = ['Train', 'Test', 'Validation']
    df = pd.DataFrame({'$ROI_t < th_{fast}$': [train_fast, test_fast, val_fast],
                          '$th_{fast}\geq ROI_t \leq th_{slow}$': [train_medium, test_medium, val_medium],
                          '$ROI_t > th_{slow}$': [train_slow, test_slow, val_slow]}, index=index)
    df.plot(kind='bar', stacked=True, rot = 0, color = ['red', 'green', 'blue'])
    plt.xlabel('Dataset', font=font)
    plt.ylabel('Number of videos', font=font)
    plt.title('Number of Fast, Medium, and Slow Fish in Each Dataset', font=font)
    labels = [r'$ROI_t < th_{fast}$', '$th_{fast}\geq ROI_t \leq th_{slow}$', '$ROI_t > th_{slow}$']
    plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor='none'),
                            plt.Rectangle((0, 0), 1, 1, fc="green", edgecolor='none'),
                            plt.Rectangle((0, 0), 1, 1, fc="blue", edgecolor='none')],
                labels=labels)
    plt.legend()
    # Plot a pie chart of the distribution of Other_fish, multiple fish/ occlusions, fast, medium, and slow fish in the train, test, and validation sets

    multiple_occlusions_video_count = 6
    other_fish_video_count = 4
    fast_fish_video_count = train_fast + test_fast + val_fast
    medium_fish_video_count = train_medium + test_medium + val_medium
    slow_fish_video_count = train_slow + test_slow + val_slow
    labels = 'Multiple Salmons/ \n Occlusions', 'Other species', r'$ROI_t < th_{fast}$', '$th_{fast}\geq ROI_t \leq th_{slow}$ ', '$ROI_t > th_{slow}$'
    sizes = [multiple_occlusions_video_count, other_fish_video_count, fast_fish_video_count, medium_fish_video_count, slow_fish_video_count]
    explode = (0.1, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    colors = ['yellowgreen', 'gold', 'red', 'green', 'blue']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct=lambda pct: func(pct, sizes),
            shadow=True, startangle=0, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Distribution of video categories in the entire dataset', y=1.08, font=font)

    plot_datasets_distribution(train_fast, test_fast, val_fast, train_slow, test_slow, val_slow, train_medium,test_medium, val_medium, font)



if __name__ == '__main__':
    print(find_folder())
    fish_time = get_fish_time()
    updated_dict = save_time_to_json(fish_time)
    analyze_fish_time(fish_time, updated_dict)
    with open('Annotations_full.json', 'w') as file:
        file.seek(0)
        json.dump(updated_dict, file, indent=4)
    print("Finished getting fish time")
    plt.show()