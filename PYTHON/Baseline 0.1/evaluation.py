import json

def evaluate_frame_count():
    # Annotations file is not saved in the same format as the results file, so we need to index it differently
    anno_file = open('Annotations_full.json', 'r')
    json_file = open('results.json', 'r')
    # Get video names for indexing annotations file
    anno_data = json.load(anno_file)
    data = json.load(json_file)
    # Get annotations for video
    sum = 0
    negative_diff_count = 0
    positive_diff_count = 0
    for eval_video in data['results']:
        # Get ground truth for video
        gt_video = anno_data[eval_video['video']]
        # Get frame count for videos and compare
        sum += abs(gt_video['fish_count'] - eval_video['fish_count'])
        if abs(gt_video['fish_count'] - eval_video['fish_count']) > 0:
            if gt_video['fish_count'] - eval_video['fish_count'] < 0:
                negative_diff_count -= 1
            else:
                positive_diff_count += 1

    print('Average frame count difference: ' + str(sum / len(data['results'])))
    print('Number of videos with negative difference: ' + str(negative_diff_count))
    print('Number of videos with positive difference: ' + str(positive_diff_count))
    print('Percentage of videos counted right: ' + str(1-(negative_diff_count + positive_diff_count)/len(data['results'])))

def evaluate_frame_time():
    pass