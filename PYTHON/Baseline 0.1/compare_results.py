import json
import matplotlib.pyplot as plt
import evaluation
import numpy as np
class solution_results:
    def __init__(self, val_results, test_results, gt_data, name):
        self.val_results = val_results
        self.test_results = test_results
        self.gt_data = gt_data
        self.name = name

        # Variables for storing the accuracy for counting the sum.
        self.count_sum_accuracy_val = self.get_count_accuracy_val()
        self.count_sum_accuracy_test = self.get_count_accuracy_test()

        # Variables for storing the accuracy for the intermediate counting.
        self.intermediate_count_accuracy_val = self.get_intermediate_count_accuracy_val()
        self.intermediate_count_accuracy_test = self.get_intermediate_count_accuracy_test()

        # Variables for storing the accuracy for the intermediate counting of each category.
        self.intermediate_count_accuracy_test_categories = self.get_intermediate_count_accuracy_test_categories()
        self.intermediate_count_accuracy_val_categories = self.get_intermediate_count_accuracy_val_categories()

        # Variables for storing the accuracy for counting the sum of each category.
        self.count_sum_accuracy_val_categories = self.get_count_accuracy_val_accuracy()
        self.count_sum_accuracy_test_categories = self.get_count_accuracy_test_accuracy()

        # Variables for storing the list of sum accuracies for all categories for test and val.
        self.count_sum_accuracy_val_categories_list = self.combine_category_results(self.count_sum_accuracy_val_categories)
        self.count_sum_accuracy_test_categories_list = self.combine_category_results(self.count_sum_accuracy_test_categories)

        # Variables for storing the list of intermediate accuracies for all categories for test and val.
        self.intermediate_count_accuracy_val_categories_list = self.combine_category_results(self.intermediate_count_accuracy_val_categories)
        self.intermediate_count_accuracy_test_categories_list = self.combine_category_results(self.intermediate_count_accuracy_test_categories)


    def combine_category_results(self, results):
        classes_list = ['Occlusion/multiple fish', 'Other_fish', 'Fast fish', 'Normal fish', 'Slow fish']
        combined_results = []
        for category in classes_list:
            combined_results.append(results[category]['accuracy'])
        return combined_results


    def get_count_accuracy_val(self):
        sum, negative_diff_count, positive_diff_count, wrong_count, wrong_count_i = self.get_count_accuracy(self.val_results, mode="sum")
        return 1 - (negative_diff_count + positive_diff_count) / len(self.val_results)

    def get_count_accuracy_test(self):
        sum, negative_diff_count, positive_diff_count, wrong_count, wrong_count_i = self.get_count_accuracy(
            self.test_results, mode="sum")
        return 1 - (negative_diff_count + positive_diff_count) / len(self.test_results)

    def get_count_accuracy_test_accuracy(self):
        return self.get_count_accuracy(self.test_results, mode="categories")

    def get_count_accuracy_val_accuracy(self):
        return self.get_count_accuracy(self.val_results, mode="categories")

    def get_intermediate_count_accuracy_test(self):
        print(5*"-" + " Finding intermediate accuracy for "+ self.name +" test " + 5*"-")
        return self.get_intermediate_count_accuracy(self.test_results, mode="sum")

    def get_intermediate_count_accuracy_val(self):
        print(5 * "-" + " Finding intermediate accuracy for "+ self.name +" val " + 5 * "-")
        return self.get_intermediate_count_accuracy(self.val_results, mode="sum")

    def get_intermediate_count_accuracy_test_categories(self):
        print(5 * "-" + " Finding intermediate accuracy for categories in "+ self.name +" test " + 5 * "-")
        return self.get_intermediate_count_accuracy(self.test_results, mode="categories")

    def get_intermediate_count_accuracy_val_categories(self):
        print(5 * "-" + " Finding intermediate accuracy for categories in val "+ self.name +" " + 5 * "-")
        return self.get_intermediate_count_accuracy(self.val_results, mode="categories")


    def get_count_accuracy(self, result_list, mode):
        sum = 0
        negative_diff_count = 0
        positive_diff_count = 0
        wrong_count = []
        wrong_count_i = []
        i = 0
        if mode == "categories":
            count_seq_count = {
                "Occlusion/multiple fish": {"sum": 0, "count": 0, "accuracy": 0},
                "Other_fish": {"sum": 0, "count": 0, "accuracy": 0},
                "Fast fish": {"sum": 0, "count": 0, "accuracy": 0},
                "Normal fish": {"sum": 0, "count": 0, "accuracy": 0},
                "Slow fish": {"sum": 0, "count": 0, "accuracy": 0}
            }
        for result_video in result_list['results']:
            try:
                gt_video = self.gt_data[result_video['video']]

            except KeyError:
                anno_data = {
                    result_video['video']: {
                        "fish_count_frames": [],
                        'fish_count': 0,
                        'enter_frame': [],
                        'exit_frame': [],
                        "classification": "Other_fish"
                    }
                }
                gt_video = anno_data[result_video['video']]

            # Get frame count for videos
            if mode == "sum":
                sum += abs(gt_video['fish_count'] - result_video['fish_count'])
                if abs(gt_video['fish_count'] - result_video['fish_count']) > 0:
                    if gt_video['fish_count'] - result_video['fish_count'] < 0:
                        negative_diff_count += 1
                    else:
                        positive_diff_count += 1
                    wrong_count.append(result_video['video'])
                    wrong_count_i.append(i)
            elif mode == "categories":
                if abs(gt_video['fish_count'] - result_video['fish_count']) > 0:
                    count_seq_count[gt_video['classification']]["sum"] += 1
                count_seq_count[gt_video['classification']]["count"] += 1
            i += 1
        if mode == "sum":
            return sum, negative_diff_count, positive_diff_count, wrong_count, wrong_count_i
        elif mode == "categories":
            count_seq_count['Occlusion/multiple fish']['accuracy'] = (1 - count_seq_count['Occlusion/multiple fish']['sum'] / count_seq_count['Occlusion/multiple fish']['count'])*100
            count_seq_count['Other_fish']['accuracy'] = (1 - count_seq_count['Other_fish']['sum'] / count_seq_count['Other_fish']['count'])*100
            count_seq_count['Fast fish']['accuracy'] = (1 - count_seq_count['Fast fish']['sum'] / count_seq_count['Fast fish']['count'])*100
            count_seq_count['Normal fish']['accuracy'] = (1 - count_seq_count['Normal fish']['sum'] / count_seq_count['Normal fish']['count'])*100
            count_seq_count['Slow fish']['accuracy'] = (1 - count_seq_count['Slow fish']['sum'] / count_seq_count['Slow fish']['count'])* 100
            return count_seq_count

    def get_intermediate_count_accuracy(self, result_list, mode):
        count_seq_sum = 0
        if mode == "categories":
            count_seq_count = {
                "Occlusion/multiple fish": {"sum": 0, "count": 0, "accuracy": 0},
                "Other_fish": {"sum": 0, "count": 0, "accuracy": 0},
                "Fast fish": {"sum": 0, "count": 0, "accuracy": 0},
                "Normal fish": {"sum": 0, "count": 0, "accuracy": 0},
                "Slow fish": {"sum": 0, "count": 0, "accuracy": 0}
            }
        for result_video in result_list['results']:
            try:
                gt_video = self.gt_data[result_video['video']]

            except KeyError:
                #print("Video not found in annotations")
                #print("Assuming its Other_fish, so making a fake dict for it")
                anno_data = {
                    result_video['video']: {
                        "fish_count_frames": [],
                        'fish_count': 0,
                        'enter_frame': [],
                        'exit_frame': [],
                        "classification": "Other_fish"
                    }
                }
                gt_video = anno_data[result_video['video']]
            if mode == "sum":
                if gt_video['fish_count_frames'] != result_video['fish_count_frames']:
                    count_seq_sum += 1
                    print("fish_count_frames not equal for video: ", result_video['video'], " ground truth sequence: ",gt_video['fish_count_frames'], " found sequence is: ", result_video['fish_count_frames'])
            elif mode == "categories":
                if gt_video['fish_count_frames']!= result_video['fish_count_frames']:
                    count_seq_count[gt_video['classification']]['sum'] += 1
                    #print("fish_count_frames not equal for video: ", result_video['video'], " ground truth sequence: ",gt_video['fish_count_frames'], " found sequence is: ", result_video['fish_count_frames'])
                count_seq_count[gt_video['classification']]['count'] += 1
                pass
        if mode == "sum":
            return count_seq_sum
        if mode == "categories":
            count_seq_count['Occlusion/multiple fish']['accuracy'] = (1 - count_seq_count['Occlusion/multiple fish']['sum'] / count_seq_count['Occlusion/multiple fish']['count'])*100
            count_seq_count['Other_fish']['accuracy'] = (1 - count_seq_count['Other_fish']['sum'] / count_seq_count['Other_fish']['count'])*100
            count_seq_count['Fast fish']['accuracy'] = (1 - count_seq_count['Fast fish']['sum'] / count_seq_count['Fast fish']['count'])*100
            count_seq_count['Normal fish']['accuracy'] = (1 - count_seq_count['Normal fish']['sum'] / count_seq_count['Normal fish']['count'])*100
            count_seq_count['Slow fish']['accuracy'] = (1 - count_seq_count['Slow fish']['sum'] / count_seq_count['Slow fish']['count'])* 100
            return count_seq_count


def load_data():
    root = "Results/"
    deepsort_data_val = json.load(open(root+'deepsort_val.json', 'r'))
    deepsort_data_test = json.load(open(root+'deepsort_test.json', 'r'))

    two_line_v1_data_val = json.load(open(root+'two_line_v1_val.json', 'r'))
    two_line_v1_data_test = json.load(open(root+'two_line_v1_test.json', 'r'))

    two_line_v2_data_val = json.load(open(root+'two_line_v2_val.json', 'r'))
    two_line_v2_data_test = json.load(open(root+'two_line_v2_test.json', 'r'))

    gt_data = json.load(open('Annotations_full.json', 'r'))

    deepsort_data = solution_results(deepsort_data_val, deepsort_data_test, gt_data, "Deepsort")
    two_line_v1_data = solution_results(two_line_v1_data_val, two_line_v1_data_test, gt_data, "Two line v1")
    two_line_v2_data = solution_results(two_line_v2_data_val, two_line_v2_data_test, gt_data, "Two line v2")
    #two_line_v1_data = None
    #two_line_v2_data = None


    return deepsort_data, two_line_v1_data, two_line_v2_data


def plot_results(deepsort_data, two_line_v1_data, two_line_v2_data):
    '''
    Plot the results as a bar chart for validation and test set. The x-axis is the 5 categories of video types and
    the y-axis is the accuracy of the solution for each category (in percentage).
    Args:
        deepsort_data:
        two_line_v1_data:
        two_line_v2_data:

    Returns:
    '''
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    plt.rc('font', **font)

    #Prepare the data:
    labels = ['Multiple Salmons/ \n Occlusions', 'Other species', r'$ROI_t < th_{fast}$', '$th_{fast}\geq ROI_t \leq th_{slow}$ ', '$ROI_t > th_{slow}$']
    classes_list = ['Occlusion/multiple fish', 'Other_fish', 'Fast fish', 'Normal fish', 'Slow fish']

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    # Plotting the intermediate count accuracy for categories in validation set
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.grid(axis='y')
    rects1 = ax.bar(x - width, two_line_v1_data.intermediate_count_accuracy_val_categories_list, width, label='Two-line')
    rects2 = ax.bar(x, two_line_v2_data.intermediate_count_accuracy_val_categories_list, width, label='OFCD')
    rects3 = ax.bar(x + width, deepsort_data.intermediate_count_accuracy_val_categories_list, width, label='DeepSORT')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage accuracy', font=font)
    ax.set_title('Solutions intermediate counting accuracy for each category in the validation set', font=font)
    ax.set_xticks(x);ax.set_xticklabels(labels)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.126)

    # Plotting the intermediate count accuracy for categories in test set
    fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.grid(axis='y')
    rects1 = ax.bar(x - width, two_line_v1_data.intermediate_count_accuracy_test_categories_list, width, label='Two-line')
    rects2 = ax.bar(x, two_line_v2_data.intermediate_count_accuracy_test_categories_list, width,label='OFCD')
    rects3 = ax.bar(x + width, deepsort_data.intermediate_count_accuracy_test_categories_list, width, label='DeepSORT')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage accuracy', font=font)
    ax.set_title('Solutions intermediate counting accuracy for each category in the test set', font=font)
    ax.set_xticks(x);ax.set_xticklabels(labels)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.126)

    # Plotting the sum count accuracy for categories in validation set
    fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.grid(axis='y')
    rects1 = ax.bar(x - width, two_line_v1_data.count_sum_accuracy_val_categories_list, width, label='Two-line')
    rects2 = ax.bar(x, two_line_v2_data.count_sum_accuracy_val_categories_list, width, label='OFCD')
    rects3 = ax.bar(x + width, deepsort_data.count_sum_accuracy_val_categories_list, width, label='DeepSORT')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage accuracy', font=font)
    ax.set_title('Solutions sum counting accuracy for each category in the validation set', font=font)
    ax.set_xticks(x);ax.set_xticklabels(labels)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.126)


    # Plotting the sum count accuracy for categories in test set
    fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.grid(axis='y')
    rects1 = ax.bar(x - width*1.1, two_line_v1_data.count_sum_accuracy_test_categories_list, width, label='Two-line')
    rects2 = ax.bar(x, two_line_v2_data.count_sum_accuracy_test_categories_list, width, label='OFCD')
    rects3 = ax.bar(x + width*1.1, deepsort_data.count_sum_accuracy_test_categories_list, width, label='DeepSORT')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage accuracy', font=font)
    ax.set_title('Solutions sum counting accuracy for each category in the test set', font=font)
    ax.set_xticks(x);ax.set_xticklabels(labels)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.126)
    plt.show()


    pass
if __name__ == '__main__':
    deepsort_data, two_line_v1_data, two_line_v2_data = load_data()
    print("Deepsort val accuracy: ", deepsort_data.count_sum_accuracy_val)
    print("Deepsort test accuracy: ", deepsort_data.count_sum_accuracy_test)
    print("Deepsort val intermediate accuracy: ", deepsort_data.intermediate_count_accuracy_val)
    print("Deepsort test intermediate accuracy: ", deepsort_data.intermediate_count_accuracy_test)

    print("Two line v1 val accuracy: ", two_line_v1_data.count_sum_accuracy_val)
    print("Two line v1 test accuracy: ", two_line_v1_data.count_sum_accuracy_test)
    print("Two line v1 val intermediate accuracy: ", two_line_v1_data.intermediate_count_accuracy_val)
    print("Two line v1 test intermediate accuracy: ", two_line_v1_data.intermediate_count_accuracy_test)
    plot_results(deepsort_data, two_line_v1_data, two_line_v2_data)





