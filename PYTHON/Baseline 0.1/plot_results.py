import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle
import seaborn as sns


def plot_precision_recall(test_classes, y_pred, model_name, show_plot=False):
    """
    Plot the precision-recall curve for the model
    :param test_classes: List of actual classes
    :param y_pred: List of predicted classes
    :param model_name: Name of the model
    :param show_plot: Boolean to show the plot
    :return: None
    """
    # Find the unique classes:
    n_classes = np.unique(test_classes)
    # Convert classes and predicted results to binary format
    #y_pred_bin = label_binarize(y_pred, classes=n_classes)
    test_classes_bin = label_binarize(test_classes, classes=n_classes)
    y_pred_bin = y_pred
    # Find the total number of unique classes
    n_classes = y_pred_bin.shape[1]
    """average_precision = 0
    for i in range(n_classes):
        precision_class = metrics.precision_score(test_classes_bin[:, i], y_pred_bin[:,i])
        print("Precision MLP: {}".format(precision_class), "for class: {}".format(i))
        average_precision += precision_class
    average_precision = average_precision / n_classes
    print("Average Precision MLP: {}".format(average_precision))
    print("Alternative Average Precision MLP: {}".format(metrics.average_precision_score(test_classes_bin, y_pred_bin)))"""

    # Set plot colors
    class_name = ["alarm", "apple", "angel", "anvil", "banana"]
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
    # Set plot size
    _, ax = plt.subplots(figsize=(10, 8))
    # For each class find the precision and recall and average precision
    precision = dict()
    recall = dict()
    th = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], th[i] = precision_recall_curve(test_classes_bin[:, i], y_pred_bin[:, i])
        average_precision[i] = average_precision_score(test_classes_bin[:, i], y_pred_bin[:, i])

    # A "mean-average": quantifying score on all classes jointly
    precision["mean"], recall["mean"], _ = precision_recall_curve(
        test_classes_bin.ravel(), y_pred_bin.ravel()
    )
    average_precision["mean"] = average_precision_score(test_classes_bin, y_pred_bin, average="weighted")

    display = PrecisionRecallDisplay(
        recall=recall["mean"],
        precision=precision["mean"],
        average_precision=average_precision["mean"],
    )
    display.plot(ax=ax, name="Mean-average precision-recall", color="gold")

    # Plot the precision-recall curve for each class
    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {class_name[i]}", color=color)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class for {}".format(model_name))
    plt.tight_layout()
    plt.savefig("figs/precision_recall_{}.png".format(model_name))
    if show_plot:
        plt.show()

def plot_confusion_matrix(cm, model_name, show_plot):
    """
    Plot the confusion matrix
    :param cm: Confusion matrix
    :param model_name: Name of the model
    :param show_plot: Boolean to show the plot
    :return: None
    """

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}

    heatmap_font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}
    plt.rc('font', **heatmap_font)
    # Set plot size
    _, ax = plt.subplots(figsize=(10, 8))
    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_title("Confusion Matrix of {}".format(model_name), font=font, pad=20)
    ax.set_xlabel("Predicted", font=font)
    ax.set_ylabel("Actual", font=font)
    class_name = ["Positive", "Negative"]
    ax.xaxis.set_ticklabels(class_name, font=font)
    ax.yaxis.set_ticklabels(class_name, font=font)
    plt.tight_layout()
    if model_name == "YOLOv5/DeepSORT":
        plt.savefig("confusion_matrix_yolov5_deepsort.pdf")
    else:
        plt.savefig("confusion_matrix_{}.pdf".format(model_name))
    if show_plot:
        plt.show()

