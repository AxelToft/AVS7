import matplotlib.pyplot as plt
import numpy as np
import plot_results as pr
if __name__ == "__main__":
    # Plot confusion matrix for YOLO
    yolo_cm = np.array([[ 10, 0],[2, 0]])
    pr.plot_confusion_matrix(yolo_cm, "YOLOv5/DeepSORT", True)

    # Plot confusion matrix for 2-Line logic
    logic_cm = np.array([[ 8, 1],[3, 0]])
    pr.plot_confusion_matrix(logic_cm,"2-Line Logic", True)

    # Plot confusion matrix for OFCD
    ofcd_cm = np.array([[7, 2],[28, 0]])
    pr.plot_confusion_matrix(ofcd_cm, "Contour tracking", True)