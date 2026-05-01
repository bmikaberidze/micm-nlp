import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def calc_confusion_matrix(predictions, true_labels, config, eval_path):
    # # Flatten the lists for confusion matrix computation
    predictions = predictions.flatten()
    true_labels = true_labels.flatten()
    # Cast label names to integers if true labels are ints
    label_names = config.ds.label.names
    labels = list(range(len(label_names))) if np.issubdtype(true_labels.dtype, np.integer) else label_names
    # Compute the confusion matrix and Save the confusion matrix plot
    # print('labels:', labels, 'true_labels:', true_labels, 'predictions:', predictions)
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
    cm_path = f'{eval_path}/confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
