"""Confusion-matrix rendering for classification evaluations.

``calc_confusion_matrix`` runs when ``task.preproc_rules.calc_confusion_matrix`` is
set and writes ``confusion_matrix.png`` into the run's evaluation directory. Axis
labels come from ``ds.label.names``, or from integer ids when the true labels are
integers.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def calc_confusion_matrix(predictions, true_labels, config, eval_path):
    """Compute a confusion matrix and write it to ``<eval_path>/confusion_matrix.png``.

    Both arrays are flattened first, so this works for per-token and per-example
    predictions alike. The label axis comes from ``config.ds.label.names``: integer
    labels are plotted as indices into that list, string labels as the names
    themselves.

    :param predictions: predicted labels.
    :param true_labels: gold labels, same shape.
    :param config: the run config; ``ds.label.names`` supplies the axis.
    :param eval_path: directory to write the PNG into.
    """
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
