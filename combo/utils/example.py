# -*- coding: utf-8 -*-
"""Utility functions for running examples
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice


def visualize_clusters(model_name, X, predicted_labels, show_figure=True,
                       save_figure=False):  # pragma: no cover
    """Utility function for visualizing the results in examples.
    Internal use only.

    Parameters
    ----------
    model_name : str
        The name of the clustering method.

    X : numpy array of shape (n_samples, n_features)
        The input samples.

    predicted_labels : numpy array of shape (n_samples, n_features)
        The predicted labels of the input samples.

    show_figure : bool, optional (default=True)
        If set to True, show the figure.

    save_figure : bool, optional (default=False)
        If set to True, save the figure to the local.

    """
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(predicted_labels) + 1))))

    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[predicted_labels])
    plt.title(model_name)
    plt.xticks(())
    plt.yticks(())

    if save_figure:
        plt.savefig('{clf_name}.png'.format(clf_name=model_name), dpi=300)

    if show_figure:
        plt.show()
