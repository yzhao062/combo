# -*- coding: utf-8 -*-
"""Utility functions for manipulating data
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause


import numpy as np
from warnings import warn

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state
from sklearn.utils import check_consistent_length

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

MAX_INT = np.iinfo(np.int32).max


def evaluate_print(clf_name, y, y_pred):
    """Utility function for evaluating and printing the results for examples.
    Default metrics include accuracy, roc, and F1 score

    Parameters
    ----------
    clf_name : str
        The name of the estimator.

    y : list or numpy array of shape (n_samples,)
        The ground truth.

    y_pred : list or numpy array of shape (n_samples,)
        The raw scores as returned by a fitted model.

    """

    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)

    print('{clf_name} Accuracy:{acc}, ROC:{roc}, F1:{f1}'.format(
        clf_name=clf_name,
        acc=np.round(accuracy_score(y, y_pred), decimals=4),
        roc=np.round(roc_auc_score(y, y_pred), decimals=4),
        f1=np.round(f1_score(y, y_pred), decimals=4)))
