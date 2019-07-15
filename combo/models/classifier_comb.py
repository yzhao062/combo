# -*- coding: utf-8 -*-
"""A collection of combination methods for combining classifier
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause


import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_array
from sklearn.utils import column_or_1d
# noinspection PyProtectedMember
from sklearn.utils import shuffle
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.testing import assert_equal
from ..utils.utility import check_parameter

from abc import ABC, abstractmethod


class BaseClassiferAggregator(ABC):

    @abstractmethod
    def __init__(self, classifiers, method='average', pre_fitted=False):
        assert (isinstance(classifiers, (list)))
        self.classifiers_ = classifiers


    @abstractmethod
    def fit(self, X, y):
        """Fit detector.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """


    # predict using a group of clfs
    def ensemble_predict(X, clfs, weights=None):
        all_scores = np.zeros([X.shape[0], len(clfs)])
        for c in range(len(clfs)):
            clf = clfs[c]
            all_scores[:, c] = clf.predict(X)

        if weights is not None:
            all_scores = all_scores * weights
        return np.mean(all_scores, axis=1)

    # predict probability using a group of clfs
    def ensemble_predict_proba(X, clfs, weights=None):
        all_scores_0 = np.zeros([X.shape[0], len(clfs)])
        all_scores_1 = np.zeros([X.shape[0], len(clfs)])

        for c in range(len(clfs)):
            clf = clfs[c]
            all_scores_0[:, c] = clf.predict_proba(X)[:, 0]
            all_scores_1[:, c] = clf.predict_proba(X)[:, 1]

        if weights is not None:
            all_scores_0 = all_scores_0 * weights
            all_scores_1 = all_scores_1 * weights

        prob_scores = np.zeros([X.shape[0], 2])

        prob_scores[:, 0] = np.mean(all_scores_0, axis=1)
        prob_scores[:, 1] = np.mean(all_scores_1, axis=1)
        return prob_scores


    def __len__(self):
        """Returns the number of estimators in the ensemble."""
        return len(self.classifiers_)

    def __getitem__(self, index):
        """Returns the index'th estimator in the ensemble."""
        return self.classifiers_[index]

    def __iter__(self):
        """Returns iterator over estimators in the ensemble."""
        return iter(self.classifiers_)
