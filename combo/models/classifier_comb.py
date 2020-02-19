# -*- coding: utf-8 -*-
"""A collection of methods for combining classifiers
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause


import numpy as np

from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import column_or_1d

from pyod.utils.utility import check_parameter

from .base import BaseAggregator
from .score_comb import average, maximization, majority_vote, median

from ..utils.utility import score_to_proba


class SimpleClassifierAggregator(BaseAggregator):
    """A collection of simple classifier combination methods.

    Parameters
    ----------
    base_estimators: list or numpy array (n_estimators,)
        A list of base classifiers.

    method : str, optional (default='average')
        Combination method: {'average', 'maximization', 'majority vote',
        'median'}. Pass in weights of classifier for weighted version.

    threshold : float in (0, 1), optional (default=0.5)
        Cut-off value to convert scores into binary labels.

    weights : numpy array of shape (1, n_classifiers)
        Classifier weights.

    pre_fitted : bool, optional (default=False)
        Whether the base classifiers are trained. If True, `fit`
        process may be skipped.
    """

    def __init__(self, base_estimators, method='average', threshold=0.5,
                 weights=None, pre_fitted=False):

        super(SimpleClassifierAggregator, self).__init__(
            base_estimators=base_estimators, pre_fitted=pre_fitted)

        # validate input parameters
        if method not in ['average', 'maximization', 'majority_vote',
                          'median']:
            raise ValueError("{method} is not a valid parameter.".format(
                method=method))

        self.method = method
        check_parameter(threshold, 0, 1, include_left=False,
                        include_right=False, param_name='threshold')
        self.threshold = threshold

        # set estimator weights
        self._set_weights(weights)

    def fit(self, X, y):
        """Fit classifier.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """

        # Validate inputs X and y
        X, y = check_X_y(X, y)
        X = check_array(X)
        self._set_n_classes(y)

        if self.pre_fitted:
            print("Training skipped")
            return
        else:
            for clf in self.base_estimators:
                clf.fit(X, y)
                clf.fitted_ = True
            return

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Class labels for each data sample.
        """
        X = check_array(X)

        all_scores = np.zeros([X.shape[0], self.n_base_estimators_])

        for i, clf in enumerate(self.base_estimators):
            if clf.fitted_ is not True and self.pre_fitted == False:
                ValueError('Classifier should be fitted first!')
            else:
                if hasattr(clf, 'predict'):
                    all_scores[:, i] = clf.predict(X)
                else:
                    raise ValueError(
                        "{clf} does not have predict.".format(clf=clf))

        if self.method == 'average':
            agg_score = average(all_scores, estimator_weights=self.weights)
        if self.method == 'maximization':
            agg_score = maximization(all_scores)
        if self.method == 'majority_vote':
            agg_score = majority_vote(all_scores, weights=self.weights)
        if self.method == 'median':
            agg_score = median(all_scores)

        return (agg_score >= self.threshold).astype('int').ravel()

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : numpy array of shape (n_samples,)
            The class probabilities of the input samples.
            Classes are ordered by lexicographic order.
        """
        X = check_array(X)
        all_scores = np.zeros(
            [X.shape[0], self._classes, self.n_base_estimators_])

        for i in range(self.n_base_estimators_):
            clf = self.base_estimators[i]
            if clf.fitted_ is not True and self.pre_fitted == False:
                ValueError('Classifier should be fitted first!')
            else:
                if hasattr(clf, 'predict_proba'):
                    all_scores[:, :, i] = clf.predict_proba(X)
                else:
                    raise ValueError(
                        "{clf} does not have predict_proba.".format(clf=clf))

        if self.method == 'average':
            return np.mean(all_scores * self.weights, axis=2)
        if self.method == 'maximization':
            scores = np.max(all_scores * self.weights, axis=2)
            return score_to_proba(scores)
        if self.method == 'majority_vote':
            Warning('average method is invoked for predict_proba as'
                    'probability is not continuous')
            return np.mean(all_scores * self.weights, axis=2)
        if self.method == 'median':
            Warning('average method is invoked for predict_proba as'
                    'probability is not continuous')
            return np.mean(all_scores * self.weights, axis=2)

    def fit_predict(self, X, y):
        """Fit estimator and predict on X

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Class labels for each data sample.
        """
        raise NotImplementedError(
            'fit_predict should not be used in supervised learning models.')
