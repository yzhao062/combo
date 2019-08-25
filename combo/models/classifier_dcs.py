# -*- coding: utf-8 -*-
"""Stacking (meta ensembling). See http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
for more information.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import warnings
import numpy as np

from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from pyod.utils.utility import check_parameter

from .base import BaseAggregator


class DCS_LA(BaseAggregator):
    """Dynamic Classifier Selection (DCS) is an established combination
    framework for classification tasks. The technique was first proposed by Ho
    et al. in 1994 :cite:`ho1994decision` and then extended, under the name
    DCS Local Accuracy, by Woods et al. in 1997 :cite:`woods1997combination`
    to select the most accurate base classifier in a local region.
    The motivation behind this approach is that base classifiers often make
    distinctive errors and over a degree of complementarity. Consequently,
    selectively combining base classifier can result in a performance
    improvement over generic ensembles which use the majority vote of all
    base classifiers.

    See :cite:`woods1997combination` for details.

    Parameters
    ----------
    base_estimators: list or numpy array (n_estimators,)
        A list of base classifiers.

    local_region_size : int, optional (default=30)
        Number of training points to consider in each iteration of the local
        region generation process (30 by default).

    threshold : float in (0, 1), optional (default=None)
        Cut-off value to convert scores into binary labels.

    pre_fitted : bool, optional (default=False)
        Whether the base classifiers are trained. If True, `fit`
        process may be skipped.

    """

    def __init__(self, base_estimators, local_region_size=30, threshold=None,
                 pre_fitted=None):

        super(DCS_LA, self).__init__(
            base_estimators=base_estimators, pre_fitted=pre_fitted)

        # validate input parameters
        if not isinstance(local_region_size, int):
            raise ValueError('local_region_size must be an integer variable')
        check_parameter(local_region_size, low=2, include_left=True,
                        param_name='local_region_size')
        self.local_region_size = local_region_size

        if threshold is not None:
            warnings.warn(
                "DCS does not support threshold setting option. "
                "Please set the threshold in classifiers directly.")

        if pre_fitted is not None:
            warnings.warn("DCS does not support pre_fitted option.")

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
        check_classification_targets(y)
        self._classes = len(np.unique(y))
        n_samples = X.shape[0]

        # save the train ground truth for evaluation purpose
        self.y_train_ = y

        # build KDTree out of training subspace
        self.tree_ = KDTree(X)

        self.y_train_predicted_ = np.zeros(
            [n_samples, self.n_base_estimators_])

        # train all base classifiers on X, and get their local predicted scores
        # iterate over all base classifiers
        for i, clf in enumerate(self.base_estimators):
            clf.fit(X, y)
            self.y_train_predicted_[:, i] = clf.predict(X)
            clf.fitted_ = True

        self.fitted_ = True

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
        return self._predict_internal(X, predict_proba=False)

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
        return self._predict_internal(X, predict_proba=True)

    def _predict_internal(self, X, predict_proba):
        """Internal function for predict and predict_proba

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        predict_proba : bool
            if True, return the result of predict_proba

        Returns
        -------
        """
        check_is_fitted(self, ['fitted_'])
        X = check_array(X)
        n_samples = X.shape[0]

        # Find neighbors for all test instances
        _, ind_arr = self.tree_.query(X, k=self.local_region_size)

        if predict_proba:
            y_predicted = np.zeros([n_samples, self._classes])
        else:
            y_predicted = np.zeros([n_samples, ])

        # For each test sample
        for i in range(n_samples):
            test_sample = X[i, :].reshape(1, -1)
            train_inds = ind_arr[i, :]

            # ground truth
            y_train_sample = self.y_train_[train_inds]

            clf_performance = np.zeros([self.n_base_estimators_, ])

            for j, clf in enumerate(self.base_estimators):
                y_train_clf = self.y_train_predicted_[train_inds, j]
                clf_performance[j] = accuracy_score(y_train_sample,
                                                    y_train_clf)

            # select the best clf. may get multiple results
            select_clf_inds = np.argwhere(
                clf_performance == np.amax(clf_performance)).ravel()

            # select the first element from all candidates
            best_clf_ind = select_clf_inds[-1]

            # make prediction
            if predict_proba:
                y_predicted[i] = self.base_estimators[
                    best_clf_ind].predict_proba(test_sample)
            else:
                y_predicted[i] = self.base_estimators[best_clf_ind].predict(
                    test_sample)

        return y_predicted

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
