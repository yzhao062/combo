# -*- coding: utf-8 -*-
"""Dynamic Classifier Selection (DES) is an established combination framework
for classification tasks.
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
from pyod.utils.utility import argmaxn

from .base import BaseAggregator
from ..utils.utility import score_to_proba
from .classifier_comb import average
from .classifier_comb import majority_vote


class DES_LA(BaseAggregator):
    """Dynamic Ensemble Selection (DES) is an established combination
    framework for classification tasks. The technique was based on
    Dynamic Classifier Selection (DCS) proposed by Ho et al. in
    1994 :cite:`ho1994decision`. The motivation behind this approach is that
    base classifiers often make distinctive errors and over a degree of
    complementarity. Consequently, selectively combining base classifier can
    result in a performance improvement over generic ensembles which use the
    majority vote of all base classifiers.

    Compared with DCS, DES uses a group of best classifiers to conduct a
    second phase combination, other than only the best classifier. The
    implemented version in this class is DES_LA which uses local accuracy
    as the metric for evaluating base classifier performance. `predict`
    uses (weighted) majority vote and `predict_proba` uses (weighted) average.

    See :cite:`ko2008dynamic` for details.

    Parameters
    ----------
    base_estimators: list or numpy array (n_estimators,)
        A list of base classifiers.

    local_region_size : int, optional (default=30)
        Number of training points to consider in each iteration of the local
        region generation process (30 by default).

    n_selected_clfs : int, optional (default=None)
        Number of selected base classifiers in the second phase combination.
        If None, set it to 1/2 * n_base_estimators

    use_weights : bool, optional (default=False)
        If True, use the classifiers' performance on the local region as
        their weight.

    threshold : float in (0, 1), optional (default=None)
        Cut-off value to convert scores into binary labels.

    pre_fitted : bool, optional (default=False)
        Whether the base classifiers are trained. If True, `fit`
        process may be skipped.

    """

    def __init__(self, base_estimators, local_region_size=30,
                 n_selected_clfs=None, use_weights=False, threshold=None,
                 pre_fitted=None):

        super(DES_LA, self).__init__(
            base_estimators=base_estimators, pre_fitted=pre_fitted)

        # validate input parameters
        if not isinstance(local_region_size, int):
            raise ValueError('local_region_size must be an integer variable')
        check_parameter(local_region_size, low=2, include_left=True,
                        param_name='local_region_size')
        self.local_region_size = local_region_size

        if n_selected_clfs is None:
            self.n_selected_clfs = int(self.n_base_estimators_ * 0.5)
        else:
            if not isinstance(n_selected_clfs, int):
                raise ValueError('n_selected_clfs must be an integer variable')
            check_parameter(n_selected_clfs, low=1,
                            high=self.n_base_estimators_, include_left=True,
                            include_right=True, param_name='n_selected_clfs')
            self.n_selected_clfs = n_selected_clfs

        self.use_weights = use_weights

        if threshold is not None:
            warnings.warn(
                "DES does not support threshold setting option. "
                "Please set the threshold in classifiers directly.")

        if pre_fitted is not None:
            warnings.warn("DES does not support pre_fitted option.")

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

            # print(clf_performance)

            # get the indices of the best performing clfs
            select_clf_inds = argmaxn(clf_performance, n=self.n_selected_clfs)
            select_clf_weights = clf_performance[select_clf_inds]. \
                reshape(1, len(select_clf_inds))

            # print(select_clf_inds)

            all_scores = np.zeros([1, len(select_clf_inds)])
            all_proba = np.zeros([1, self._classes, len(select_clf_inds)])

            for k, clf_ind in enumerate(select_clf_inds):

                clf = self.base_estimators[clf_ind]
                # make prediction
                if predict_proba:
                    all_proba[:, :, k] = clf.predict_proba(test_sample)
                else:
                    all_scores[:, k] = clf.predict(test_sample)

                # print('score', len(select_clf_inds), all_scores)

            if predict_proba:
                if self.use_weights:
                    y_predicted[i] = np.mean(all_proba * select_clf_weights,
                                             axis=2)
                else:
                    y_predicted[i] = np.mean(all_proba, axis=2)

            else:
                if self.use_weights:
                    y_predicted[i] = majority_vote(all_scores,
                                                   n_classes=self._classes,
                                                   weights=select_clf_weights)
                else:
                    y_predicted[i] = majority_vote(all_scores,
                                                   n_classes=self._classes)
        if predict_proba:
            return score_to_proba(y_predicted)
        else:
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
