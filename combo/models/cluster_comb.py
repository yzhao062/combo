# -*- coding: utf-8 -*-
"""A collection of combination methods for clustering
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause


import numpy as np

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.testing import assert_equal

from pyod.utils.utility import check_parameter

from .base import BaseAggregator
from .score_comb import majority_vote

OFFSET_FACTOR = 1000000


class ClustererEnsemble(BaseAggregator):
    """Clusterer Ensemble combines multiple base clustering estimators by
    alignment. See :cite:`zhou2006clusterer` for details.

    Parameters
    ----------
    base_estimators : list or numpy array of shape (n_estimators,)
        A list of base estimators. Estimators must have a `labels_`
        attribute once fitted. Sklearn clustering estimators are recommended.

    n_clusters : int, optional (default=8)
        The number of clusters.

    weights : numpy array of shape (n_estimators,)
        Estimator weights. May be used after the alignment.

    reference_idx : int in range [0, n_estimators-1], optional (default=0)
        The ith base estimator used as the reference for label alignment.

    pre_fitted : bool, optional (default=False)
        Whether the base estimators are trained. If True, `fit`
        process may be skipped.

    Attributes
    ----------
    labels_ : int
        The predicted label of the fitted data.
    """

    def __init__(self, base_estimators, n_clusters, weights=None,
                 reference_idx=0,
                 pre_fitted=False):

        super(ClustererEnsemble, self).__init__(
            base_estimators=base_estimators, pre_fitted=pre_fitted)

        check_parameter(n_clusters, low=2, param_name='n_clusters')
        self.n_clusters = n_clusters

        check_parameter(reference_idx, low=0, high=self.n_base_estimators_ - 1,
                        include_left=True, include_right=True)
        self.reference_idx = reference_idx

        # set estimator weights
        self._set_weights(weights)

    def fit(self, X):
        """Fit estimators.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        """

        # Validate inputs X
        X = check_array(X)

        # initialize the score matrix to store the results
        original_labels = np.zeros([X.shape[0], self.n_base_estimators_])

        if self.pre_fitted:
            print("Training Skipped")

        else:
            for clf in self.base_estimators:
                clf.fit(X)
                clf.fitted_ = True

        for i, estimator in enumerate(self.base_estimators):
            check_is_fitted(estimator, ['labels_'])
            original_labels[:, i] = estimator.labels_
        self.original_labels_ = original_labels

        # get the aligned result
        self.labels_, self.aligned_labels_ = clusterer_ensemble_scores(
            original_labels,
            self.n_base_estimators_,
            n_clusters=self.n_clusters,
            weights=self.weights,
            return_results=True,
            reference_idx=self.reference_idx)

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
        # TODO: decide whether enable predict function for clustering
        raise NotImplemented("predict function is currently disabled for"
                             "clustering due to inconsistent behaviours.")

        # Validate inputs X
        X = check_array(X)

        # initialize the score matrix to store the results
        original_labels = np.zeros([X.shape[0], self.n_base_estimators_])

        for i, estimator in enumerate(self.base_estimators):
            check_is_fitted(estimator, ['labels_'])
            original_labels[:, i] = estimator.predict(X)

        # get the aligned result
        predicted_labels = clusterer_ensemble_scores(
            original_labels,
            self.n_base_estimators_,
            n_clusters=self.n_clusters,
            weights=self.weights,
            return_results=False,
            reference_idx=self.reference_idx)

        return predicted_labels

    def predict_proba(self, X):
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
        raise NotImplemented("predict_proba function is currently disabled for"
                             "clustering due to inconsistent behaviours.")

    def fit_predict(self, X, y=None):
        """Fit estimator and predict on X. y is optional for unsupervised
        methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Cluster labels for each data sample.
        """
        self.fit(X)
        return self.labels_


def clusterer_ensemble_scores(original_labels, n_estimators, n_clusters,
                              weights=None, return_results=False,
                              reference_idx=0):
    """Function to align the raw clustering results from base estimators.
    Different from ClustererEnsemble class, this function takes in the output
    from base estimators directly without training and prediction.

    Parameters
    ----------
    original_labels : numpy array of shape (n_samples, n_estimators)
        The raw output from base estimators

    n_estimators : int
        The number of base estimators.

    n_clusters : int, optional (default=8)
        The number of clusters.

    weights : numpy array of shape (1, n_estimators)
        Estimators weights.

    return_results : bool, optional (default=False)
        If True, also return the aligned label matrix.

    reference_idx : int in range [0, n_estimators-1], optional (default=0)
        The ith base estimator used as the reference for label alignment.

    Returns
    -------
    aligned_labels : numpy array of shape (n_samples, n_estimators)
        The aligned label results by using reference_idx estimator as the
        reference.

    """

    original_labels = _validate_cluster_number(original_labels, n_clusters)
    alignment_mat = np.zeros([n_clusters, n_estimators])
    aligned_labels = np.copy(original_labels)

    for i in range(n_estimators):
        inter_mat = _intersection_mat(original_labels, reference_idx, i,
                                      n_clusters)
        index_mapping = _alignment(inter_mat, n_clusters, i, aligned_labels,
                                   OFFSET_FACTOR)
        alignment_mat[:, i] = index_mapping[:, 1]

    aligned_labels = aligned_labels - OFFSET_FACTOR

    if weights is not None:
        assert_equal(original_labels.shape[1], weights.shape[1])

    # equal weights if not set
    else:
        weights = np.ones([1, n_estimators])

    labels_by_vote = majority_vote(aligned_labels, n_classes=n_clusters,
                                   weights=weights)
    if return_results:
        return labels_by_vote.astype(int), aligned_labels.astype(int)
    else:
        return labels_by_vote.astype(int)


def _intersection_mat(result_mat, first_idx, second_idx, n_clusters):
    """Calculate the number of overlappings of second_idx based on first_idx.
    alignment_mat[i,j] represents the number of labels == j in second_idx
    when labels == i in the first idx.

    In other words, we should do the alignment based on the max by first
    assigning the most

    Parameters
    ----------
    result_mat
    first_idx
    second_idx
    n_clusters

    Returns
    -------

    """
    alignment_mat = np.zeros([n_clusters, n_clusters])
    for i in range(n_clusters):
        for j in range(n_clusters):
            i_index = np.argwhere(result_mat[:, first_idx] == i)
            j_index = np.argwhere(result_mat[:, second_idx] == j)
            inter_ij = np.intersect1d(i_index, j_index)
            alignment_mat[i, j] = len(inter_ij)

    return alignment_mat


def _alignment(inter_mat, n_clusters, second_idx, result_mat_aligned,
               offset=OFFSET_FACTOR):
    index_mapping = np.zeros([n_clusters, 2])
    index_mapping[:, 0] = list(range(0, n_clusters))

    while np.sum(inter_mat) > (-1 * n_clusters * n_clusters):
        max_i, max_j = np.unravel_index(inter_mat.argmax(), inter_mat.shape)
        index_mapping[max_i, 1] = max_j
        inter_mat[max_i, :] = -1
        inter_mat[:, max_j] = -1

        # print('component 1 cluser', max_i, '==', 'component 2 cluser', max_j)
        result_mat_aligned[
            np.where(result_mat_aligned[:, second_idx] == max_j), second_idx] \
            = max_i + offset
    return index_mapping


def _validate_cluster_number(original_results, n_clusters):
    """validate all estimators form the same number of clusters as defined
    in n_clusters.

    Parameters
    ----------
    original_results :
    n_clusters

    Returns
    -------

    """
    original_results = check_array(original_results)

    for i in range(original_results.shape[1]):

        values, counts = np.unique(original_results[:, i], return_counts=True)
        if len(values) != n_clusters:
            print(len(values), len(counts))
            RuntimeError('cluster result does not equal to n_clusters')

    return original_results
