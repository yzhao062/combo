# -*- coding: utf-8 -*-
"""Combining multiple clusterings using evidence accumulation (EAC).
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import warnings
import numpy as np

from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from pyod.utils.utility import check_parameter

from .base import BaseAggregator


def _generate_similarity_mat(labels):
    """Internal function to generate similarity matrix.

    Parameters
    ----------
    labels : numpy array of shape (n_samples, 1)

    Returns
    -------
    sim_mat : numpy array of shape (n_samples, n_samples)
        Similarity matrix. If label_i == label_j, sim_mat[i,j] = 1, else 0.

    """
    l_mat = np.repeat(labels, len(labels), axis=1)
    l_mat_t = l_mat.T

    sim_mat = np.equal(l_mat, l_mat_t).astype(int)
    return sim_mat


class EAC(BaseAggregator):
    """Combining multiple clusterings using evidence accumulation (EAC) first
    builds similarity matrix for each base clustering to model the similarity
    among the cluster assignment among each sample. After the similarity
    matrices are aggregated, a hierarchical clustering is built on it. See
    :cite:`fred2005combining` for details.

    Parameters
    ----------
    base_estimators : list or numpy array of shape (n_estimators,)
        A list of base estimators. Estimators must have a `labels_`
        attribute once fitted. Sklearn clustering estimators are recommended.

    n_clusters : int, optional (default=8)
        The number of clusters.

    linkage_method : str, optional (default='single')
        The linkage method to use (single, complete, average,
        weighted, median centroid, ward). See
        https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
        for more information.

    weights : numpy array of shape (n_estimators,)
        Estimator weights. May be used after the alignment.

    pre_fitted : bool, optional (default=False)
        Whether the base estimators are trained. If True, `fit`
        process may be skipped.

    Attributes
    ----------
    labels_ : int
        The predicted label of the fitted data.

    Z_ : numpy array
        The linkage matrix encoding the hierarchical clustering. This can be
        used to plot dendrogram using scipy.
    """

    def __init__(self, base_estimators, n_clusters, linkage_method='single',
                 weights=None, pre_fitted=False):

        super(EAC, self).__init__(
            base_estimators=base_estimators, pre_fitted=pre_fitted)

        check_parameter(n_clusters, low=2, param_name='n_clusters')
        self.n_clusters = n_clusters

        # set estimator weights
        self._set_weights(weights)

        self.linkage_method = linkage_method

    def fit(self, X):
        """Fit estimators.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        """

        # Validate inputs X
        X = check_array(X)
        n_samples = X.shape[0]

        # initialize similarity matrix
        sim_mat_all = np.zeros([n_samples, n_samples])

        if self.pre_fitted:
            print("Training Skipped")

        else:
            for clf in self.base_estimators:
                clf.fit(X)
                clf.fitted_ = True

        for i, estimator in enumerate(self.base_estimators):
            check_is_fitted(estimator, ['labels_'])

            # get the labels from each base estimator
            labels = estimator.labels_.reshape(n_samples, 1)

            # generate the similarity matrix for the current estimator
            sim_mat = _generate_similarity_mat(labels)

            # add to the main similarity mat
            sim_mat_all = sim_mat_all + sim_mat

        # get the average of the similarity mat
        sim_mat_avg = np.divide(sim_mat_all, self.n_base_estimators_)

        # flip the similarity. smaller value implies more similarity
        sim_mat_avg = np.abs(np.max(sim_mat_avg) - sim_mat_avg)

        # build clusters
        self.Z_ = linkage(sim_mat_avg, method=self.linkage_method)
        self.labels_ = fcluster(self.Z_, self.n_clusters, criterion='maxclust')

        # it may leads to different number of clusters as specified by the user
        if len(np.unique(self.labels_)) != self.n_clusters:
            warnings.warn(
                'EAC generates {n} clusters instead of {n_clusters}'.format(
                    n=len(np.unique(self.labels_)),
                    n_clusters=self.n_clusters))

        return self

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
