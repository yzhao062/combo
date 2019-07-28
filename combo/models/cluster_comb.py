# -*- coding: utf-8 -*-
"""A collection of combination methods for clustering
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import warnings
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
from sklearn.utils.testing import assert_equal

from .score_comb import majority_vote
from .sklearn_base import _pprint
from ..utils.utility import check_parameter
from ..utils.utility import _sklearn_version_21

if _sklearn_version_21():
    from inspect import signature
else:
    from sklearn.externals.funcsigs import signature

OFFSET_FACTOR = 1000000


class BaseClusteringAggregator(ABC):
    """Abstract class for all clustering aggregation methods.

    Parameters
    ----------
    estimators: list of shape (n_estimators,)
        A list of base estimators.

    pre_fitted: bool, optional (default=False)
        Whether the base estimators are fitted. If True, `fit`
        process may be skipped.
    """

    @abstractmethod
    def __init__(self, estimators, pre_fitted=False):
        assert (isinstance(estimators, (list)))
        if len(estimators) < 2:
            raise ValueError('At least 2 estimators are required')
        self.estimators = estimators
        self.n_estimators_ = len(self.estimators)
        self.pre_fitted = pre_fitted

    @abstractmethod
    def fit(self, X):
        """Fit detector.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        """
        pass

    @abstractmethod
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
        pass

    def __len__(self):
        """Returns the number of estimators in the ensemble."""
        return len(self.estimators)

    def __getitem__(self, index):
        """Returns the index'th estimator in the ensemble."""
        return self.estimators[index]

    def __iter__(self):
        """Returns iterator over estimators in the ensemble."""
        return iter(self.estimators)

    # noinspection PyMethodParameters
    def _get_param_names(cls):
        # noinspection PyPep8
        """Get parameter names for the estimator

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    # noinspection PyPep8
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        # noinspection PyPep8
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        Returns
        -------
        self : object
        """

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __repr__(self):
        # noinspection PyPep8
        """
        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name), ),)


class ClustererEnsemble(BaseClusteringAggregator):
    """Clusterer Ensemble combines base cluster results by alignment.
    See :cite:`zhou2006clusterer` for details.

    Parameters
    ----------
    estimators : list or numpy array of shape (n_estimators,)
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

    def __init__(self, estimators, n_clusters, weights=None, reference_idx=0,
                 pre_fitted=False):

        super(ClustererEnsemble, self).__init__(
            estimators=estimators, pre_fitted=pre_fitted)

        check_parameter(n_clusters, low=2, param_name='n_clusters')
        self.n_clusters = n_clusters

        check_parameter(reference_idx, low=0, high=self.n_estimators_ - 1,
                        include_left=True, include_right=True)
        self.reference_idx = reference_idx

        if weights is None:
            self.weights = np.ones([1, self.n_estimators_])
        else:

            self.weights = column_or_1d(weights).reshape(1, len(weights))
            assert (self.weights.shape[1] == self.n_estimators_)

            # adjust probability by a factor for integrity
            adjust_factor = self.weights.shape[1] / np.sum(weights)
            self.weights = self.weights * adjust_factor

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
        original_labels = np.zeros([X.shape[0], self.n_estimators_])

        if self.pre_fitted:
            print("Training Skipped")

        else:
            for clf in self.estimators:
                clf.fit(X)
                clf.fitted_ = True

        for i, estimator in enumerate(self.estimators):
            check_is_fitted(estimator, ['labels_'])
            original_labels[:, i] = estimator.labels_
        self.oiginal_labels_ = original_labels

        # get the aligned result
        self.labels_, self.aligned_labels_ = clusterer_ensemble_scores(
            original_labels,
            self.n_estimators_,
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
        original_labels = np.zeros([X.shape[0], self.n_estimators_])

        for i, estimator in enumerate(self.estimators):
            check_is_fitted(estimator, ['labels_'])
            original_labels[:, i] = estimator.predict(X)

        # get the aligned result
        predicted_labels = clusterer_ensemble_scores(
            original_labels,
            self.n_estimators_,
            n_clusters=self.n_clusters,
            weights=self.weights,
            return_results=False,
            reference_idx=self.reference_idx)

        return predicted_labels


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
        return labels_by_vote, aligned_labels
    else:
        return labels_by_vote


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

        #  print('component 1 cluser', max_i, '==', 'component 2 cluser', max_j)
        result_mat_aligned[np.where(result_mat_aligned[:,
                                    second_idx] == max_j), second_idx] = max_i + offset
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
