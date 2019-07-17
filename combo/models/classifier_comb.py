# -*- coding: utf-8 -*-
"""A collection of methods for combining classifiers
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import warnings
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np

from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import check_classification_targets

from .score_comb import average, maximization, majority_vote
from .sklearn_base import _sklearn_version_21
from .sklearn_base import _pprint
from ..utils.utility import check_parameter

if _sklearn_version_21():
    from inspect import signature
else:
    from sklearn.externals.funcsigs import signature


class BaseClassifierAggregator(ABC):
    """Abstract class for all classifier aggregation methods.

    Parameters
    ----------
    classifiers: list or numpy array (n_estimators,)
        A list of base classifiers.

    pre_fitted: bool, optional (default=False)
        Whether the base classifiers are trained. If True, `fit`
        process may be skipped.
    """

    @abstractmethod
    def __init__(self, classifiers, pre_fitted=False):
        assert (isinstance(classifiers, (list)))
        if len(classifiers) < 2:
            raise ValueError('At least 2 classifiers are required')
        self.classifiers = classifiers
        self.len_classifiers_ = len(self.classifiers)
        self.pre_fitted = pre_fitted

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

    @abstractmethod
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
        pass

    def __len__(self):
        """Returns the number of estimators in the ensemble."""
        return len(self.classifiers)

    def __getitem__(self, index):
        """Returns the index'th estimator in the ensemble."""
        return self.classifiers[index]

    def __iter__(self):
        """Returns iterator over estimators in the ensemble."""
        return iter(self.classifiers)

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


class SimpleClassifierAggregator(BaseClassifierAggregator):
    """A collection of simple combination methods.

    Parameters
    ----------
    classifiers: list or numpy array of shape (n_classifiers,)
        A list of base classifiers.

    method : str, optional (default='average')
        Combination method: {'average', 'maximization', 'majority vote'}.
        Pass in weights of classifiers for weighted version.

    threshold : float in (0, 1), optional (default=0.5)
        Cut-off value to convert scores into binary labels.

    weights : numpy array of shape (n_classifiers,)
        Classifier weights.

    pre_fitted: bool, optional (default=False)
        Whether the base classifiers are trained. If True, `fit`
        process may be skipped.
    """

    def __init__(self, classifiers, method='average', threshold=0.5,
                 weights=None, pre_fitted=False):

        super(SimpleClassifierAggregator, self).__init__(
            classifiers=classifiers, pre_fitted=pre_fitted)

        # validate input parameters
        if method not in ['average', 'maximization', 'majority_vote']:
            raise ValueError("{method} is not a valid parameter.".format(
                shape=method))

        self.method = method
        check_parameter(threshold, 0, 1, include_left=False,
                        include_right=False, param_name='threshold')
        self.threshold = threshold

        if weights is None:
            self.weights = np.ones([1, self.len_classifiers_])
        else:

            self.weights = column_or_1d(weights).reshape(1, len(weights))
            assert (self.weights.shape[1] == self.len_classifiers_)

            # adjust probability by a factor for integrity
            adjust_factor = self.weights.shape[1] / np.sum(weights)
            self.weights = self.weights * adjust_factor

    def fit(self, X, y):
        """Fit detector.

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

        if self.pre_fitted:
            print("Training skipped")
            return
        else:
            for clf in self.classifiers:
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

        all_scores = np.zeros([X.shape[0], self.len_classifiers_])

        for i in range(self.len_classifiers_):
            clf = self.classifiers[i]
            if clf.fitted_ != True and self.pre_fitted == False:
                ValueError('Classifier should be fitted first!')
            else:
                all_scores[:, i] = clf.predict(X)

        if self.method == 'average':
            agg_score = average(all_scores, estimator_weights=self.weights)
        if self.method == 'maximization':
            agg_score = maximization(all_scores)
        if self.method == 'majority_vote':
            agg_score = majority_vote(all_scores, weights=self.weights)

        return (agg_score >= self.threshold).astype('int').ravel()

    def predict_proba(self, X, method='average'):
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

        all_scores = np.zeros(
            [X.shape[0], self._classes, self.len_classifiers_])

        for i in range(self.len_classifiers_):
            clf = self.classifiers[i]
            if clf.fitted_ != True and self.pre_fitted == False:
                ValueError('Classifier should be fitted first!')
            else:
                all_scores[:, :, i] = clf.predict_proba(X)

        if self.method == 'average':
            return np.mean(all_scores * self.weights, axis=2)
        if self.method == 'maximization':
            return np.max(all_scores * self.weights, axis=2)
        if self.method == 'majority_vote':
            Warning('average method is invoked for predict_proba as'
                    'probability is not continuous')
            return np.mean(all_scores * self.weights, axis=2)
