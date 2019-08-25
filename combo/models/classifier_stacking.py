# -*- coding: utf-8 -*-
"""Stacking (meta ensembling). See http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
for more information.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import warnings
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from pyod.utils.utility import check_parameter

from ..utils.utility import list_diff
from .base import BaseAggregator


def split_datasets(X, y, n_folds=3, shuffle_data=False, random_state=None):
    """Utility function to split the data for stacking. The data is split
    into n_folds with roughly equal rough size.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The input samples.

    y : numpy array of shape (n_samples,)
        The ground truth of the input samples (labels).

    n_folds : int, optional (default=3)
        The number of splits of the training sample.

    shuffle_data : bool, optional (default=False)
        If True, shuffle the input data.

    random_state : RandomState, optional (default=None)
        A random number generator instance to define the state of the random
        permutations generator.

    Returns
    -------
    X : numpy array of shape (n_samples, n_features)
        The input samples. If shuffle_data, return the shuffled data.

    y : numpy array of shape (n_samples,)
        The ground truth of the input samples (labels). If shuffle_data,
        return the shuffled data.

    index_lists : list of list
        The list of indexes of each fold regarding the returned X and y.
        For instance, index_lists[0] contains the indexes of fold 0.

    """

    if not isinstance(n_folds, int):
        raise ValueError('n_folds must be an integer variable')
    check_parameter(n_folds, low=2, include_left=True, param_name='n_folds')

    random_state = check_random_state(random_state)

    if shuffle_data:
        X, y = shuffle(X, y, random_state=random_state)

    idx_length = len(y)
    idx_list = list(range(idx_length))

    avg_length = int(idx_length / n_folds)

    index_lists = []
    for i in range(n_folds - 1):
        index_lists.append(idx_list[i * avg_length:(i + 1) * avg_length])

    index_lists.append(idx_list[(n_folds - 1) * avg_length:])

    return X, y, index_lists


class Stacking(BaseAggregator):
    """Meta ensembling, also known as stacking. See
    http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
    for more information

    Parameters
    ----------
    base_estimators: list or numpy array (n_estimators,)
        A list of base classifiers.

    meta_clf : object, optional (default=LogisticRegression)
        The meta classifier to make the final prediction.

    n_folds : int, optional (default=2)
        The number of splits of the training sample.

    keep_original : bool, optional (default=False)
        If True, keep the original features for training and predicting.

    use_proba : bool, optional (default=False)
        If True, use the probability prediction as the new features.

    shuffle_data : bool, optional (default=False)
        If True, shuffle the input data.

    random_state : int, RandomState or None, optional (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    threshold : float in (0, 1), optional (default=None)
        Cut-off value to convert scores into binary labels.

    pre_fitted : bool, optional (default=False)
        Whether the base classifiers are trained. If True, `fit`
        process may be skipped.

    """

    def __init__(self, base_estimators, meta_clf=None, n_folds=2,
                 keep_original=True,
                 use_proba=False, shuffle_data=False, random_state=None,
                 threshold=None, pre_fitted=None):

        super(Stacking, self).__init__(
            base_estimators=base_estimators, pre_fitted=pre_fitted)

        # validate input parameters
        if not isinstance(n_folds, int):
            raise ValueError('n_folds must be an integer variable')
        check_parameter(n_folds, low=2, include_left=True,
                        param_name='n_folds')
        self.n_folds = n_folds

        if meta_clf is not None:
            self.meta_clf = meta_clf
        else:
            self.meta_clf = LogisticRegression()

        # set flags
        self.keep_original = keep_original
        self.use_proba = use_proba
        self.shuffle_data = shuffle_data

        self.random_state = random_state

        if threshold is not None:
            warnings.warn(
                "Stacking does not support threshold setting option. "
                "Please set the threshold in classifiers directly.")

        if pre_fitted is not None:
            warnings.warn("Stacking does not support pre_fitted option.")

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

        n_samples = X.shape[0]

        # initialize matrix for storing newly generated features
        new_features = np.zeros([n_samples, self.n_base_estimators_])

        # build CV datasets
        X_new, y_new, index_lists = split_datasets(
            X, y, n_folds=self.n_folds, shuffle_data=self.shuffle_data,
            random_state=self.random_state)

        # iterate over all base classifiers
        for i, clf in enumerate(self.base_estimators):
            # iterate over all folds
            for j in range(self.n_folds):
                # build train and test index
                full_idx = list(range(n_samples))
                test_idx = index_lists[j]
                train_idx = list_diff(full_idx, test_idx)
                X_train, y_train = X_new[train_idx, :], y_new[train_idx]
                X_test, y_test = X_new[test_idx, :], y_new[test_idx]

                # train the classifier
                clf.fit(X_train, y_train)

                # generate the new features on the pseudo test set
                if self.use_proba:
                    new_features[test_idx, i] = clf.predict_proba(
                        X_test)[:, 1]
                else:
                    new_features[test_idx, i] = clf.predict(X_test)

        # build the new dataset for training
        if self.keep_original:
            X_new_comb = np.concatenate([X_new, new_features], axis=1)
        else:
            X_new_comb = new_features
        y_new_comb = y_new

        # train the meta classifier
        self.meta_clf.fit(X_new_comb, y_new_comb)
        self.fitted_ = True

        # train all base classifiers on the full train dataset
        # iterate over all base classifiers
        for i, clf in enumerate(self.base_estimators):
            clf.fit(X_new, y_new)

        return

    def _process_data(self, X):
        """Internal class for both `predict` and `predict_proba`

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_new_comb : Numpy array
            The processed dataset of X.
        """
        check_is_fitted(self, ['fitted_'])
        X = check_array(X)
        n_samples = X.shape[0]

        # initialize matrix for storing newly generated features
        new_features = np.zeros([n_samples, self.n_base_estimators_])

        # build the new features for unknown samples
        # iterate over all base classifiers
        for i, clf in enumerate(self.base_estimators):
            # generate the new features on the test set
            if self.use_proba:
                new_features[:, i] = clf.predict_proba(X)[:, 1]
            else:
                new_features[:, i] = clf.predict(X)

        # build the new dataset for unknown samples
        if self.keep_original:
            X_new_comb = np.concatenate([X, new_features], axis=1)
        else:
            X_new_comb = new_features

        return X_new_comb

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
        X_new_comb = self._process_data(X)
        return self.meta_clf.predict(X_new_comb)

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
        X_new_comb = self._process_data(X)
        return self.meta_clf.predict_proba(X_new_comb)

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
