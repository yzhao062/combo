# -*- coding: utf-8 -*-

import os
import sys

import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.datasets import load_breast_cancer
# noinspection PyProtectedMember
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true

# temporary solution for relative imports in case combo is not installed
# if  combo is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from combo.models.cluster_comb import BaseClusteringAggregator
from combo.models.cluster_comb import ClustererEnsemble
from combo.models.cluster_comb import clusterer_ensemble_scores


# Check sklearn\tests\test_base
# A few test classes
# noinspection PyMissingConstructor,PyPep8Naming
class MyEstimator(BaseClusteringAggregator):

    def __init__(self, l1=0, empty=None):
        self.l1 = l1
        self.empty = empty

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass


# noinspection PyMissingConstructor
class K(BaseClusteringAggregator):
    def __init__(self, c=None, d=None):
        self.c = c
        self.d = d

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass


# noinspection PyMissingConstructor
class T(BaseClusteringAggregator):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


# noinspection PyMissingConstructor
class ModifyInitParams(BaseClusteringAggregator):
    """Deprecated behavior.
    Equal parameters but with a type cast.
    Doesn't fulfill a is a
    """

    def __init__(self, a=np.array([0])):
        self.a = a.copy()

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass


# noinspection PyMissingConstructor
class VargEstimator(BaseClusteringAggregator):
    """scikit-learn estimators shouldn't have vargs."""

    def __init__(self, *vargs):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        pass


class TestBase(unittest.TestCase):
    def test_repr(self):
        # Smoke test the repr of the base estimator.
        my_estimator = MyEstimator()
        repr(my_estimator)
        test = T(K(), K())
        assert_equal(
            repr(test),
            "T(a=K(c=None, d=None), b=K(c=None, d=None))"
        )

        some_est = T(a=["long_params"] * 1000)
        assert_equal(len(repr(some_est)), 415)

    def test_str(self):
        # Smoke test the str of the base estimator
        my_estimator = MyEstimator()
        str(my_estimator)

    def test_get_params(self):
        test = T(K(), K())

        assert_true('a__d' in test.get_params(deep=True))
        assert_true('a__d' not in test.get_params(deep=False))

        test.set_params(a__d=2)
        assert_true(test.a.d == 2)
        assert_raises(ValueError, test.set_params, a__a=2)


class TestClustererEnsemble(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)

        n_clusters = 5
        n_estimators = 3

        # Initialize a set of estimators
        estimators = [KMeans(n_clusters=n_clusters),
                      MiniBatchKMeans(n_clusters=n_clusters),
                      AgglomerativeClustering(n_clusters=n_clusters)]

        self.estimator = ClustererEnsemble(estimators, n_clusters=n_clusters)
        self.estimator.fit(self.X)

    def test_parameters(self):
        assert_true(hasattr(self.estimator, 'estimators') and
                    self.estimator.estimators is not None)

    def test_scores(self):
        predicted_labels = self.estimator.labels_
        assert_equal(predicted_labels.shape[0], self.X.shape[0])

    def tearDown(self):
        pass


class TestClustererEnsembleScores(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)

        self.n_clusters = 5
        self.n_estimators = 3

        # Initialize a set of estimators
        estimators = [KMeans(n_clusters=self.n_clusters),
                      MiniBatchKMeans(n_clusters=self.n_clusters),
                      AgglomerativeClustering(n_clusters=self.n_clusters)]

        # Clusterer Ensemble without initializing a new Class
        self.original_labels = np.zeros([self.X.shape[0], self.n_estimators])

        for i, estimator in enumerate(estimators):
            estimator.fit(self.X)
            self.original_labels[:, i] = estimator.labels_

    def test_scores(self):
        labels_by_vote1 = clusterer_ensemble_scores(self.original_labels,
                                                    self.n_estimators,
                                                    self.n_clusters)
        assert_equal(labels_by_vote1.shape[0], self.X.shape[0])

        # return aligned_labels as well
        labels_by_vote2, aligned_labels = clusterer_ensemble_scores(
            self.original_labels, self.n_estimators, self.n_clusters,
            return_results=True)
        assert_equal(labels_by_vote2.shape[0], self.X.shape[0])
        assert_equal(aligned_labels.shape, self.original_labels.shape)

        # select a different reference base estimator (default is 0)
        labels_by_vote3 = clusterer_ensemble_scores(self.original_labels,
                                                    self.n_estimators,
                                                    self.n_clusters,
                                                    reference_idx=1)
        assert_equal(labels_by_vote3.shape[0], self.X.shape[0])

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
