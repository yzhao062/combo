# -*- coding: utf-8 -*-

import os
import sys

import unittest

import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.datasets import load_breast_cancer
# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_equal


# temporary solution for relative imports in case combo is not installed
# if  combo is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from combo.models.cluster_eac import EAC
from combo.models.cluster_eac import _generate_similarity_mat


class TestEAC(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)

        n_clusters = 5

        # Initialize a set of estimators
        estimators = [KMeans(n_clusters=n_clusters),
                      MiniBatchKMeans(n_clusters=n_clusters),
                      AgglomerativeClustering(n_clusters=n_clusters)]

        self.estimator = EAC(estimators, n_clusters=n_clusters)
        self.estimator.fit(self.X)

    def test_similarity_max(self):
        labels = np.array([[1, 2, 3, 1]])
        expected_mat = np.array([[1, 0, 0, 1],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [1, 0, 0, 1]])
        sim_mat = _generate_similarity_mat(labels)
        assert_allclose(expected_mat, sim_mat)

    def test_weights(self):
        assert_equal(np.sum(self.estimator.weights),
                     self.estimator.n_base_estimators_)

    def test_parameters(self):
        assert(hasattr(self.estimator, 'base_estimators') and
                    self.estimator.base_estimators is not None)

    def test_scores(self):
        predicted_labels = self.estimator.labels_
        assert_equal(predicted_labels.shape[0], self.X.shape[0])

    def test_fit_predict(self):
        predicted_labels = self.estimator.fit_predict(self.X)
        assert_equal(predicted_labels.shape[0], self.X.shape[0])


    def tearDown(self):
        pass
