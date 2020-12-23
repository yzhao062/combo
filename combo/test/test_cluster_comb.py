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
from numpy.testing import assert_equal


# temporary solution for relative imports in case combo is not installed
# if  combo is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from combo.models.cluster_comb import ClustererEnsemble
from combo.models.cluster_comb import clusterer_ensemble_scores


class TestClustererEnsemble(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)

        n_clusters = 5

        # Initialize a set of estimators
        estimators = [KMeans(n_clusters=n_clusters),
                      MiniBatchKMeans(n_clusters=n_clusters),
                      AgglomerativeClustering(n_clusters=n_clusters)]

        self.estimator = ClustererEnsemble(estimators, n_clusters=n_clusters)
        self.estimator.fit(self.X)

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
