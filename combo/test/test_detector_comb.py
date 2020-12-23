# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import unittest
# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises


from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

# temporary solution for relative imports in case pyod is not installed
# if combo is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from combo.models.detector_comb import SimpleDetectorAggregator


class TestAverage(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.y_train, self.X_test, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        detectors = [KNN(), LOF(), OCSVM()]

        self.clf = SimpleDetectorAggregator(base_estimators=detectors,
                                            method='average',
                                            contamination=self.contamination)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        assert(hasattr(self.clf, 'decision_scores_') and
                    self.clf.decision_scores_ is not None)
        assert(hasattr(self.clf, 'labels_') and
                    self.clf.labels_ is not None)
        assert(hasattr(self.clf, 'threshold_') and
                    self.clf.threshold_ is not None)
        assert(hasattr(self.clf, '_mu') and
                    self.clf._mu is not None)
        assert(hasattr(self.clf, '_sigma') and
                    self.clf._sigma is not None)

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert(roc_auc_score(self.y_test, pred_scores)>= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, proba_method='linear')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, proba_method='unify')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, proba_method='something')

    def tearDown(self):
        pass


class Maximization(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.y_train, self.X_test, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        detectors = [KNN(), LOF(), OCSVM()]

        self.clf = SimpleDetectorAggregator(base_estimators=detectors,
                                            method='maximization',
                                            contamination=self.contamination)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        assert(hasattr(self.clf, 'decision_scores_') and
                    self.clf.decision_scores_ is not None)
        assert(hasattr(self.clf, 'labels_') and
                    self.clf.labels_ is not None)
        assert(hasattr(self.clf, 'threshold_') and
                    self.clf.threshold_ is not None)
        assert(hasattr(self.clf, '_mu') and
                    self.clf._mu is not None)
        assert(hasattr(self.clf, '_sigma') and
                    self.clf._sigma is not None)

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert(roc_auc_score(self.y_test, pred_scores)>= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, proba_method='linear')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, proba_method='unify')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, proba_method='something')

    def tearDown(self):
        pass


class TestMedian(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.y_train, self.X_test, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        detectors = [KNN(), LOF(), OCSVM()]

        self.clf = SimpleDetectorAggregator(base_estimators=detectors,
                                            method='median',
                                            contamination=self.contamination)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        assert(hasattr(self.clf, 'decision_scores_') and
                    self.clf.decision_scores_ is not None)
        assert(hasattr(self.clf, 'labels_') and
                    self.clf.labels_ is not None)
        assert(hasattr(self.clf, 'threshold_') and
                    self.clf.threshold_ is not None)
        assert(hasattr(self.clf, '_mu') and
                    self.clf._mu is not None)
        assert(hasattr(self.clf, '_sigma') and
                    self.clf._sigma is not None)

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert(roc_auc_score(self.y_test, pred_scores)>= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, proba_method='linear')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, proba_method='unify')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, proba_method='something')

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
