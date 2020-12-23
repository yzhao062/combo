# -*- coding: utf-8 -*-
import os
import sys
from os import path

import unittest
# noinspection PyProtectedMember

from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y
from scipy.io import loadmat

from sklearn.metrics import roc_auc_score

from pyod.utils.data import generate_data
from pyod.models.lof import LOF

# temporary solution for relative imports in case pyod is not installed
# if combo is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from combo.models.detector_lscp import LSCP


class TestLSCP(unittest.TestCase):
    def setUp(self):
        # Define data file and read X and y
        # Generate some data if the source data is missing
        this_directory = path.abspath(path.dirname(__file__))
        mat_file = 'cardio.mat'
        try:
            mat = loadmat(path.join(*[this_directory, 'data', mat_file]))

        except TypeError:
            print('{data_file} does not exist. Use generated data'.format(
                data_file=mat_file))
            X, y = generate_data(train_only=True)  # load data
        except IOError:
            print('{data_file} does not exist. Use generated data'.format(
                data_file=mat_file))
            X, y = generate_data(train_only=True)  # load data
        else:
            X = mat['X']
            y = mat['y'].ravel()
            X, y = check_X_y(X, y)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.4, random_state=42)

        detectors = [LOF(), LOF()]

        self.clf = LSCP(base_estimators=detectors)
        self.clf.fit(self.X_train)
        self.roc_floor = 0.6

    def test_parameters(self):
        assert (hasattr(self.clf, 'decision_scores_') and
                self.clf.decision_scores_ is not None)
        assert (hasattr(self.clf, 'labels_') and
                self.clf.labels_ is not None)
        assert (hasattr(self.clf, 'threshold_') and
                self.clf.threshold_ is not None)
        assert (hasattr(self.clf, '_mu') and
                self.clf._mu is not None)
        assert (hasattr(self.clf, '_sigma') and
                self.clf._sigma is not None)

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

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
