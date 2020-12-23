# -*- coding: utf-8 -*-

import os
import sys

import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_breast_cancer
# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# temporary solution for relative imports in case combo is not installed
# if  combo is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from combo.models.classifier_dcs import DCS_LA
from combo.utils.data import evaluate_print


class TestDCS_LA(unittest.TestCase):
    def setUp(self):
        self.roc_floor = 0.9
        self.accuracy_floor = 0.9

        random_state = 42
        X, y = load_breast_cancer(return_X_y=True)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.4, random_state=random_state)

        classifiers = [DecisionTreeClassifier(random_state=random_state),
                       LogisticRegression(random_state=random_state),
                       KNeighborsClassifier(),
                       RandomForestClassifier(random_state=random_state),
                       GradientBoostingClassifier(random_state=random_state)]

        self.clf = DCS_LA(classifiers, local_region_size=30)
        self.clf.fit(self.X_train, self.y_train)

    def test_parameters(self):
        assert(hasattr(self.clf, 'base_estimators') and
                    self.clf.base_estimators is not None)

    def test_train_scores(self):
        y_train_predicted = self.clf.predict(self.X_train)
        assert_equal(len(y_train_predicted), self.X_train.shape[0])

        # check performance
        assert(accuracy_score(self.y_train, y_train_predicted)>=
                       self.accuracy_floor)

    def test_prediction_scores(self):
        y_test_predicted = self.clf.predict(self.X_test)
        assert_equal(len(y_test_predicted), self.X_test.shape[0])

        # check performance
        assert(accuracy_score(self.y_test, y_test_predicted)>=
                       self.accuracy_floor)

        # test utility function
        evaluate_print('averaging', self.y_test, y_test_predicted)

    def test_prediction_proba(self):
        y_test_predicted = self.clf.predict_proba(self.X_test)
        assert (y_test_predicted.min() >= 0)
        assert (y_test_predicted.max() <= 1)

        # check performance
        assert (roc_auc_score(self.y_test,
                              y_test_predicted[:, 1]) >= self.roc_floor)

        # check shape of integrity
        n_classes = len(np.unique(self.y_train))
        assert_equal(y_test_predicted.shape, (self.X_test.shape[0], n_classes))

        # check probability sum is 1
        y_test_predicted_sum = np.sum(y_test_predicted, axis=1)
        assert_allclose(np.ones([self.X_test.shape[0], ]),
                        y_test_predicted_sum)

    def test_fit_predict(self):
        with assert_raises(NotImplementedError):
            y_train_predicted = self.clf.fit_predict(self.X_train,
                                                     self.y_train)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
