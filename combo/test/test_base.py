# -*- coding: utf-8 -*-
import os
import sys

import unittest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import numpy as np

# temporary solution for relative imports in case combo is not installed
# if combo is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from combo.models.base import BaseAggregator


# Check sklearn\tests\test_base
# A few test classes
# noinspection PyMissingConstructor,PyPep8Naming
class MyEstimator(BaseAggregator):

    def __init__(self, l1=0, empty=None):
        self.l1 = l1
        self.empty = empty

    def fit(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


# noinspection PyMissingConstructor
class K(BaseAggregator):
    def __init__(self, c=None, d=None):
        self.c = c
        self.d = d

    def fit(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


# noinspection PyMissingConstructor
class T(BaseAggregator):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b

    def fit(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


# noinspection PyMissingConstructor
class ModifyInitParams(BaseAggregator):
    """Deprecated behavior.
    Equal parameters but with a type cast.
    Doesn't fulfill a is a
    """

    def __init__(self, a=np.array([0])):
        self.a = a.copy()

    def fit(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


# noinspection PyMissingConstructor
class VargEstimator(BaseAggregator):
    """scikit-learn estimators shouldn't have vargs."""

    def __init__(self, *vargs):
        pass

    def fit(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


class Dummy1(BaseAggregator):
    def __init__(self, base_estimators=[DecisionTreeClassifier(),
                                        LogisticRegression()]):
        super(Dummy1, self).__init__(base_estimators=base_estimators)

    def fit(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


class Dummy2(BaseAggregator):
    def __init__(self, base_estimators=[DecisionTreeClassifier(),
                                        LogisticRegression()]):
        super(Dummy2, self).__init__(base_estimators=base_estimators)

    def fit(self, X, y=None):
        return X

    def fit_predict(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


class Dummy3(BaseAggregator):
    def __init__(self, base_estimators=[DecisionTreeClassifier(),
                                        LogisticRegression()]):
        super(Dummy3, self).__init__(base_estimators=base_estimators)

    def decision_function(self, X):
        pass

    def fit(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        pass

    def predict(self, X):
        return X

    def predict_proba(self, X):
        pass


class Dummy4(BaseAggregator):
    def __init__(self, base_estimators=[DecisionTreeClassifier(),
                                        LogisticRegression()]):
        super(Dummy4, self).__init__(base_estimators=base_estimators)

    def decision_function(self, X):
        pass

    def fit(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        return X


class Dummy5(BaseAggregator):
    def __init__(self, base_estimators=[DecisionTreeClassifier(),
                                        LogisticRegression()]):
        super(Dummy5, self).__init__(base_estimators=base_estimators)

    def decision_function(self, X):
        pass

    def fit(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        return X

    def predict(self, X):
        pass

    def predict_proba(self, X):
        return X


class TestBASE(unittest.TestCase):
    def setUp(self):
        random_state = 42
        X, y = load_breast_cancer(return_X_y=True)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.4, random_state=random_state)

    def test_init(self):
        """
        Test base class initialization

        :return:
        """
        self.dummy_clf = Dummy1()
        self.dummy_clf = Dummy1(base_estimators=[DecisionTreeClassifier(),
                                                 DecisionTreeClassifier()])
        # assert_equal(self.dummy_clf.base_estimators,
        #              [DecisionTreeClassifier(), LogisticRegression()])
        #
        # self.dummy_clf = Dummy1(
        #     base_estimators=[LogisticRegression(), DecisionTreeClassifier()])
        # assert_equal(self.dummy_clf.base_estimators,
        #              [LogisticRegression(), DecisionTreeClassifier()])

        # with assert_raises(ValueError):
        #     Dummy1(base_estimators=[LogisticRegression()])
        #
        # with assert_raises(ValueError):
        #     Dummy1(base_estimators=0)
        #
        # with assert_raises(ValueError):
        #     Dummy1(base_estimators=-0.5)

    def test_fit(self):
        self.dummy_clf = Dummy2()
        assert_equal(self.dummy_clf.fit(0), 0)

    def test_fit_predict(self):
        self.dummy_clf = Dummy5()
        assert_equal(self.dummy_clf.fit_predict(0), 0)

    def test_predict(self):
        # TODO: add more testcases

        self.dummy_clf = Dummy3()
        assert_equal(self.dummy_clf.predict(0), 0)

    def test_predict_proba(self):
        # TODO: create uniform testcases
        self.dummy_clf = Dummy4()
        assert_equal(self.dummy_clf.predict_proba(0), 0)

    def test_rank(self):
        # TODO: create uniform testcases
        pass

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

        assert ('a__d' in test.get_params(deep=True))
        assert ('a__d' not in test.get_params(deep=False))

        test.set_params(a__d=2)
        assert (test.a.d == 2)
        assert_raises(ValueError, test.set_params, a__a=2)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
