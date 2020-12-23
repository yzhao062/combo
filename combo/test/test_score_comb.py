# -*- coding: utf-8 -*-

import os
import sys

import unittest
# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import numpy as np
from sklearn.utils import shuffle

# temporary solution for relative imports in case combo is not installed
# if combo is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from combo.models.score_comb import aom
from combo.models.score_comb import moa
from combo.models.score_comb import average
from combo.models.score_comb import maximization
from combo.models.score_comb import median
from combo.models.score_comb import majority_vote


class TestAOM(unittest.TestCase):
    def setUp(self):
        self.scores = np.asarray([[0.5, 0.8, 0.6, 0.9, 0.7, 0.6],
                                  [0.8, 0.75, 0.25, 0.6, 0.45, 0.8],
                                  [0.8, 0.3, 0.28, 0.99, 0.28, 0.3],
                                  [0.74, 0.85, 0.38, 0.47, 0.27, 0.69]])

    def test_aom_static_norepeat(self):
        score = aom(self.scores, 3, method='static',
                    bootstrap_estimators=False,
                    random_state=42)

        assert_equal(score.shape, (4,))

        shuffled_list = shuffle(list(range(0, 6, 1)), random_state=42)
        manual_scores = np.zeros([4, 3])
        manual_scores[:, 0] = np.max(self.scores[:, shuffled_list[0:2]],
                                     axis=1)
        manual_scores[:, 1] = np.max(self.scores[:, shuffled_list[2:4]],
                                     axis=1)
        manual_scores[:, 2] = np.max(self.scores[:, shuffled_list[4:6]],
                                     axis=1)

        manual_score = np.mean(manual_scores, axis=1)
        assert_array_equal(score, manual_score)

    def test_aom_static_repeat(self):
        score = aom(self.scores, 3, method='static', bootstrap_estimators=True,
                    random_state=42)
        assert_equal(score.shape, (4,))

    def test_aom_static_n_buckets(self):
        with assert_raises(ValueError):
            aom(self.scores, 5, method='static', bootstrap_estimators=False,
                random_state=42)

        # TODO: add more complicated testcases

    def test_aom_dynamic_repeat(self):
        score = aom(self.scores, 3, method='dynamic',
                    bootstrap_estimators=True,
                    random_state=42)
        assert_equal(score.shape, (4,))

        # TODO: add more complicated testcases

    def tearDown(self):
        pass


class TestMOA(unittest.TestCase):
    def setUp(self):
        self.scores = np.asarray([[0.5, 0.8, 0.6, 0.9, 0.7, 0.6],
                                  [0.8, 0.75, 0.25, 0.6, 0.45, 0.8],
                                  [0.8, 0.3, 0.28, 0.99, 0.28, 0.3],
                                  [0.74, 0.85, 0.38, 0.47, 0.27, 0.69]])

    def test_moa_static_norepeat(self):
        score = moa(self.scores, 3, method='static',
                    bootstrap_estimators=False, random_state=42)

        assert_equal(score.shape, (4,))

        shuffled_list = shuffle(list(range(0, 6, 1)), random_state=42)
        manual_scores = np.zeros([4, 3])
        manual_scores[:, 0] = np.mean(self.scores[:, shuffled_list[0:2]],
                                      axis=1)
        manual_scores[:, 1] = np.mean(self.scores[:, shuffled_list[2:4]],
                                      axis=1)
        manual_scores[:, 2] = np.mean(self.scores[:, shuffled_list[4:6]],
                                      axis=1)

        manual_score = np.max(manual_scores, axis=1)
        assert_array_equal(score, manual_score)

    def test_moa_static_repeat(self):
        score = moa(self.scores, 3, method='static', bootstrap_estimators=True,
                    random_state=42)
        assert_equal(score.shape, (4,))

    def test_moa_static_n_buckets(self):
        with assert_raises(ValueError):
            moa(self.scores, 5, method='static', bootstrap_estimators=False,
                random_state=42)

        # TODO: add more complicated testcases

    def test_moa_dynamic_repeat(self):
        score = moa(self.scores, 3, method='dynamic',
                    bootstrap_estimators=True, random_state=42)
        assert_equal(score.shape, (4,))

        # TODO: add more complicated testcases

    def tearDown(self):
        pass


class TestStatic(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([[1, 2], [3, 4], [5, 6]])
        self.weights = np.array([[0.2, 0.6]])

    def test_average(self):
        score = average(self.scores)
        assert_allclose(score, np.array([1.5, 3.5, 5.5]))

    def test_weighted_average(self):
        score = average(self.scores, self.weights)
        assert_allclose(score, np.array([1.75, 3.75, 5.75]))

    def test_maximization(self):
        score = maximization(self.scores)
        assert_allclose(score, np.array([2, 4, 6]))

    def test_median(self):
        score = median(np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7]]))
        assert_allclose(score, np.array([1, 3, 6]))


class TestMajorityVote(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([[0, 1, 1], [0, 1, 2], [2, 2, 2], [1, 1, 2]])
        self.weights = np.array([[0.1, 0.8, 0.1]])

    def test_majority_vote(self):
        score = majority_vote(self.scores, n_classes=3)
        assert_allclose(score, np.array([1, 0, 2, 1]))

    def test_weighted_majority_vote(self):
        score = majority_vote(self.scores, n_classes=3, weights=self.weights)
        assert_allclose(score, np.array([1, 1, 2, 1]))


if __name__ == '__main__':
    unittest.main()
