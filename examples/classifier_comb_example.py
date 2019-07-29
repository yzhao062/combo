# -*- coding: utf-8 -*-
"""Example of combining multiple base classifiers. Two combination
frameworks are demonstrated:

1. Average: take the average of all base detectors
2. maximization : take the maximum score across all detectors as the score

"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause


import os
import sys

# temporary solution for relative imports in case combo is not installed
# if combo is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from combo.models.classifier_comb import SimpleClassifierAggregator
from combo.utils.data import evaluate_print

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Define data file and read X and y
    random_state = 42
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=random_state)

    # fit and predict by individual classifiers
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    evaluate_print('Decision Tree        |', y_test, clf.predict(X_test))

    clf = LogisticRegression(random_state=random_state)
    clf.fit(X_train, y_train)
    evaluate_print('Logistic Regression  |', y_test, clf.predict(X_test))

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    evaluate_print('K Neighbors          |', y_test, clf.predict(X_test))

    clf = GradientBoostingClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    evaluate_print('Gradient Boosting    |', y_test, clf.predict(X_test))

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    evaluate_print('Random Forest        |', y_test, clf.predict(X_test))

    print()

    # initialize a group of classifiers
    classifiers = [DecisionTreeClassifier(random_state=random_state),
                   LogisticRegression(random_state=random_state),
                   KNeighborsClassifier(),
                   RandomForestClassifier(random_state=random_state),
                   GradientBoostingClassifier(random_state=random_state)]

    # combine by averaging
    clf = SimpleClassifierAggregator(classifiers, method='average')
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    evaluate_print('Combination by avg   |', y_test, y_test_predicted)

    # combine by weighted averaging
    clf_weights = np.array([0.1, 0.4, 0.1, 0.2, 0.2])
    clf = SimpleClassifierAggregator(classifiers, method='average',
                                     weights=clf_weights)
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    evaluate_print('Combination by w_avg |', y_test, y_test_predicted)

    # combine by maximization
    clf = SimpleClassifierAggregator(classifiers, method='maximization')
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    evaluate_print('Combination by max   |', y_test, y_test_predicted)

    # combine by weighted majority vote
    clf_weights = np.array([0.1, 0.4, 0.1, 0.2, 0.2])
    clf = SimpleClassifierAggregator(classifiers, method='majority_vote',
                                     weights=clf_weights)
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    evaluate_print('Combination by w_vote|', y_test, y_test_predicted)

    # combine by median
    clf = SimpleClassifierAggregator(classifiers, method='median')
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    evaluate_print('Combination by median|', y_test, y_test_predicted)
