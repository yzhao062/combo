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
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

from sklearn.datasets import load_breast_cancer

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
    print('Decision Tree       |',
          np.round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
                   decimals=4))

    clf = LogisticRegression(random_state=random_state)
    clf.fit(X_train, y_train)
    print('Logistic Regression |',
          np.round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
                   decimals=4))

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print('K Neighbors         |',
          np.round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
                   decimals=4))

    clf = GradientBoostingClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    print('Gradient Boosting   |',
          np.round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
                   decimals=4))

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    print('Random Forest       |',
          np.round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
                   decimals=4))

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
    y_test_predicted = clf.predict_proba(X_test)
    print('Combination by avg  |', np.round(
        roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
        decimals=4))

    # combine by weighted averaging
    clf_weights = np.array([0.1, 0.4, 0.1, 0.2, 0.2])
    clf = SimpleClassifierAggregator(classifiers, method='average')
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict_proba(X_test)
    print('Combination by w_avg|', np.round(
        roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
        decimals=4))

    # combine by maximization
    clf = SimpleClassifierAggregator(classifiers, method='maximization')
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict_proba(X_test)
    print('Combination by max  |', np.round(
        roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
        decimals=4))
