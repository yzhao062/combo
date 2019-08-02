# -*- coding: utf-8 -*-
"""Example of Dynamic Classifier Selection by Local Accuracy
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause


import os
import sys

# temporary solution for relative imports in case combo is not installed
# if combo is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from scipy.io import loadmat

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from combo.models.classifier_dcs import DCS_LA
from combo.utils.data import evaluate_print

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # Define data file and read X and y
    # Generate some data if the source data is missing
    mat_file = 'letter.mat'
    try:
        mat = loadmat(os.path.join('data', mat_file))
    except TypeError:
        X, y = load_breast_cancer(return_X_y=True)  # load data
    except IOError:
        X, y = load_breast_cancer(return_X_y=True)  # load data
    else:
        X = mat['X']
        y = mat['y'].ravel()

    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=random_state)

    # initialize a group of clfs
    classifiers = [DecisionTreeClassifier(random_state=random_state),
                   LogisticRegression(random_state=random_state),
                   KNeighborsClassifier(),
                   RandomForestClassifier(random_state=random_state),
                   GradientBoostingClassifier(random_state=random_state)]

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
    clf = DCS_LA(classifiers)
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    y_test_proba_predicted = clf.predict_proba(X_test)
    evaluate_print('DCS_LA               |', y_test, y_test_predicted)
