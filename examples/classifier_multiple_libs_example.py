# -*- coding: utf-8 -*-
"""Example of combining the models from different ML libraries. The example 
shows the combination of scikit-learn, xgboost, and LightGBM models.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause


import os
import sys

# temporary solution for relative imports in case combo is not installed
# if combo is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from combo.models.classifier_stacking import Stacking
from combo.utils.data import evaluate_print

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Define data file and read X and y
    random_state = 42
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state)

    # initialize a group of clfs from scikit-learn, xgboost, and LightGBM
    classifiers = [DecisionTreeClassifier(random_state=random_state),
                   LogisticRegression(random_state=random_state),
                   KNeighborsClassifier(),
                   RandomForestClassifier(random_state=random_state),
                   GradientBoostingClassifier(random_state=random_state),
                   LGBMClassifier(random_state=random_state),
                   XGBClassifier(random_state=random_state)]
    clf_names = ['DT', 'LR', 'KNN', 'RF', 'GBDT', 'LGB', 'XGB']

    # evaluate individual classifiers
    for i, clf in enumerate(classifiers):
        clf.fit(X_train, y_train)
        y_test_predict = clf.predict(X_test)
        evaluate_print(clf_names[i] + '   |   ', y_test, y_test_predict)

    print()
    # build a Stacking model and evaluate
    clf = Stacking(classifiers, n_folds=4, shuffle_data=False,
                   keep_original=True, use_proba=False,
                   random_state=random_state)

    clf.fit(X_train, y_train)
    y_test_predict = clf.predict(X_test)
    y_test_predict_proba = clf.predict_proba(X_test)
    evaluate_print('Stacking | ', y_test, y_test_predict)
