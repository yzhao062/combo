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

from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from combo.models.cluster_comb import clusterer_ensemble
from combo.utils.data import evaluate_print
from combo.utils.utility import generate_bagging_indices

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Define data file and read X and y
    random_state = 42
    X, y = load_breast_cancer(return_X_y=True)
    X_norm = StandardScaler().fit_transform(X)
    n_samples = X.shape[0]
    n_features = X.shape[1]

    n_clusters = 5
    n_ite = 10

    original_results = np.zeros([n_samples, n_ite])

    for ite in range(n_ite):
        print("build cluster... ite", ite + 1, "...")
        # random_state = np.random.RandomState(random_state_seed.tomaxint())
        random_state = np.random.RandomState(ite)
        # randomly generate feature subspaces
        sub_features = generate_bagging_indices(
            random_state=random_state,
            bootstrap_features=False,
            n_features=n_features,
            min_features=n_features * 0.5,
            max_features=n_features)

        X_sub = X_norm[:, sub_features]
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
            X_sub)
        original_results[:, ite] = kmeans.labels_

    aligned_results = clusterer_ensemble(original_results, n_clusters, n_ite)
