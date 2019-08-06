# -*- coding: utf-8 -*-
"""Example of Combining multiple clusterings using evidence accumulation (EAC).
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

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.datasets import load_breast_cancer

from combo.models.cluster_eac import EAC

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Define data file and read X and y
    X, y = load_breast_cancer(return_X_y=True)

    n_clusters = 5
    n_estimators = 3

    # Initialize a set of estimators
    estimators = [KMeans(n_clusters=n_clusters),
                  MiniBatchKMeans(n_clusters=n_clusters),
                  AgglomerativeClustering(n_clusters=n_clusters)]

    clf = EAC(estimators, n_clusters=n_clusters)
    clf.fit(X)

    # generate the labels on X
    predicted_labels = clf.labels_

    # generate the labels on X
    predicted_labels = clf.fit_predict(X)



