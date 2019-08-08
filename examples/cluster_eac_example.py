# -*- coding: utf-8 -*-
"""Example of Combining multiple clusterings using evidence accumulation (EAC).
Part of the code is adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause


import os
import sys

# temporary solution for relative imports in case combo is not installed
# if combo is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets

from combo.models.cluster_eac import EAC
from combo.utils.example import visualize_clusters

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    random_state = 42

    n_clusters = 3
    n_estimators = 3
    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 1500
    X, y = datasets.make_moons(n_samples=n_samples, noise=.05)

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
    visualize_clusters('ECA Clustering', X, predicted_labels, show_figure=True,
                       save_figure=False)
