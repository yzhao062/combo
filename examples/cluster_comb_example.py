# -*- coding: utf-8 -*-
"""Example of combining multiple clustering algorithm. The example uses
Clusterer Ensemble by Zhi-hua Zhou, 2006.
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
from sklearn import datasets

from combo.models.cluster_comb import clusterer_ensemble_scores
from combo.models.cluster_comb import ClustererEnsemble
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

    clf = ClustererEnsemble(estimators, n_clusters=n_clusters)
    clf.fit(X)

    # generate the labels on X
    aligned_labels = clf.aligned_labels_
    predicted_labels = clf.labels_

    visualize_clusters('Clusterer Ensemble', X, predicted_labels,
                       show_figure=True, save_figure=False)

    # Clusterer Ensemble without initializing a new Class
    original_labels = np.zeros([X.shape[0], n_estimators])

    for i, estimator in enumerate(estimators):
        estimator.fit(X)
        original_labels[:, i] = estimator.labels_

    # Invoke method directly without initializing a new Class
    # Demo the effect of different parameters
    labels_by_vote1 = clusterer_ensemble_scores(original_labels, n_estimators,
                                                n_clusters)
    # return aligned_labels as well
    labels_by_vote2, aligned_labels = clusterer_ensemble_scores(
        original_labels, n_estimators, n_clusters, return_results=True)

    # select a different reference base estimator (default is 0)
    labels_by_vote3 = clusterer_ensemble_scores(original_labels, n_estimators,
                                                n_clusters, reference_idx=1)
