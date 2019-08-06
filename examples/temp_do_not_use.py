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

from sklearn.datasets import load_breast_cancer

from combo.models.cluster_comb import clusterer_ensemble_scores
from combo.models.cluster_comb import ClustererEnsemble
from scipy.cluster.hierarchy import complete, fcluster

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

    clf = ClustererEnsemble(estimators, n_clusters=n_clusters)
    clf.fit(X)

    # generate the labels on X
    aligned_labels = clf.aligned_labels_
    predicted_labels = clf.labels_

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
#%%
    c_mat = np.array([[0, 1, 7, 8],[1, 0, 3, 9],[7, 3, 0, 10],[8, 9, 10, 0]])
    Z = complete(c_mat)
    
    k=3
    fcluster(Z, k, criterion='maxclust')
#%%
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage
    from scipy.cluster import hierarchy
    
    
    n_samples = X.shape[0]
    def generate_similarity_mat(labels):
        
#        labels = column_or_1d(labels)
        l_mat = np.repeat(labels, len(labels), axis=1)
        l_mat_t = l_mat.T
        sim_mat = np.equal(l_mat, l_mat_t).astype(int)
        return sim_mat
    
    sim_mat_all = np.zeros([n_samples, n_samples])
    
    for i, clf in enumerate(estimators):
        
        clf.fit(X)
        labels = clf.labels_.reshape(n_samples, 1)
        sim_mat = generate_similarity_mat(labels)
        
        sim_mat_all = sim_mat_all+sim_mat
    
    # get the average of the similarity
    sim_mat_avg = np.divide(sim_mat_all, n_estimators)
    
    # flip the similarity. smaller value implies more similarity
    sim_mat_avg = np.abs(np.max(sim_mat_avg)-sim_mat_avg)
    
    # build clusters
    Z = linkage(sim_mat_avg, method='single')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    n_clusters_build = np.unique(labels)
    
    
    
    
    
    
    
