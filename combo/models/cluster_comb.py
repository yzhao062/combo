# -*- coding: utf-8 -*-
"""A collection of combination methods for clustering
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import os
import numpy as np
from sklearn.utils import check_array


def intersection_mat(result_mat, first_idx, second_idx, n_clusters):
    '''Calculate the number of overlappings of second_idx based on first_idx.
    alignment_mat[i,j] represents the number of labels == j in second_idx
    when labels == i in the first idx.

    In other words, we should do the alignment based on the max by first
    assigning the most

    Parameters
    ----------
    result_mat
    first_idx
    second_idx
    n_clusters

    Returns
    -------

    '''
    alignment_mat = np.zeros([n_clusters, n_clusters])
    for i in range(n_clusters):
        for j in range(n_clusters):
            i_index = np.argwhere(result_mat[:, first_idx] == i)
            j_index = np.argwhere(result_mat[:, second_idx] == j)
            inter_ij = np.intersect1d(i_index, j_index)
            alignment_mat[i, j] = len(inter_ij)

    return alignment_mat


def alignment(inter_mat, n_clusters, second_idx, result_mat_aligned,
              offset=100000):
    index_mapping = np.zeros([n_clusters, 2])
    index_mapping[:, 0] = list(range(0, n_clusters))

    while np.sum(inter_mat) > (-1 * n_clusters * n_clusters):
        max_i, max_j = np.unravel_index(inter_mat.argmax(), inter_mat.shape)
        index_mapping[max_i, 1] = max_j
        inter_mat[max_i, :] = -1
        inter_mat[:, max_j] = -1

        #  print('component 1 cluser', max_i, '==', 'component 2 cluser', max_j)
        result_mat_aligned[np.where(result_mat_aligned[:,
                                    second_idx] == max_j), second_idx] = max_i + offset
    return index_mapping


def validate_cluster_number(original_results, n_clusters):
    original_results = check_array(original_results)

    for i in range(original_results.shape[1]):

        values, counts = np.unique(original_results[:, i], return_counts=True)
        if len(values) != n_clusters:
            print(len(values), len(counts))
            RuntimeError('cluster result does not equal to n_clusters')

    return original_results


def clusterer_ensemble(original_results, n_clusters, n_ite, selected_idx=0):
    '''
    Parameters
    ----------
    original_results
    selected_idx
    n_clusters
    n_ite

    Returns
    -------

    '''

    original_results = validate_cluster_number(original_results, n_clusters)
    alignment_mat = np.zeros([n_clusters, n_ite])
    result_mat_aligned = np.copy(original_results)
    offset = 100000  # fixed offset offset. DO NOT CHANGE.

    for i in range(n_ite):
        inter_mat = intersection_mat(original_results, selected_idx, i,
                                     n_clusters)
        index_mapping = alignment(inter_mat, n_clusters, i,
                                  result_mat_aligned, offset)
        alignment_mat[:, i] = index_mapping[:, 1]

    result_mat_aligned = result_mat_aligned - offset
    return result_mat_aligned
