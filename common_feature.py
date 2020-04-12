import numpy as np
from decode_fn_v2 import decode_fn_v2
from functools import reduce


categories_dict = dict()
categories_dict.update(zip(range(4), ['q1','q2','type1','type2']))


def common_feature(scaled_data=np.array([]), predicted_values=np.array([]), n_clusters=2):
    """
    If a cluster has combinations in which a certain feature is unique among all the samples, then this function
    returns that common feature.

    E.g.: if the samples in the cluster have all the 1st particles with charge -1 and type 'EE', then this function
    returns [['q1', '-1'],['type1', 'EE']]
    :param scaled_data: the training data. array
    :param predicted_values: cluster labels of the data points
    :param n_clusters: number of clusters resulting from the clustering algorithm
    :return: list of lists.
    """


    unique_labels = [np.unique(scaled_data[:, 16:][predicted_values == i], axis=0) for i in range(n_clusters)]

    common_elems = [[reduce(np.intersect1d, [i for i in j]).tolist() for j in decode_fn_v2(label).T] for label in
                    unique_labels]

    common_elems = [list(map(list, filter(lambda j: j[0] != [], zip(cluster, range(len(cluster))))))
                    for cluster in common_elems]
    # returns list of empty lists if no common element found

    if common_elems.count([]):
        return 'No common particle charge/type combination within the clusters'

    return [[[categories_dict[j[1]], j[0][0]] for j in i] for i in common_elems]