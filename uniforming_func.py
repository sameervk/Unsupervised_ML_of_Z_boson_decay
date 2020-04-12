import numpy as np

def uniforming_func(scaled_traindata=np.array([]),predicted_values=np.array([]),n_clusters=2):

    """
    If the number of combinations in certain clusters is less than that of the cluster which contains all the combinations,
    then this function modifies the clusters with subset of combinations by adding zeroes to the cluster counts and
    combinations at the respective missing indices and by comparing to the superset cluster and returns the cluster counts
    and the cluster combinations.

    E.g: if the results of clustering with 5 clusters contains [16,16,12,16,16] charge/type combinations with number of
    samples [x1,x2,x3,x4,x5], then this function finds the missing combinations in the 12 length cluster and returns
     list with the 16 combinations and a 16 length cluster with zeroes added in the locations of the missing combinations.


    :param scaled_traindata: the training data, type: array
    :param predicted_values: predicted cluster index of the training data, type: array
    :param n_clusters: no. of clusters used in the training, type@ int
    :return: counts_list, type list
            unique_elems, type: list of unique combinations of charge and type of the 2 particles
    """

    unique_elems = [list(np.unique(scaled_traindata[:, 16:][predicted_values ==i], axis=0, return_counts=True))
                    for i in range(n_clusters)]
    # need to convert to list as np.unique returns tuple which is immutable

    type_counts = [len(i[1]) for i in unique_elems]

    type_counts_len, type_counts_idx = np.unique(type_counts, return_index=True)

    if type_counts_len.size == 1:
        return 'All clusters have the same number of particle charge and type combinations.', False

    for i, counts in zip(type_counts_idx[0:-1], type_counts_len):
        relative_indices = [np.argwhere(np.all(unique_elems[type_counts_idx[-1]][0] ==
                                               unique_elems[i][0][j], axis=1)).ravel().tolist()[0] for j in range(counts)]

        mask_arr = np.isin(np.arange(16), relative_indices)

        temp_elems = np.zeros(shape=unique_elems[type_counts_idx[-1]][0].shape,
                              dtype=unique_elems[type_counts_idx[-1]][0].dtype)
        temp_elems[mask_arr] = unique_elems[i][0]
        missing_idx_mask = np.logical_not(mask_arr)
        temp_elems[missing_idx_mask] = unique_elems[type_counts_idx[-1]][0][missing_idx_mask]

        temp_counts = np.zeros(shape=(16,), dtype=unique_elems[type_counts_idx[-1]][1].dtype)
        temp_counts[mask_arr] = unique_elems[i][1]

        unique_elems[i][0] = temp_elems
        unique_elems[i][1] = temp_counts


    count_list = [i[1].tolist() for i in unique_elems]
    return count_list, unique_elems