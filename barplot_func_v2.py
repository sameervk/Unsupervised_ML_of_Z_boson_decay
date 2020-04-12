import numpy as np
from uniforming_func import  uniforming_func
from decode_fn_v2 import decode_fn_v2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import OneHotEncoder

def barplot_func_v2(training_data, pred_val, n_clusters=2, encoder_charge = OneHotEncoder(), encoder_type = OneHotEncoder(),
                    figsize=(12, 8), bar_width=0.25, textloc=(0.99, 0.2), title='Clustering', title_loc=(0.5, 1)):
    """
    Bar plot of the number of samples belonging to each combination in the clusters.
    A total of 16 x n_clusters bars will be plotted.
    This is for checking any anomalies in the distribution of the samples amongst the clusters.

    :param training_data: scaled training data. ndarray
    :param pred_val: cluster labels predicting by the clustering algorithm. array
    :param n_clusters: no. of clusters from the algorithm. int
    :param encoder_charge: onehotencoder of charge [-1,1]
    :param encoder_type: onehotencoder of type [EB, EE]
    :param figsize: dimensions of the bar plot figure. tuple of floats
    :param bar_width: float
    :param textloc: text location to display the combination value. tuple
    :param title: title of the figure. string
    :param title_loc: title location. tuple of floats
    :return: matplotlib figure
    """


    unique_elems = [list(np.unique(training_data[:, 16:][pred_val == i], axis=0, return_counts=True))
                    for i in range(n_clusters)]
    # need to convert to list as np.unique returns tuple which is immutable

    type_counts = [len(i[1]) for i in unique_elems]

    type_counts_len, type_counts_idx = np.unique(type_counts, return_index=True)

    if type_counts_len.size == 1:
        #         cluster_labels_counts = [np.unique(scaled_traindata[:,16:][pred_val==i], axis=0, return_counts=True) for i in range(n_clusters)]
        counts = [i[1] for i in unique_elems]

    else:
        counts, unique_elems = uniforming_func(training_data, pred_val, n_clusters)

    cluster_features = decode_fn_v2(encoder_charge, encoder_type, np.array(unique_elems[0][0])).tolist()

    for i in cluster_features:
        i.insert(0, str(cluster_features.index(i)) + ': ')

    for j in cluster_features:
        j[-1] = j[-1] + '\n'

    fig_text = ''
    for s in np.array(cluster_features).flatten():
        fig_text = fig_text + s

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    bar_locations = [np.arange(-(n_clusters - 1) / 2 * bar_width, stop=(n_clusters + 1) / 2 * bar_width,
                               step=bar_width) if n_clusters % 2 == 1 else
                     np.arange(-(n_clusters - 1) / 2 * bar_width, stop=n_clusters / 2 * bar_width, step=bar_width)][0]

    cluster_labels = np.arange(0, 16)

    for i in range(n_clusters):
        color = cm.jet(i / n_clusters)
        ax.bar(cluster_labels + bar_locations[i], counts[i], color=color, width=bar_width, label='cluster %d' % (i))

    ax.legend(loc='best')
    ax.set_xticks(cluster_labels)

    ax.set_xlabel('Combination')
    ax.set_ylabel('No. of data samples in the cluster')

    fig.suptitle(x=title_loc[0], y=title_loc[1], t=title)
    fig.text(x=textloc[0], y=textloc[1], s=fig_text)
    fig.tight_layout()
    plt.show()