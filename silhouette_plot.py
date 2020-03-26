import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score

def silhouette_plot(data, predicted_values, title = 'title'):
    
    sil_values = silhouette_samples(data, predicted_values)
    cluster_labels = np.unique(predicted_values)
    
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    
    for i, c in enumerate(cluster_labels):
        c_sil_vals = sil_values[predicted_values == c]
        c_sil_vals.sort()

        y_ax_upper += len(c_sil_vals)

        color =  cm.jet(i/len(cluster_labels))
        plt.barh(range(y_ax_lower, y_ax_upper), c_sil_vals, height=1.0, edgecolor=None, color=color)
        yticks.append((y_ax_lower+y_ax_upper)/2)
        y_ax_lower += len(c_sil_vals)
    sil_values_mean = np.mean(sil_values)
    plt.axvline(sil_values_mean, color = 'red', linestyle='--')
    plt.yticks(yticks, labels = cluster_labels+1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette Coefficient')
    plt.show()