from sklearn.decomposition import PCA

import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D



def plot_func(data, predicted_values, figsize = (10,6), text = 'Title', text_loc = (0.1,0.3)):
    
    # pca = 2
    reduced_2d = PCA(n_components=2, whiten=True).fit_transform(data)
    
    # pca = 3
    reduced_3d = PCA(n_components=3, whiten=True).fit_transform(data)
    
    cluster_labels = np.unique(predicted_values)
    
    fig = plt.figure(figsize=figsize)
    gridspec.GridSpec(2,3) # for plotting subplots of different sizes
    
    # 2D plot
    # ax2d = fig.add_subplot(121)    
    ax2d = plt.subplot2grid((2,3), (0,0))
    for i in cluster_labels:
        ax2d.scatter(reduced_2d[predicted_values==i,0],reduced_2d[predicted_values==i,1])
    ax2d.set_xlabel('PCA comp. 1')
    ax2d.set_ylabel('PCA comp. 2')

        
    # 3D plot
    # ax3d = fig.add_subplot(122, projection = '3d')    
    ax3d = plt.subplot2grid((2,3), (0,1), rowspan=2, colspan=2, projection = '3d')
    # spanning the dimensions of 2 rows and 2 columns
    for i in cluster_labels:
        ax3d.scatter(reduced_3d[predicted_values==i,0],reduced_3d[predicted_values==i,1], reduced_3d[predicted_values==i,2])
    
    ax3d.set_xlabel('PCA comp. 1')
    ax3d.set_ylabel('PCA comp. 2')
    ax3d.set_zlabel('PCA comp. 3')
    
    text_x, text_y = text_loc
    fig.text(x = text_x, y=text_y, s= text, fontsize = 16)
    fig.tight_layout()
    
    plt.show()    